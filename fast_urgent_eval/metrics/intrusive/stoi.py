import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pystoi import stoi

# =========================
# Constants (match reference)
# =========================
FS = 10_000                       # Target internal sampling rate
N_FRAME = 256                     # Window size
NFFT = 512                        # FFT size
NUMBAND = 15                      # Number of 1/3 octave bands
MINFREQ = 150                     # Center frequency of first band (Hz)
N = 30                            # Number of frames per segment (STOI)
BETA = -15.0                      # Lower SDR bound (dB)
DYN_RANGE = 40                    # Energy dynamic range for silence trimming (dB)

# Use float64 everywhere to best match NumPy/SciPy
_DEFAULT_DTYPE = torch.float32


# =========================
# Utilities
# =========================
def _eps(dtype=_DEFAULT_DTYPE) -> float:
    return torch.finfo(dtype).eps


def _hanning_like_matlab(win: int, *, dtype=_DEFAULT_DTYPE, device=None) -> torch.Tensor:
    # Matches np.hanning(win+2)[1:-1]
    w = torch.hann_window(win + 2, periodic=False, dtype=dtype, device=device)
    return w[1:-1]


# =========================
# 1/3 Octave bands
# =========================
@torch.no_grad()
def thirdoct_torch(fs: int, nfft: int, num_bands: int, min_freq: float,
                   *, dtype=_DEFAULT_DTYPE, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (OBM, CF) as torch tensors (match your numpy version).
    """
    f = torch.linspace(0.0, float(fs), nfft + 1, dtype=dtype, device=device)
    f = f[: (nfft // 2) + 1]

    k = torch.arange(num_bands, dtype=dtype, device=device)
    cf = (2.0 ** (1.0 / 3.0)) ** k * min_freq

    freq_low = min_freq * (2.0 ** ((2 * k - 1) / 6.0))
    freq_high = min_freq * (2.0 ** ((2 * k + 1) / 6.0))

    obm = torch.zeros((num_bands, f.numel()), dtype=dtype, device=device)

    # Match nearest bins exactly like your argmin(squared diff)
    for i in range(num_bands):
        fl_idx = torch.argmin((f - freq_low[i]).abs() ** 2).item()
        fh_idx = torch.argmin((f - freq_high[i]).abs() ** 2).item()
        # Assign band = 1 between indices
        if fh_idx > fl_idx:
            obm[i, fl_idx:fh_idx] = 1.0
        # else: empty (no-op)
    return obm, cf


# =========================
# STFT (match frame indexing & windowing)
# =========================
def stft_torch(x: torch.Tensor, win_size: int, fft_size: int, *, overlap: int = 4,
               dtype=_DEFAULT_DTYPE, device=None) -> torch.Tensor:
    """
    x: 1D tensor (time domain signal)
    Returns 2D complex tensor (freqs, frames), matching:
      hop = win_size / overlap
      frames at starts in range(0, len(x)-win_size, hop)  [stop BEFORE last possible]
      window = np.hanning(win+2)[1:-1]
    """
    hop = int(win_size / overlap)
    w = _hanning_like_matlab(win_size, dtype=dtype, device=device)

    starts = list(range(0, x.numel() - win_size, hop))  # stop BEFORE last frame (matches your code)
    if len(starts) == 0:
        return torch.empty((fft_size // 2 + 1, 0), dtype=torch.complex128 if dtype==torch.float64 else torch.complex64, device=device)

    frames = []
    for s in starts:
        seg = x[s:s + win_size] * w
        # zero-pad or truncate to fft_size inside rfft
        X = torch.fft.rfft(seg, n=fft_size)
        frames.append(X)
    Xmat = torch.stack(frames, dim=1)  # (freqs, frames)
    return Xmat


# =========================
# Overlap & Add (used after silence removal)
# =========================
def overlap_and_add_torch(x_frames: torch.Tensor, hop: int, *, dtype=_DEFAULT_DTYPE, device=None) -> torch.Tensor:
    """
    x_frames: (num_frames, framelen)
    Deterministic left-to-right OLA (simple adds), stable and reproducible.
    """
    x_frames = x_frames.to(device=device, dtype=dtype)
    num_frames, framelen = x_frames.shape
    end = (num_frames - 1) * hop + framelen
    out = torch.zeros(end, dtype=dtype, device=device)
    pos = 0
    for i in range(num_frames):
        out[pos:pos + framelen] += x_frames[i]
        pos += hop
    return out


# =========================
# Silence removal
# =========================
def remove_silent_frames_torch(x: torch.Tensor, y: torch.Tensor, dyn_range: float, framelen: int, hop: int,
                               *, dtype=_DEFAULT_DTYPE, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param x:  1-D tensor (time domain signal)
    :param y:  1-D tensor (time domain signal, same length as x)
    :param dyn_range:
    :param framelen:
    :param hop:
    :param dtype:
    :param device:
    :return:
    """

    w = _hanning_like_matlab(framelen, dtype=dtype, device=device)

    starts = list(range(0, x.numel() - framelen, hop))  # stop BEFORE last (matches your code)
    if len(starts) == 0:
        return torch.empty(0, dtype=dtype, device=device), torch.empty(0, dtype=dtype, device=device)

    x_frames = torch.stack([x[s:s + framelen] * w for s in starts], dim=0)  # (F, L)
    y_frames = torch.stack([y[s:s + framelen] * w for s in starts], dim=0)

    # Energies in dB
    energies = 20.0 * torch.log10(torch.linalg.vector_norm(x_frames, dim=1) + _eps(dtype))

    # Keep frames where (maxE - dyn_range - E) < 0  -> i.e., E > maxE - dyn_range
    mask = (energies > (energies.max() - dyn_range))

    x_kept = x_frames[mask]
    y_kept = y_frames[mask]

    x_sil = overlap_and_add_torch(x_kept, hop, dtype=dtype, device=device) if x_kept.numel() > 0 else torch.empty(0, dtype=dtype, device=device)
    y_sil = overlap_and_add_torch(y_kept, hop, dtype=dtype, device=device) if y_kept.numel() > 0 else torch.empty(0, dtype=dtype, device=device)
    return x_sil, y_sil


# =========================
# Helpers for extended STOI
# =========================
def vect_two_norm_torch(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.sum(x * x, dim=dim, keepdim=True)


def row_col_normalize_torch(x: torch.Tensor, *, dtype=_DEFAULT_DTYPE, device=None) -> torch.Tensor:
    x = x.to(device=device, dtype=dtype)

    # Make a reproducible RNG (so results are deterministic run-to-run)
    # Feel free to pick another fixed seed if you like.
    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    # Row normalization (over the last dim)
    noise = torch.normal(mean=0.0, std=float(_eps(dtype)), size=x.shape, generator=gen, device=device, dtype=dtype)
    x_normed = x + noise
    x_normed -= x_normed.mean(dim=-1, keepdim=True)
    inv = 1.0 / (torch.sqrt(torch.sum(x_normed * x_normed, dim=-1, keepdim=True)) + _eps(dtype))
    x_normed = x_normed * inv

    # Column normalization (over band dim = 1)
    noise2 = torch.normal(mean=0.0, std=float(_eps(dtype)), size=x_normed.shape, generator=gen, device=device, dtype=dtype)
    x_normed = x_normed + noise2
    x_normed -= x_normed.mean(dim=1, keepdim=True)
    inv2 = 1.0 / (torch.sqrt(torch.sum(x_normed * x_normed, dim=1, keepdim=True)) + _eps(dtype))
    x_normed = x_normed * inv2
    return x_normed



# =========================
# STOI (main)
# =========================
class STOI(torch.nn.Module):
    def __init__(self,
                 fs_internal: int = FS,
                 n_frame: int = N_FRAME,
                 nfft: int = NFFT,
                 numband: int = NUMBAND,
                 minfreq: float = MINFREQ,
                 n_ctx_frames: int = N,
                 beta_db: float = BETA,
                 dyn_range_db: float = DYN_RANGE,
                 dtype=_DEFAULT_DTYPE,
                 device=None):
        super().__init__()
        self.fs_internal = fs_internal
        self.n_frame = n_frame
        self.nfft = nfft
        self.numband = numband
        self.minfreq = minfreq
        self.n_ctx_frames = n_ctx_frames
        self.beta_db = beta_db
        self.dyn_range_db = dyn_range_db
        self.dtype = dtype

        obm, cf = thirdoct_torch(fs_internal, nfft, numband, minfreq, dtype=dtype, device=device)
        # Register as buffers for device/dtype tracking
        self.register_buffer("OBM", obm, persistent=False)
        self.register_buffer("CF", cf, persistent=False)

    @torch.no_grad()
    def forward(self, ref: torch.Tensor, inf: torch.Tensor, fs: int, extended: bool = False, **kwargs):
        """
        x, y: 1-D tensors (same length), dtype float64 preferred for bit equivalence
        fs_sig: original sampling rate of x,y
        Returns: scalar STOI score
        """
        device = ref.device
        assert fs == self.fs_internal, f"Expected fs_sig={self.fs_internal}, got {fs}"
        assert ref.ndim == 1 and inf.ndim == 1, f"Expected 1-D tensors, got {ref.ndim}D and {inf.ndim}D"

        if ref.shape != inf.shape:
            raise ValueError(f"x and y must have the same shape, got {ref.shape} and {inf.shape}")

        # Remove silent frames
        x_sil, y_sil = remove_silent_frames_torch(ref, inf, self.dyn_range_db, self.n_frame, self.n_frame // 2,
                                                  dtype=self.dtype, device=device)

        # STFT (overlap=2 in your code)
        X = stft_torch(x_sil, self.n_frame, self.nfft, overlap=2, dtype=self.dtype, device=device)  # (F, T)
        Y = stft_torch(y_sil, self.n_frame, self.nfft, overlap=2, dtype=self.dtype, device=device)

        if X.shape[-1] < self.n_ctx_frames:
            # Same warning behavior as your code (but here we return tensor)
            return torch.tensor(1e-5, dtype=self.dtype, device=ref.device)

        # Third-octave band energies (Eq.1 in paper)
        X_tob = torch.sqrt(self.OBM @ (torch.abs(X) ** 2))
        Y_tob = torch.sqrt(self.OBM @ (torch.abs(Y) ** 2))

        # Build segments of length N across time (sliding end index m)
        # segments: shape (J, bands, N), where J = T - N + 1
        T = X_tob.shape[1]

        X_segs = torch.stack([X_tob[:, m - self.n_ctx_frames:m] for m in range(self.n_ctx_frames, T + 1)], dim=0)  # (J, B, N)
        Y_segs = torch.stack([Y_tob[:, m - self.n_ctx_frames:m] for m in range(self.n_ctx_frames, T + 1)], dim=0)

        if extended:
            Xn = row_col_normalize_torch(X_segs, dtype=self.dtype, device=device)
            Yn = row_col_normalize_torch(Y_segs, dtype=self.dtype, device=device)
            # sum(Xn * Yn / N) / bands
            d = torch.sum(Xn * Yn / self.n_ctx_frames) / Xn.shape[0]
            return d

        # Normalization constants & normalization
        X_norm = torch.linalg.vector_norm(X_segs, dim=2, keepdim=True)
        Y_norm = torch.linalg.vector_norm(Y_segs, dim=2, keepdim=True)
        norm_consts = X_norm / (Y_norm + _eps(self.dtype))
        Y_n = Y_segs * norm_consts

        # Clipping
        clip_value = 10.0 ** (-self.beta_db / 20.0)
        Y_prime = torch.minimum(Y_n, X_segs * (1.0 + clip_value))

        # Mean subtraction
        Y_prime = Y_prime - Y_prime.mean(dim=2, keepdim=True)
        Xc = X_segs - X_segs.mean(dim=2, keepdim=True)

        # L2 normalization
        Y_prime = Y_prime / (torch.linalg.vector_norm(Y_prime, dim=2, keepdim=True) + _eps(self.dtype))
        Xc = Xc / (torch.linalg.vector_norm(Xc, dim=2, keepdim=True) + _eps(self.dtype))

        # Correlation components and mean
        corr = Y_prime * Xc
        J_ = Xc.shape[0]
        M_ = Xc.shape[1]
        d = torch.sum(corr) / (J_ * M_)
        return d.item()


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Dummy example (white noise vs. same)
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(16000, dtype=_DEFAULT_DTYPE)  # 1.6 s at 10 kHz or any fs_sig you pass
    y = torch.randn(16000, dtype=_DEFAULT_DTYPE)  # Same length, same fs_sig

    stoi_torch = STOI(dtype=_DEFAULT_DTYPE, device="cpu")
    score = stoi_torch(x, y, fs_sig=10000, extended=True)
    print("STOI:", float(score))

    orig_score = stoi(x.numpy(), y.numpy(), fs_sig=10000, extended=True)
    print("Original STOI (pystoi):", orig_score)

    assert math.isclose(float(score), orig_score, rel_tol=1e-5), "STOI scores do not match!"
    print("STOI scores match!")
