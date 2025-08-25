import librosa
import numpy as np
import  torch
from torch import nn


def lsd_metric(ref, inf, fs, nfft=0.032, hop=0.016, p=2, eps=1.0e-08):
    """Calculate Log-Spectral Distance (LSD).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
        nfft (float): FFT length in seconds
        hop (float): hop length in seconds
        p (float): the order of norm
        eps (float): epsilon value for numerical stability
    Returns:
        mcd (float): LSD value between [0, +inf)
    """
    scaling_factor = np.sum(ref * inf) / (np.sum(inf**2) + eps)
    inf = inf * scaling_factor

    nfft = int(fs * nfft)
    hop = int(fs * hop)
    # T x F
    ref_spec = np.abs(librosa.stft(ref, hop_length=hop, n_fft=nfft)).T
    inf_spec = np.abs(librosa.stft(inf, hop_length=hop, n_fft=nfft)).T
    lsd = np.log(ref_spec**2 / ((inf_spec + eps) ** 2) + eps) ** p
    lsd = np.mean(np.mean(lsd, axis=1) ** (1 / p), axis=0)
    return lsd


class LSDMetric(nn.Module):
    """Log-Spectral Distance (LSD) metric module."""

    def __init__(self, nfft=0.032, hop=0.016, p=2, eps=1.0e-08):
        super(LSDMetric, self).__init__()
        self.nfft = nfft
        self.hop = hop
        self.p = p
        self.eps = eps
        self.valid_srs = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
        for sr in self.valid_srs:
            self.register_buffer("window_{}".format(sr), torch.hann_window(int(sr * nfft)))

    def forward(self, ref: torch.Tensor, inf: torch.Tensor, fs: int, **kwargs) -> float:
        """Calculate Log-Spectral Distance (LSD).

        Args:
            ref (torch.Tensor): reference signal (time,)
            inf (torch.Tensor): enhanced signal (time,)
            fs (int): sampling rate in Hz
        Returns:
            mcd (float): LSD value between [0, +inf)
        """
        assert ref.shape == inf.shape, "Reference and enhanced signals must have the same shape."
        assert fs in self.valid_srs, f"Sampling rate {fs} is not supported. Valid rates are {self.valid_srs}."

        scaling_factor = torch.sum(ref * inf) / (torch.sum(inf ** 2) + self.eps)
        inf = inf * scaling_factor

        nfft = int(fs * self.nfft)
        hop = int(fs * self.hop)

        # Compute complex STFT
        window = getattr(self, f"window_{fs}", None)
        ref_stft = torch.stft(ref, n_fft=nfft, hop_length=hop, window=window, return_complex=True, pad_mode="constant")
        inf_stft = torch.stft(inf, n_fft=nfft, hop_length=hop, window=window, return_complex=True, pad_mode="constant")

        # Convert to magnitude and transpose
        ref_spec = ref_stft.abs().transpose(-1, -2)
        inf_spec = inf_stft.abs().transpose(-1, -2)

        # Calculate log-spectral distance
        lsd = torch.log(ref_spec ** 2 / ((inf_spec + self.eps) ** 2) + self.eps) ** self.p
        lsd = (lsd.mean(dim=1) ** (1.0 / self.p)).mean(dim=0)

        return lsd.item()


if __name__ == "__main__":
    # verify the implementation
    fs = 16000
    duration = 3  # seconds

    ref_tensor = torch.randn(duration * fs, dtype=torch.float64)
    inf_tensor = ref_tensor + 0.05 * torch.randn(duration * fs, dtype=torch.float64)

    lsd_value = lsd_metric(ref_tensor.numpy(), inf_tensor.numpy(), fs)
    print(f"LSD (numpy): {lsd_value}")

    lsd_module = LSDMetric()
    lsd_value_module = lsd_module(ref_tensor, inf_tensor, fs)
    print(f"LSD (module): {lsd_value_module}")

    print("Difference:", abs(lsd_value - lsd_value_module))
    assert abs(lsd_value - lsd_value_module) < 1e-5, "The implementations do not match!"
    print("The implementations match!")