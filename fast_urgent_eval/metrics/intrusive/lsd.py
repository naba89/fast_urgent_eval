import  torch
from torch import nn


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
        ref_stft = torch.stft(ref, n_fft=nfft, hop_length=hop, window=window, return_complex=True)
        inf_stft = torch.stft(inf, n_fft=nfft, hop_length=hop, window=window, return_complex=True)

        # Convert to magnitude and transpose
        ref_spec = ref_stft.abs().T
        inf_spec = inf_stft.abs().T

        # Calculate log-spectral distance
        lsd = torch.log(ref_spec ** 2 / ((inf_spec + self.eps) ** 2) + self.eps) ** self.p
        lsd = (lsd.mean(dim=1) ** (1.0 / self.p)).mean(dim=0)

        return lsd.item()
