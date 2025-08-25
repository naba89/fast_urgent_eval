# Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics

"""Evaluate MCD between generated and groundtruth audios with diff-SPTK-based mcep."""
from typing import Tuple

import diffsptk
import numpy as np
import torch
from fastdtw import fastdtw
from scipy import spatial
from torch import nn


def _get_best_mcep_params(fs: int) -> Tuple[int, float]:
    # https://sp-nitech.github.io/sptk/latest/main/mgcep.html#_CPPv4N4sptk19MelCepstralAnalysisE
    if fs == 8000:
        return 13, 0.31
    elif fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 32000:
        return 36, 0.50
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


class MCDMetric(nn.Module):
    """Mel Cepstral Distortion (MCD) metric module."""

    def __init__(self, n_fft=1024, n_shift=256):
        super().__init__()
        self.n_fft = n_fft
        self.n_shift = n_shift

        self.stft = diffsptk.STFT(
            frame_length=n_fft,
            frame_period=n_shift,
            fft_length=n_fft,
            window="hamming",
            center=False,
            eps=1e-6,
            dtype=torch.float64,  # Use float64 for precision
        )
        self.valid_sr = [8000, 16000, 22050, 24000, 32000, 44100, 48000]
        self.mcep_fncs = nn.ModuleDict()
        for fs in self.valid_sr:
            mcep_dim, mcep_alpha = _get_best_mcep_params(fs)
            self.mcep_fncs[str(fs)] = diffsptk.MelCepstralAnalysis(
                cep_order=mcep_dim,
                fft_length=n_fft,
                alpha=mcep_alpha,
                n_iter=6,  # gives closer results atol around 1e-4
                dtype=torch.float64,  # Use float64 for precision
            )

    @torch.inference_mode()
    def forward(self, ref: torch.Tensor, inf: torch.Tensor, fs: int, **kwargs) -> float:
        """Calculate Mel Cepstral Distortion (MCD).

        Args:
            ref (torch.Tensor): reference signal (time,)
            inf (torch.Tensor): enhanced signal (time,)
            fs (int): sampling rate in Hz

        Returns:
            mcd (float): MCD value between [0, +inf)
        """
        if fs not in self.valid_sr:
            raise ValueError(f"fs must be one of {self.valid_sr}.")
        if ref.ndim != 1 or inf.ndim != 1:
            raise ValueError("ref and inf must be 1D array.")
        if len(ref) < self.n_fft or len(inf) < self.n_fft:
            raise ValueError(f"ref and inf length must be larger than {self.n_fft}.")

        # extract features
        ref = ref.double()
        inf = inf.double()

        gen_mcep = self.mcep_fncs[str(fs)](self.stft(inf)).cpu().numpy()
        gt_mcep = self.mcep_fncs[str(fs)](self.stft(ref)).cpu().numpy()

        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        twf = np.array(path).T
        gen_mcep_dtw = gen_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD
        diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)

        return mcd
