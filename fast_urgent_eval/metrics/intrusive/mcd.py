#!/usr/bin/env python3

# Copyright 2020 Wen-Chin Huang and Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# ported from https://github.com/espnet/espnet/blob/master/utils/mcd_calculate.py

"""Evaluate MCD between generated and groundtruth audios with SPTK-based mcep."""
import time
from typing import Tuple

import diffsptk
import numpy as np
import pysptk
import torch
from fastdtw import fastdtw
from scipy import spatial
from torch import nn


def sptk_extract(
    x: np.ndarray,
    fs: int,
    n_fft: int = 512,
    n_shift: int = 256,
    mcep_dim: int = 25,
    mcep_alpha: float = 0.41,
    is_padding: bool = False,
) -> np.ndarray:
    """Extract SPTK-based mel-cepstrum.

    Args:
        x (ndarray): 1D waveform array.
        fs (int): Sampling rate
        n_fft (int): FFT length in point (default=512).
        n_shift (int): Shift length in point (default=256).
        mcep_dim (int): Dimension of mel-cepstrum (default=25).
        mcep_alpha (float): All pass filter coefficient (default=0.41).
        is_padding (bool): Whether to pad the end of signal (default=False).

    Returns:
        ndarray: Mel-cepstrum with the size (N, n_fft).

    """
    # perform padding
    if is_padding:
        n_pad = n_fft - (len(x) - n_fft) % n_shift
        x = np.pad(x, (0, n_pad), "reflect")

    # get number of frames
    n_frame = (len(x) - n_fft) // n_shift + 1

    # get window function
    win = pysptk.sptk.hamming(n_fft)

    # check mcep and alpha
    if mcep_dim is None or mcep_alpha is None:
        mcep_dim, mcep_alpha = _get_best_mcep_params(fs)

    # calculate spectrogram
    mcep = [
        pysptk.mcep(
            x[n_shift * i : n_shift * i + n_fft] * win,
            mcep_dim,
            mcep_alpha,
            eps=1e-6,
            etype=1,
        )
        for i in range(n_frame)
    ]

    return np.stack(mcep)


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


def calculate(
    inf_audio,
    ref_audio,
    fs,
    n_fft=1024,
    n_shift=256,
    mcep_dim=None,
    mcep_alpha=None,
):
    """Calculate MCD."""

    # extract ground truth and converted features
    # start_time= time.time()
    gen_mcep = sptk_extract(
        x=inf_audio,
        fs=fs,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )
    gt_mcep = sptk_extract(
        x=ref_audio,
        fs=fs,
        n_fft=n_fft,
        n_shift=n_shift,
        mcep_dim=mcep_dim,
        mcep_alpha=mcep_alpha,
    )
    # print("SPTK feature extraction time:", time.time() - start_time)
    # start_time = time.time()

    # DTW
    _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
    # print("DTW time:", time.time() - start_time)
    # start_time = time.time()
    twf = np.array(path).T
    gen_mcep_dtw = gen_mcep[twf[0]]
    gt_mcep_dtw = gt_mcep[twf[1]]

    # MCD
    diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
    # print("MCD calculation time:", time.time() - start_time)

    return mcd


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
        # start_time= time.time()
        gen_mcep = self.mcep_fncs[str(fs)](self.stft(inf)).cpu().numpy()
        gt_mcep = self.mcep_fncs[str(fs)](self.stft(ref)).cpu().numpy()
        # print("DiffSPTK feature extraction time:", time.time() - start_time)
        # start_time = time.time()
        # DTW
        _, path = fastdtw(gen_mcep, gt_mcep, dist=spatial.distance.euclidean)
        # print("DTW time:", time.time() - start_time)
        # start_time = time.time()
        twf = np.array(path).T
        gen_mcep_dtw = gen_mcep[twf[0]]
        gt_mcep_dtw = gt_mcep[twf[1]]

        # MCD
        diff2sum = np.sum((gen_mcep_dtw - gt_mcep_dtw) ** 2, 1)
        mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
        # print("MCD calculation time:", time.time() - start_time)

        return mcd


if __name__ == "__main__":
    fs = 44100
    # seed
    torch.manual_seed(0)
    np.random.seed(0)

    ref = torch.randn(2*fs)
    inf = ref + 0.01 * torch.randn(2*fs)

    mcd_metric = MCDMetric(n_fft=1024, n_shift=256)
    mcd = mcd_metric(ref, inf, fs)
    print(mcd)

    # compare with SPTK
    mcd_sptk = calculate(
        inf_audio=inf.numpy(),
        ref_audio=ref.numpy(),
        fs=fs,
        n_fft=1024,
        n_shift=256,
    )
    print(mcd_sptk)

    print("Difference:", abs(mcd - mcd_sptk))
    assert abs(mcd - mcd_sptk) < 1e-3, "The implementations do not match!"
    # Note that the results may differ slightly depending on some parameter mismatches
    # such as n_iter in MelCepstralAnalysis. But the difference should be small enough (<1e-3).

