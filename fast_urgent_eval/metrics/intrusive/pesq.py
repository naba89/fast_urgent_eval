# Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics

import logging
import numpy as np
from pesq import PesqError, pesq


def pesq_metric(ref, inf, fs=8000):
    """Calculate Perceptual Evaluation of Speech Quality (PESQ).

    Args:
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        pesq (float): PESQ value between [-0.5, 4.5]
    """
    assert ref.shape == inf.shape
    assert fs in [8000, 16000], "PESQ evaluation requires a sample rate of 8000 or 16000 Hz."
    if fs == 8000:
        mode = "nb"
    else:
        mode = "wb"
    pesq_score = pesq(
        fs,
        ref,
        inf,
        mode=mode,
        on_error=PesqError.RETURN_VALUES,
    )
    if pesq_score == PesqError.NO_UTTERANCES_DETECTED:
        logging.warning(
            f"[PESQ] Error: No utterances detected. " "Skipping this sample."
        )
    else:
        return pesq_score


class PESQMetric:
    """Perceptual Evaluation of Speech Quality (PESQ) metric module."""

    def __init__(self):
        pass

    def __call__(self, ref: np.ndarray, inf: np.ndarray, fs: int, **kwargs) -> float:
        """Calculate PESQ.

        Args:
            ref (np.ndarray): reference signal (time,)
            inf (np.ndarray): enhanced signal (time,)

        Returns:
            pesq (float): PESQ value between [-0.5, 4.5]
        """
        return pesq_metric(ref, inf, fs)