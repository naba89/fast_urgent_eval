# Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics

import torch
import torch.nn as nn
import fast_bss_eval


def sdr_metric(ref, inf):
    """Calculate signal-to-distortion ratio (SDR).

    Args:
        ref (torch.Tensor): reference signal (num_src, time)
        inf (torch.Tensor): enhanced signal (num_src, time)
    Returns:
        sdr (float): SDR values (unbounded)
    """
    assert ref.shape == inf.shape
    if ref.ndim == 1:
        ref = ref[None, :]
        inf = inf[None, :]
    else:
        assert ref.ndim == 2, ref.shape
    num_src, _ = ref.shape
    sdr, sir, sar = fast_bss_eval.bss_eval_sources(
        ref, inf, compute_permutation=False, clamp_db=50.0
    )
    return sdr.mean().item()


class SDRMetric(nn.Module):
    """Signal-to-Distortion Ratio (SDR) metric module."""

    def __init__(self):
        super().__init__()
        pass

    def forward(self, ref: torch.Tensor, inf: torch.Tensor, **kwargs) -> float:
        """Calculate SDR.

        Args:
            ref (torch.Tensor): reference signal (num_src, time)
            inf (torch.Tensor): enhanced signal (num_src, time)

        Returns:
            sdr (float): SDR value (unbounded)
        """
        return sdr_metric(ref, inf)