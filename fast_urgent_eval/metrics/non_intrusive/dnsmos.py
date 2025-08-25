# Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics

import os

import torch
import torch.nn as nn
from espnet2.enh.layers.dnsmos import DNSMOS_local

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

this_dir = os.path.dirname(os.path.realpath(__file__))
PRIMARY_MODEL_PATH = os.path.join(this_dir, "dnsmos_models", "sig_bak_ovr.onnx")
P808_MODEL_PATH = os.path.join(this_dir, "dnsmos_models", "model_v8.onnx")


class DNSMOS(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.model = DNSMOS_local(
            PRIMARY_MODEL_PATH,
            P808_MODEL_PATH,
            use_gpu="cuda" in device,
            convert_to_torch=True,
        )

    @torch.inference_mode()
    def forward(self, inf, fs):
        assert fs == SAMPLING_RATE, f"Sampling rate must be {SAMPLING_RATE}, but got {fs}"
        dnsmos_score = self.model(inf, fs)['OVRL']
        return float(dnsmos_score)
