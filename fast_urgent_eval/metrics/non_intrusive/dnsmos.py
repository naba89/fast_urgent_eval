import os

import torch
import torch.nn as nn
from onnx2torch import convert

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

this_dir = os.path.dirname(os.path.realpath(__file__))
PRIMARY_MODEL_PATH = os.path.join(this_dir, "dnsmos_models", "sig_bak_ovr.onnx")

def poly1d(coefficients):
    coefficients = tuple(reversed(coefficients))
    def func(p):
        return sum(coef * p**i for i, coef in enumerate(coefficients))
    return func


class DNSMOSOVRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.primary_model = convert(PRIMARY_MODEL_PATH).eval()
        for param in self.primary_model.parameters():
            param.requires_grad = False

    def get_polyfit_val(self, ovr):
        p_ovr = poly1d([-0.06766283, 1.11546468, 0.04602535])
        ovr_poly = p_ovr(ovr)
        return ovr_poly

    def forward(self, inf, fs, **kwargs):
        """
        audio: torch.Tensor, shape (B, T)
        input_fs: int, input audio sampling rate
        """
        assert fs == SAMPLING_RATE, "Input sample rate must be 16000 Hz."

        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)

        repeat_factor = (len_samples + inf.shape[-1] - 1) // inf.shape[-1]  # Calculate minimum number of repeats
        audio = inf.repeat_interleave(repeat_factor, dim=-1)
        audio = audio[..., :len_samples]
        num_hops = int(audio.shape[-1] // SAMPLING_RATE - INPUT_LENGTH) + 1
        hop_len_samples = SAMPLING_RATE
        predicted_mos_ovr_seg = []
        for idx in range(num_hops):
            audio_seg = audio[
                ..., int(idx * hop_len_samples) : int((idx + INPUT_LENGTH) * hop_len_samples)
            ]
            if audio_seg.shape[-1] < len_samples:
                continue
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.primary_model(audio_seg)[0]
            mos_ovr = self.get_polyfit_val(mos_ovr_raw)
            predicted_mos_ovr_seg.append(mos_ovr)

        return torch.stack(predicted_mos_ovr_seg).mean()


if __name__ == '__main__':
    model = DNSMOSOVRL()
    dummy_inp = torch.randn(1, 16000)
    score = model(dummy_inp, 16000)
    print(score)
