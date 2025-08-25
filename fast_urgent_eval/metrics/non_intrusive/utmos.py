import torch
import  torch.nn as nn


class UTMOS(nn.Module):
    def __init__(self):
        super().__init__()
        self.utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0",
                                     "utmos22_strong", trust_repo=True).eval()


    @torch.inference_mode()
    def forward(self, inf, sr, **kwargs):
        """wave-to-score :: (B, T) -> float """
        utter_score = self.utmos_model(inf, sr)
        return utter_score.mean().item()
