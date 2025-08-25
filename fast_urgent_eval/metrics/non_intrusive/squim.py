import torch
import torch.nn as nn
import torchaudio
from torchaudio.pipelines import SQUIM_OBJECTIVE


class SQUIMMetrics(nn.Module):
    def __init__(self):
        super().__init__()

        self.objective_model = SQUIM_OBJECTIVE.get_model().eval()
        self.sample_rate = int(SQUIM_OBJECTIVE.sample_rate)

    @torch.inference_mode()
    def forward(self, inf, sr, **kwargs):
        inf = torchaudio.functional.resample(inf, sr, self.sample_rate)
        stoi_hyp, pesq_hyp, si_sdr_hyp = self.objective_model(inf)
        return stoi_hyp.item(), pesq_hyp.item(), si_sdr_hyp.item()
