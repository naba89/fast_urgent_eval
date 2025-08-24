import torch
import  torch.nn as nn
import torchaudio
from torch.nn.utils import remove_weight_norm


class UTMOS(nn.Module):
    def __init__(self, framewise=False, feature_only=False):
        super().__init__()
        self.framewise = framewise
        self.feature_only = feature_only
        self.utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0",
                                     "utmos22_strong", trust_repo=True).eval()
        self.utmos_model.blstm.train()
        for p in self.utmos_model.parameters():
            p.requires_grad = False

        # remove any weight norms
        for m in self.utmos_model.modules():
            try:
                remove_weight_norm(m)
            except ValueError:
                # no weight norm to remove
                pass

    @torch.inference_mode()
    def forward(self, inf, sr, **kwargs):
        """wave-to-score :: (B, T) -> (B, F) """

        # Resampling :: (B, T) -> (B, T)
        wave = torchaudio.functional.resample(inf, orig_freq=sr, new_freq=16000)

        # Feature extraction :: (B, T) -> (B, Frame, Feat)
        unit_series = self.utmos_model.wav2vec2(wave)

        if self.feature_only:
            return unit_series

        bsz, frm, _ = unit_series.size()

        # DataDomain/JudgeId Embedding's Batch/Time expansion :: (B=1, Feat) -> (B=bsz, Frame=frm, Feat)
        domain_series = self.utmos_model.domain_emb.unsqueeze(1).expand(bsz, frm, -1)
        judge_series  =  self.utmos_model.judge_emb.unsqueeze(1).expand(bsz, frm, -1)

        # Feature concatenation :: (B, Frame, Feat=f1) + (B, Frame, Feat=f2) + (B, Frame, Feat=f3) -> (B, Frame, Feat=f1+f2+f3)
        cat_series = torch.cat([unit_series, domain_series, judge_series], dim=2)

        # Frame-scale score estimation :: (B, Frame, Feat) -> (B, Frame, Feat) -> (B, Frame, Feat=1) - BLSTM/Projection
        feat_series = self.utmos_model.blstm(cat_series)[0]
        score_series = self.utmos_model.projection(feat_series)

        if self.framewise:
            return score_series.squeeze(2)  # (B, Frame)

        # Utterance-scale score :: (B, Frame, Feat=1) -> (B, Feat=1) -> (B,) - Time averaging
        utter_score = score_series.mean(dim=1).squeeze(1) * 2 + 3

        return utter_score.item()
