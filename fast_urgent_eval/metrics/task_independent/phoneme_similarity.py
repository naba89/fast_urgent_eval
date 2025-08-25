# Based on https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional
from Levenshtein import distance
from torch.nn import Module
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


METRICS = ("PhonemeSimilarity",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
class PhonemePredictor(Module):
    # espeak installation is required for this function to work
    # To install, try
    # https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md#linux
    def __init__(
        self, checkpoint="facebook/wav2vec2-lv-60-espeak-cv-ft",
    ):
        # https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(checkpoint)
        self.do_normalize = self.processor.feature_extractor.do_normalize
        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint)

    def forward(self, waveform):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if self.do_normalize:
            waveform = F.layer_norm(waveform, waveform.shape[-1:], eps=1e-12)

        # retrieve logits
        logits = self.model(waveform).logits

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)


class LevenshteinPhonemeSimilarity:
    """Levenshtein Phoneme Similarity.

    Reference:
        J. Pirklbauer, M. Sach, K. Fluyt, W. Tirry, W. Wardah, S. Moeller,
        and T. Fingscheidt, “Evaluation metrics for generative speech enhancement
        methods: Issues and perspectives,” in Speech Communication; 15th ITG Conference,
        2023, pp. 265-269.
        https://ieeexplore.ieee.org/document/10363040
    """

    def __init__(self, device):
        self.phoneme_predictor = PhonemePredictor().to(device)

    def __call__(self, ref: torch.Tensor, inf: torch.Tensor, sr: int) -> float:
        ref = torchaudio.functional.resample(ref, orig_freq=sr, new_freq=TARGET_FS).squeeze()
        inf = torchaudio.functional.resample(inf, orig_freq=sr, new_freq=TARGET_FS).squeeze()
        sample_phonemes = self.phoneme_predictor(inf)[0].replace(" ", "")
        ref_phonemes = self.phoneme_predictor(ref)[0].replace(" ", "")
        if len(ref_phonemes) == 0:
            return np.nan
        lev_distance = distance(sample_phonemes, ref_phonemes)
        return 1 - lev_distance / len(ref_phonemes)
