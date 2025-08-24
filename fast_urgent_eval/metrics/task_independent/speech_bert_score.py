import torch
from discrete_speech_metrics import SpeechBERTScore as SBS
from discrete_speech_metrics.speechbertscore import bert_score

TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
class SpeechBERTScore:
    """SpeechBERTScore.

    Reference:
        SpeechBERTScore: Reference-Aware Automatic Evaluation of Speech
        Generation Leveraging NLP Evaluation Metrics
        https://arxiv.org/abs/2401.16812
    """

    def __init__(self, device="cpu"):
        self.speech_bert_score = SBS(
            sr=TARGET_FS, model_type="mhubert-147", layer=8, use_gpu="cuda" in device
        )

    def __call__(self, ref: torch.Tensor, inf: torch.Tensor, fs) -> tuple:
        if ref.ndim == 1:
            ref = ref.unsqueeze(0)
        if inf.ndim == 1:
            inf = inf.unsqueeze(0)
        gt_wav = ref.float()
        gen_wav = inf.float()

        if self.speech_bert_score.sr != 16000:
            gt_wav = self.speech_bert_score.resampler(gt_wav)
            gen_wav = self.speech_bert_score.resampler(gen_wav)

        v_ref = self.speech_bert_score.process_feats(gt_wav)
        v_gen = self.speech_bert_score.process_feats(gen_wav)
        precision, recall, f1_score = bert_score(v_gen.squeeze(0), v_ref.squeeze(0))

        return precision, recall, f1_score