from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio.functional
from Levenshtein import opcodes as levenshtein_opcodes
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.text.cleaner import TextCleaner
from espnet2.bin.s2t_inference_language import Speech2Language as Speech2Lang


# Copied from Whisper utils
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


@dataclass
class EvalResult:
    uid: Optional[str]
    hyp_text: str
    ref_text: str
    WER_details: Dict[str, int | str]
    CER_details: Dict[str, int | str]
    WER: float
    CER: float
    CAcc: float
    WAcc: float


class OWSMEvaluator(torch.nn.Module):
    """ASR evaluator as a ``torch.nn.Module`` that accepts **torch.Tensors** directly.

    - Put the module on GPU with ``evaluator.to('cuda')``.
    - Feed 1-D (T,) float32 tensors (on CPU or GPU). If on GPU, resampling runs on GPU,
      then audio is moved to CPU *only* at the final step to call ``owsm_predict``
      (ESPnet's helper expects a NumPy array). The ASR model itself runs on the module's device.
    - No multiprocessing or dataset I/O—pure core logic.
    """

    def __init__(
        self,
        device: str = "cpu",
        model_tag: str = "espnet/owsm_v3.1_ebf",
        target_fs: int = 16_000,
        beam_size: int = 5,
        cleaner: str = "whisper_basic",
        chunk_size: int = 30,
    ) -> None:
        super().__init__()
        self.target_fs = int(target_fs)
        self.beam_size = int(beam_size)
        self.chunk_size = int(chunk_size)

        # Track module device via a buffer so .to() works naturally
        self.register_buffer("_devtrack", torch.zeros(1), persistent=False)
        self._devtrack = self._devtrack.to(device)

        # Load model + text cleaner once
        self.model = Speech2Text.from_pretrained(
            model_tag=model_tag,
            device=device,
            task_sym="<asr>",
            beam_size=self.beam_size,
            predict_time=False,
        )
        self.cleaner = TextCleaner(cleaner)

        self.speech2lang = Speech2Lang.from_pretrained(
            model_tag="espnet/owsm_v3.1_ebf",
            device=device,
            nbest=1,
        )

    # --------------------------
    # Public API
    # --------------------------
    @property
    def device(self) -> torch.device:
        return self._devtrack.device

    @torch.inference_mode()
    def transcribe(self, audio: torch.Tensor, sr: int, lang_id: str=None) -> str:
        """ASR on a single utterance.

        Args:
            audio: 1-D mono waveform tensor (T,) float32/float16/bfloat16. CPU or GPU.
            sr: sampling rate of ``audio``.
            lang_id: language symbol for OWSM (e.g., "<eng>").
        Returns:
            Normalized hypothesis text (cleaned).
        """
        wav = self._to_mono_1d(audio)
        if sr != self.target_fs:
            wav = self._resample_linear(wav, sr, self.target_fs)
            wav = torchaudio.functional.resample(wav, sr, self.target_fs)
            sr = self.target_fs

        long_form = wav.numel() > self.chunk_size * sr

        # owsm_predict expects a NumPy array on CPU; convert at the boundary only
        hyp = self.predict(
            wav,
            sr,
            lang_id=lang_id,
            long_form=long_form,
        )
        return self.clean(hyp)

    def fix_length(self, audio: torch.Tensor, size: int) -> torch.Tensor:
        """Ensure audio is fixed length for standard processing."""
        if audio.shape[-1] < size:
            audio = F.pad(audio, (0, size - audio.shape[-1]), "constant", 0)
        elif audio.shape[-1] > size:
            audio = audio[:size]
        return audio

    def predict(self, audio: torch.Tensor, sr: int, lang_id: str = "eng",
                long_form: bool = False, text_prev: str = ""):
        task_sym = "<asr>"
        self.model.beam_search.beam_size = int(self.beam_size)
        assert sr == self.target_fs, (sr, self.target_fs)

        if lang_id is None or lang_id == "none":
            # Detect language using the first 30s of speech
            speech = self.fix_length(audio, size=(self.target_fs * self.chunk_size))
            lang_id = self.speech2lang(speech)[0][0].strip()[1:-1]

        lang_sym = f"<{lang_id}>"

        if long_form:  # speech will be padded in decode_long()
            try:
                self.model.maxlenratio = -300
                utts = self.model.decode_long(
                    audio,
                    condition_on_prev_text=False,
                    init_text=text_prev,
                    end_time_threshold=f"<{self.long_form_seconds - 1}.00>",
                    lang_sym=lang_sym,
                    task_sym=task_sym,
                )
                text = []
                for t1, t2, res in utts:
                    text.append(
                        f"[{format_timestamp(seconds=t1)} --> "
                        f"{format_timestamp(seconds=t2)}] {res}"
                    )
                text = "\n".join(text)

                return text
            except:
                print(
                    "An exception occurred in long-form decoding. "
                    "Fall back to standard decoding (only first 30s)"
                )
        # assuming 10 tokens per second
        self.model.maxlenratio = -min(300, int((len(audio) / self.target_fs) * 10))

        speech = self.fix_length(audio, size=(self.target_fs * self.chunk_size))
        text = self.model(speech, text_prev, lang_sym=lang_sym, task_sym=task_sym)[0][-2]

        return text

    @torch.inference_mode()
    def forward(
        self,
        audio: torch.Tensor,
        sr: int,
        ref_text: str,
        lang_id: str,
        uid: Optional[str] = None,
    ) -> EvalResult:
        """Compute WER and CER for one utterance and return counts.

        If ``ref_text`` is "<not-available>", returns empty dicts for WER/CER.
        """
        if ref_text == "<not-available>":
            return EvalResult(uid, hyp_text="", ref_text=ref_text, WER={}, CER={})

        ref_text_clean = self.clean(ref_text)
        hyp_text = self.transcribe(audio, sr, lang_id)

        wer = self._opcount_words(ref_text_clean, hyp_text)
        cer = self._opcount_chars(ref_text_clean, hyp_text)

        overall_wer = wer_from_counts(wer) * 100.0
        overall_cer = wer_from_counts(cer) * 100.0

        CAcc = 100.0 - overall_cer
        WAcc = 100.0 - overall_wer

        return EvalResult(uid, hyp_text=hyp_text, ref_text=ref_text_clean,
                          WER_details=wer, CER_details=cer,
                          WER=overall_wer, CER=overall_cer,
                          CAcc=CAcc, WAcc=WAcc)

    # --------------------------
    # Utilities
    # --------------------------
    def clean(self, text: str) -> str:
        return self.cleaner(text)

    @staticmethod
    def _to_mono_1d(x: torch.Tensor) -> torch.Tensor:
        """Ensure (T,) mono float tensor; average channels if needed."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("audio must be a torch.Tensor")
        if x.ndim == 0:
            raise ValueError("audio must have at least 1 dimension")
        x = x.float()
        if x.ndim == 1:
            return x
        # Allow shapes (C, T) or (T, C)
        if x.ndim == 2:
            if x.shape[0] == 1:
                return x[0]
            if x.shape[1] == 1:
                return x[:, 0]
            # Average over channel dim: assume (C, T) if C <= 8 else (T, C)
            if x.shape[0] <= x.shape[1]:  # (C, T)
                return x.mean(dim=0)
            else:  # (T, C)
                return x.mean(dim=1)
        # For higher dims, flatten last dim as time and mean others as channels
        return x.reshape(-1, x.shape[-1]).mean(dim=0)

    @staticmethod
    def _opcount_words(ref: str, hyp: str) -> Dict[str, int | str]:
        ref_words = ref.split()
        hyp_words = hyp.split()
        ret: Dict[str, int | str] = {"hyp_text": hyp, "ref_text": ref, "delete": 0, "insert": 0, "replace": 0, "equal": 0}
        for op, ref_st, ref_et, hyp_st, hyp_et in levenshtein_opcodes(ref_words, hyp_words):
            if op == "insert":
                ret[op] += hyp_et - hyp_st
            else:
                ret[op] += ref_et - ref_st
        OWSMEvaluator._validate_balances(ret, len(ref_words), len(hyp_words))
        return ret

    @staticmethod
    def _opcount_chars(ref: str, hyp: str) -> Dict[str, int | str]:
        ref_chars = list(ref)
        hyp_chars = list(hyp)
        ret: Dict[str, int | str] = {"hyp_text": hyp, "ref_text": ref, "delete": 0, "insert": 0, "replace": 0, "equal": 0}
        for op, ref_st, ref_et, hyp_st, hyp_et in levenshtein_opcodes(ref_chars, hyp_chars):
            if op == "insert":
                ret[op] += hyp_et - hyp_st
            else:
                ret[op] += ref_et - ref_st
        OWSMEvaluator._validate_balances(ret, len(ref_chars), len(hyp_chars))
        return ret

    @staticmethod
    def _validate_balances(ret: Dict[str, int | str], n_ref: int, n_hyp: int) -> None:
        total_ref_side = int(ret["delete"]) + int(ret["replace"]) + int(ret["equal"])
        total_hyp_side = int(ret["insert"]) + int(ret["replace"]) + int(ret["equal"])
        assert total_ref_side == n_ref, (total_ref_side, n_ref)
        assert total_hyp_side == n_hyp, (total_hyp_side, n_hyp)


# --------------------------
# Helpers
# --------------------------

def wer_from_counts(d: Dict[str, int]) -> float:
    num = d["replace"] + d["delete"] + d["insert"]
    den = d["replace"] + d["delete"] + d["equal"]
    return float(num) / max(1.0, float(den))


def evaluate_batch(
    evaluator: OWSMEvaluator,
    items: Iterable[Tuple[torch.Tensor, int, str, str, Optional[str]]],
) -> List[EvalResult]:
    results: List[EvalResult] = []
    for audio, sr, ref_text, lang_id, uid in items:
        results.append(evaluator.evaluate(audio, sr, ref_text, lang_id, uid))
    return results
