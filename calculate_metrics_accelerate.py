import logging

from accelerate.utils import gather_object

# Disable all logging messages at or below INFO level
logging.disable(logging.INFO)

import argparse
import os.path
import time
from datetime import timedelta

import torch
import torchaudio
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from pystoi.utils import resample_oct

from fast_urgent_eval.metrics.intrusive.lsd import LSDMetric
from fast_urgent_eval.metrics.intrusive.mcd import MCDMetric
from fast_urgent_eval.metrics.intrusive.pesq import PESQMetric
from fast_urgent_eval.metrics.intrusive.sdr import SDRMetric
from fast_urgent_eval.metrics.intrusive.stoi import STOI
from fast_urgent_eval.metrics.non_intrusive.dnsmos import DNSMOSOVRL
from fast_urgent_eval.metrics.non_intrusive.nisqa import NISQA_DIM_MOS
from fast_urgent_eval.metrics.non_intrusive.scoreq import Scoreq
from fast_urgent_eval.metrics.non_intrusive.squim import SQUIMMetrics
from fast_urgent_eval.metrics.non_intrusive.utmos import UTMOS
from fast_urgent_eval.metrics.task_dependent.speaker_similarity import SpeakerSimilarity
from fast_urgent_eval.metrics.task_dependent.wer_cer import OWSMEvaluator
from fast_urgent_eval.metrics.task_independent.phoneme_similarity import LevenshteinPhonemeSimilarity
from fast_urgent_eval.metrics.task_independent.speech_bert_score import SpeechBERTScore


def create_data_pairs(base_dir, ref_scp, inf_scp, ref_text, utt2lang):
    refs = {}
    transcripts = {}
    language_id = {}

    with open(ref_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            refs[uid] = os.path.join(base_dir, audio_path)

    with open(ref_text, "r") as f:
        for line in f:
            uid, txt = line.strip().split(maxsplit=1)
            transcripts[uid] = txt

    with open(utt2lang, "r") as f:
        for line in f:
            uid, lang_id = line.strip().split(maxsplit=1)
            assert uid in transcripts, uid
            language_id[uid] = lang_id

    data_pairs = []
    with open(inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, transcripts[uid], audio_path, language_id[uid], refs[uid]))

    return data_pairs


def setup_metrics(device, args):
    metrics = {}
    # Intrusive metrics:
    if args.intrusive_metrics:
        metrics["MCD"] = MCDMetric().to(device)
        metrics["PESQ"] = PESQMetric()  # needs to run on CPU on np arrays
        metrics["LSD"] = LSDMetric().to(device)
        metrics["SDR"] = SDRMetric().to(device)
        metrics["STOI"] = STOI().to(device)

    # Non-intrusive metrics
    if args.non_intrusive_metrics:
        metrics["DNSMOS"] = DNSMOSOVRL().to(device)
        metrics["NISQA"] = NISQA_DIM_MOS().to(device)
        metrics["Scoreq"] = Scoreq().to(device)
        metrics["UTMOS"] = UTMOS().to(device)
        metrics["SQUIM"] = SQUIMMetrics().to(device)

    # Task-dependent metrics
    if args.task_dependent_metrics:
        metrics["SpeakerSimilarity"] = SpeakerSimilarity().to(device)
        metrics["WER_CER"] = OWSMEvaluator(device=device.type).to(device)

    # Task-independent metrics
    if args.task_independent_metrics:
        metrics["PhonemeSimilarity"] = LevenshteinPhonemeSimilarity(device)
        metrics["SpeechBERTScore"] = SpeechBERTScore(device.type)

    return metrics


def compute_metrics(args, metrics, ref, inf, ref_sr, inf_sr, ref_txt, lang_id, uid, device):

    assert ref_sr == inf_sr, f"Sampling rate mismatch for {uid}: {ref_sr} vs {inf_sr}"
    assert ref.shape == inf.shape, f"Shape mismatch for {uid}: {ref.shape} vs {inf.shape}"

    ref_np = ref.numpy()
    inf_np = inf.numpy()

    ref = ref.to(device)
    inf = inf.to(device)

    # resample once, since 16khz is needed by many metrics
    ref_16k = torchaudio.functional.resample(ref, ref_sr, 16000)
    inf_16k = torchaudio.functional.resample(inf, inf_sr, 16000)

    scores = {}
    # Intrusive metrics
    if args.intrusive_metrics:
        scores["Intrusive"] = {}

        if ref_sr == 8000:
            ref_pesq = ref_np
            inf_pesq = inf_np
            sr_pesq = 8000
        else:
            ref_pesq = ref_16k.cpu().numpy()
            inf_pesq = inf_16k.cpu().numpy()
            sr_pesq = 16000

        scores["Intrusive"]["PESQ"] = metrics["PESQ"](ref_pesq.squeeze(), inf_pesq.squeeze(), sr_pesq)
        scores["Intrusive"]["LSD"] = metrics["LSD"](ref, inf, ref_sr)
        scores["Intrusive"]["MCD"] = metrics["MCD"](ref.squeeze(), inf.squeeze(), ref_sr)
        scores["Intrusive"]["SDR"] = metrics["SDR"](ref, inf)
        scores["Intrusive"]["STOI"] = metrics["STOI"](ref=ref.squeeze(),
                                                          inf=inf.squeeze(),
                                                          fs=ref_sr, extended=True)
    # Non-intrusive metrics
    if args.non_intrusive_metrics:
        scores["NonIntrusive"] = {}
        scores["NonIntrusive"]["DNSMOS"] = metrics["DNSMOS"](inf=inf_16k, fs=16000)
        scores["NonIntrusive"]["NISQA"] = metrics["NISQA"](inf=inf, fs=inf_sr)
        scores["NonIntrusive"]["Scoreq"] = metrics["Scoreq"](inf=inf_16k, fs=16000)
        scores["NonIntrusive"]["UTMOS"] = metrics["UTMOS"](inf=inf_16k, sr=16000)
        # stoi, pesq, sdr
        (scores["NonIntrusive"]["SQUIM_STOI"],
         scores["NonIntrusive"]["SQUIM_PESQ"],
         scores["NonIntrusive"]["SQUIM_SDR"]) = metrics["SQUIM"](inf_16k, 16000)

    # Task-dependent metrics
    if args.task_dependent_metrics:
        scores["TaskDependent"] = {}
        scores["TaskDependent"]["SpeakerSimilarity"] = metrics["SpeakerSimilarity"](ref=ref_16k, inf=inf_16k, fs=16000)
        scores["TaskDependent"]["WER_CER"] = metrics["WER_CER"](audio=inf_16k, ref_text=ref_txt,
                                                                     sr=16000, lang_id=lang_id, uid=uid)
    # Task-independent metrics
    if args.task_independent_metrics:
        scores["TaskIndependent"] = {}
        scores["TaskIndependent"]["PhonemeSimilarity"] = metrics["PhonemeSimilarity"](ref_16k.squeeze(), inf_16k.squeeze(), 16000)
        scores["TaskIndependent"]["SpeechBERTScore"] = metrics["SpeechBERTScore"](ref_16k, inf_16k, 16000)

    return scores


@torch.inference_mode()
def main(args):
    data_pairs = create_data_pairs(args.base_dir, args.ref_scp, args.inf_scp, args.ref_text, args.utt2lang)

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))  # 10 hours
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    torch.set_default_device(device)
    print("Using device:", device, flush=True)

    metrics = setup_metrics(device, args)

    with accelerator.split_between_processes(data_pairs, apply_padding=False) as split_data_pairs:
        print(f"Processing {len(split_data_pairs)} data pairs on device {device}", flush=True)

        for uid, ref_txt, inf_audio, lang_id, ref_audio in tqdm.tqdm(split_data_pairs,
                                                                     disable=not accelerator.is_local_main_process):
            ref, ref_sr = torchaudio.load(ref_audio)
            inf, inf_sr = torchaudio.load(inf_audio)

            scores = compute_metrics(args, metrics, ref, inf, ref_sr, inf_sr, ref_txt, lang_id, uid, device)

    accelerator.wait_for_everyone()
    scores = gather_object(scores)
    if accelerator.is_main_process:
        import json
        out_file = os.path.join(args.base_dir, "metrics.json")
        with open(out_file, "w") as f:
            json.dump(scores, f, indent=4)
        print(f"Metrics saved to {out_file}", flush=True)


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/dataprep/urgent2025_challenge/")
    parser.add_argument("--ref_scp", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/dataprep/urgent2025_challenge/data/nonblind/spk1.scp",
                        )
    parser.add_argument("--inf_scp", type=str,
                        default="/home/mil/nabarun/github/urgent2026/exp/results/nonblind/scnet_transformer_ssl_v3/enh.scp",
                        )
    parser.add_argument("--ref_text", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/dataprep/urgent2025_challenge/data/nonblind/text",
                        )
    parser.add_argument("--utt2lang", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/dataprep/urgent2025_challenge/data/nonblind/utt2lang",
                        )
    parser.add_argument("--intrusive", action="store_true", default=False)
    parser.add_argument("--non_intrusive", action="store_true", default=False)
    parser.add_argument("--task_dependent", action="store_true", default=False)
    parser.add_argument("--task_independent", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
