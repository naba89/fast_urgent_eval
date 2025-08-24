import argparse
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


def create_data_pairs(ref_scp, inf_scp, ref_text, utt2lang):
    refs = {}
    transcripts = {}
    language_id = {}

    with open(ref_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            refs[uid] = audio_path

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


def setup_models(device, args):
    # Intrusive metrics:
    mcd_fn = MCDMetric() if args.intrusive_metrics and args.mcd else None  # needs to run on CPU on np arrays
    pesq_fn = PESQMetric() if args.intrusive_metrics else None  # needs to run on CPU on np arrays
    lsd_fn = LSDMetric().to(device) if args.intrusive_metrics else None
    sdr_fn = SDRMetric().to(device) if args.intrusive_metrics else None
    stoi_fn = STOI().to(device) if args.intrusive_metrics else None

    # Non-intrusive metrics
    dnsmos_fn = DNSMOSOVRL().to(device) if args.non_intrusive_metrics else None
    nisqa_fn = NISQA_DIM_MOS().to(device) if args.non_intrusive_metrics else None
    scoreq_fn = Scoreq().to(device) if args.non_intrusive_metrics else None
    utmos_fn = UTMOS().to(device) if args.non_intrusive_metrics else None
    squim_fn = SQUIMMetrics().to(device) if args.non_intrusive_metrics else None

    # Task-dependent metrics
    speaker_sim_fn = SpeakerSimilarity().to(device) if args.task_dependent_metrics else None
    wer_cer_fn = OWSMEvaluator(device).to(device) if args.task_dependent_metrics else None

    # Task-independent metrics
    phoneme_similarity_fn = LevenshteinPhonemeSimilarity() if args.task_independent_metrics else None
    speech_bert_score_fn = SpeechBERTScore(device) if args.task_independent_metrics else None

    return {
    "Intrusive": {
        "LSD": lsd_fn,
        "MCD": mcd_fn,
        "PESQ": pesq_fn,
        "SDR": sdr_fn,
        "STOI": stoi_fn,
    },
    "Non-Intrusive": {
        "DNSMOS": dnsmos_fn,
        "NISQA": nisqa_fn,
        "Scoreq": scoreq_fn,
        "UTMOS": utmos_fn,
        "SQUIM": squim_fn,
    },
    "Task-Dependent": {
        "SpeakerSimilarity": speaker_sim_fn,
        "WER_CER": wer_cer_fn,
    },
    "Task-Independent": {
        "PhonemeSimilarity": phoneme_similarity_fn,
        "SpeechBERTScore": speech_bert_score_fn,
    }
    }



def main(args):
    data_pairs = create_data_pairs(args.ref_scp, args.inf_scp, args.ref_text, args.utt2lang)

    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))  # 10 hours
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device

    models = setup_models(device, args)

    with accelerator.split_between_processes(data_pairs, apply_padding=False) as split_data_pairs:
        print(f"Processing {len(split_data_pairs)} data pairs on device {device}", flush=True)

        # Here you would call your processing function, e.g.:
        # results = process_data_pairs(split_data_pairs, device=device)

        # For demonstration, we just print the first few pairs
        for uid, ref_txt, inf_audio, lang_id, ref_audio in tqdm.tqdm(split_data_pairs,
                                                                     disable=not accelerator.is_local_main_process):
            # print(f"UID: {uid}, Ref Text: {ref_txt}, Inf Audio: {inf_audio}, Lang ID: {lang_id}, Ref Audio: {ref_audio}")
            ref, ref_sr = torchaudio.load(ref_audio)
            inf, inf_sr = torchaudio.load(inf_audio)

            ref_np = ref.numpy()
            inf_np = inf.numpy()

            assert ref_sr == inf_sr, f"Sampling rate mismatch for {uid}: {ref_sr} vs {inf_sr}"
            assert ref.shape == inf.shape, f"Shape mismatch for {uid}: {ref.shape} vs {inf.shape}"

            ref = ref.to(device)
            inf = inf.to(device)

            if ref_sr != 16000:  # resample once, since 16khz is needed by many metrics
                ref_16k = torchaudio.functional.resample(ref, ref_sr, 16000)
                inf_16k = torchaudio.functional.resample(inf, inf_sr, 16000)


            # Run the metrics manually, since different metrics might need different arguments
            scores = {}

            # Intrusive metrics
            if args.intrusive_metrics:
                if models["Intrusive"]["LSD"] is not None:
                    scores["LSD"] = models["Intrusive"]["LSD"](ref, inf, ref_sr).item()
                if models["Intrusive"]["MCD"] is not None:
                    # requires numpy at original sampling rate
                    scores["MCD"] = models["Intrusive"]["MCD"](ref_np.squeeze(), inf_np.squeeze(), ref_sr)
                if models["Intrusive"]["PESQ"] is not None:
                    # needs either 8k or 16k
                    if ref_sr == 8000:
                        ref_pesq = ref_np
                        inf_pesq = inf_np
                        sr_pesq = 8000
                    else:
                        ref_pesq = ref_16k.cpu().numpy()
                        inf_pesq = inf_16k.cpu().numpy()
                        sr_pesq = 16000
                    scores["PESQ"] = models["Intrusive"]["PESQ"](ref_pesq.squeeze(), inf_pesq.squeeze(), sr_pesq)
                if models["Intrusive"]["SDR"] is not None:
                    scores["SDR"] = models["Intrusive"]["SDR"](ref, inf).item()
                if models["Intrusive"]["STOI"] is not None:
                    # needs 10k, so resample to 10khz using pystoi resample_oct
                    ref_10k = ref_np
                    inf_10k = inf_np
                    if ref_sr != 10000:  # for STOI
                        ref_10k = resample_oct(ref_np, 10000, ref_sr)
                        inf_10k = resample_oct(inf_np, 10000, inf_sr)
                    ref_10k = torch.from_numpy(ref_10k).to(device)
                    inf_10k = torch.from_numpy(inf_10k).to(device)
                    scores["STOI"] = models["Intrusive"]["STOI"](ref=ref_10k, inf=inf_10k).item()

            # Non-intrusive metrics
            if args.non_intrusive_metrics:
                if models["Non-Intrusive"]["DNSMOS"] is not None:
                    scores["DNSMOS"] = models["Non-Intrusive"]["DNSMOS"](inf=inf_16k, fs=16000).item()
                if models["Non-Intrusive"]["NISQA"] is not None:
                    scores["NISQA"] = models["Non-Intrusive"]["NISQA"](inf=inf, fs=inf_sr).item()
                if models["Non-Intrusive"]["Scoreq"] is not None:
                    scores["Scoreq"] = models["Non-Intrusive"]["Scoreq"](ref=ref_16k, inf=inf_16k, fs=16000).item()
                if models["Non-Intrusive"]["UTMOS"] is not None:
                    scores["UTMOS"] = models["Non-Intrusive"]["UTMOS"](inf=inf_16k, fs=16000).item()
                if models["Non-Intrusive"]["SQUIM"] is not None:
                    scores["SQUIM"] = models["Non-Intrusive"]["SQUIM"](inf_16k, 16000).item()

            # Task-dependent metrics
            if args.task_dependent_metrics:
                if models["Task-Dependent"]["SpeakerSimilarity"] is not None:
                    scores["SpeakerSimilarity"] = models["Task-Dependent"]["SpeakerSimilarity"](ref=ref_16k, inf=inf_16k, fs=16000).item()
                if models["Task-Dependent"]["WER_CER"] is not None:
                    scores["WER_CER"] = models["Task-Dependent"]["WER_CER"](audio=inf_16k, ref_text=ref_txt,
                                                                   sr=16000, lang_id=lang_id, uid=uid)

            # Task-independent metrics
            if args.task_independent_metrics:
                if models["Task-Independent"]["PhonemeSimilarity"] is not None:
                    scores["PhonemeSimilarity"] = models["Task-Independent"]["PhonemeSimilarity"](ref_16k.squeeze(), inf_16k.squeeze(), 16000)
                if models["Task-Independent"]["SpeechBERTScore"] is not None:
                    scores["SpeechBERTScore"] = models["Task-Independent"]["SpeechBERTScore"](ref_16k, inf_16k, 16000).item()

            print(f"UID: {uid}, Scores: {scores}", flush=True)



if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_scp", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/dataprep/urgent2025_challenge/data/nonblind/spk1.scp",
                        )
    parser.add_argument("--inf_scp", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/fast_urgent_eval/fast_urgent_eval/metrics/calculate_metrics_accelerate.py",
                        )
    parser.add_argument("--ref_text", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/dataprep/urgent2025_challenge/data/nonblind/text",
                        )
    parser.add_argument("--utt2lang", type=str,
                        default="/data/umiushi0/users/nabarun/projects/urgent2025/dataprep/urgent2025_challenge/data/nonblind/utt2lang",
                        )
    parser.add_argument("--intrusive_metrics", action="store_true", default=False)
    parser.add_argument("--non_intrusive_metrics", action="store_true", default=False)
    parser.add_argument("--task_dependent_metrics", action="store_true", default=False)
    parser.add_argument("--task_independent_metrics", action="store_true", default=False)
    parser.add_argument("--mcd", action="store_true", default=False, help="Compute MCD, which is slow and requires numpy arrays")
    args = parser.parse_args()

    main(args)
