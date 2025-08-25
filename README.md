# fast_urgent_eval
Fast and efficient implementation of various Speech Enhancement (SE) metrics.

Inference script is provided for multi-node/gpu evaluation for URGNENT 2025~ challenge using Huggingface Accelerate.

Peak GPU memory usage is around 16-17 GB per GPU for URGENT 2025 nonblind test set.

Evaluation on 1 node 4 A100 GPUs for 1000 utterances takes less than 10 minutes.

## Installation

```bash
conda env create -f environment.yml
conda activate fast_urgent_eval
pip install -r requirements.txt
```


## Available Metrics
- **Intrusive:** MCD, PESQ, LSD, SDR, STOI
- **Non-intrusive:** DNSMOS, NISQA, Scoreq, UTMOS, SQUIM
- **Task-dependent:** Speaker Similarity, CAcc
- **Task-independent:** Phoneme Similarity, SpeechBERTScore

Additional metrics will be added in the future as needed.

## URGNET challenge SE evaluation command line usage

```bash
# Single GPU
python calculate_metrics_accelerate.py \
    --inf_scp <path to your enhanced_scp, with fileids and abosulte audio paths> \
    --output_dir <path to save the scores> \
    --base_dir <base dir of the urgent challenge, e.g. ~/urgent2025_challenge/> \
    --ref_scp <path to your ref_scp, relative to base_dir, audio paths also relative to base_dir> \
    --ref_text <path to your ref_text, relative to base_dir> \
    --utt2lang <path to your utt2lang, relative to base_dir> \
    --non_intrusive_metrics \
    --intrusive_metrics \
    --task_dependent_metrics \
    --task_independent_metrics

# Multi Node/GPU (sample accelerate config file acc_cfg.yaml for 1 node 4 gpus)
accelerate launch --config_file acc_cfg.yaml calculate_metrics_accelerate.py \
    --inf_scp <path to your enhanced_scp, with abosulte audio paths> \
    --output_dir <path to save the scores> \
    --base_dir <base dir of the urgent challenge, e.g. ~/urgent2025_challenge/> \
    --ref_scp <path to your ref_scp, relative to base_dir, audio paths also relative to base_dir> \
    --ref_text <path to your ref_text, relative to base_dir> \
    --utt2lang <path to your utt2lang, relative to base_dir> \
    --non_intrusive_metrics \
    --intrusive_metrics \
    --task_dependent_metrics \
    --task_independent_metrics
```

Be sure specify the correct scps and paths according to your setup and test/valid sets.
File ids in the `inf_scp` and `ref_scp` should match.

## Python Usage of Individual Metrics
Check the `setup_metrics` and `compute_metrics` functions in calculate_metrics_accelerate.py for examples of how to use individual metrics.

## Citation
Implementations are based on the official URGNET 2025 evaluation recipe and various open-source repositories.
https://github.com/urgent-challenge/urgent2025_challenge/blob/main/evaluation_metrics

## License
While most of the code is released under the MIT license, please note that some components may be subject to different licenses. 
Please refer to the respective metrics and the official evaluation scripts for detailed license information.
