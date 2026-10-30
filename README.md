# TRUST-AV / TRUST-AV-E


Official implementation of the paper:

## Overview

Audio-visual emotion recognition models usually assume that speech and faces are always clean and available. In real interactions faces are occluded, turned away or off-screen, and speech is noisy or missing. Existing trusted-fusion methods use modality uncertainty only to weight the *final decisions*, so unreliable inputs still corrupt cross-modal feature interaction.

**TRUST-AV** lets reliability steer the fusion itself:

- **Evidential modality uncertainty**: Dirichlet heads estimate how much evidence supports the audio and the video stream of every clip.
- **Uncertainty- and quality-steered joint attention**: modality uncertainty and per-frame face quality bias the audio-visual attention *token by token*, so unreliable inputs receive less attention.
- **Trusted Dempster–Shafer fusion**: fused, audio and video opinions are combined into a prediction *and* an uncertainty.
- **TRUST-AV-E** adds a bidirectional state-space (Mamba-style) expert, merged by the same trusted rule.
- A complete pipeline: active-speaker face tracking (YuNet), face / eyes / mouth region tokens, frozen WavLM-Large and CLIP ViT-L/14 features, speaker-adversarial training. Runs on a single 8 GB laptop GPU.

## Main results

Audio and video only (no text). MELD: official split, 3 seeds. CREMA-D and RAVDESS: speaker-independent 5-fold cross-validation. All 13 baselines are re-implemented on the same features and trained under the same protocol.

| Model | MELD Acc | MELD wF1 | CREMA-D Acc | RAVDESS Acc | CREMA-D one modality missing | RAVDESS one modality missing |
|---|---|---|---|---|---|---|
| B4 Late fusion | 54.60 | 50.71 | 89.73 | 94.42 | 66.10 | 75.81 |
| B10 SAF (self-attention + modality dropout) | 50.84 | 50.13 | 89.93 | 92.78 | 75.78 | 80.46 |
| B11 Mamba fusion (state-space) | 51.94 | 49.72 | 90.22 | 94.40 | 51.28 | 62.80 |
| B13 Ensemble (Mamba + SAF) | 52.98 | 50.59 | 90.74 | 94.05 | 62.08 | 73.10 |
| **TRUST-AV** (ours, 3.4 M params) | **54.90** | **51.77** | 90.16 | 93.68 | 79.49 | 82.74 |
| **TRUST-AV-E** (ours, 5.8 M params) | 54.69 | 50.72 | **90.83** | **94.57** | **80.08** | **83.98** |

Full results for all 15 models, ten test-time degradations, significance tests (McNemar + Holm, Friedman / Nemenyi), ablations, calibration and efficiency are in [`paper/tables/All_Tables.xlsx`](paper/tables/All_Tables.xlsx).

![Accuracy vs robustness](paper/figures/F08_accuracy_vs_robustness.png)

## Repository structure

```
code/
  prepare_data.py       Step 1  16 kHz mono audio + 16 uniformly sampled frames per clip
  segment_detect.py     Step 2  YuNet face detection, shots, tracking, active speaker,
                                aligned face / eyes / mouth crops, frame quality, VAD
  extract_features.py   Step 3  frozen WavLM-Large (13 layers) + CLIP ViT-L/14 features
  extract_wavlm_mid.py  (opt.)  frame-level WavLM states for the fine-tuning study
  train.py                      all 15 models, ablations, robustness tests, results JSON
  run_all.py                    TRUST-AV-E + fair comparisons on all datasets + report
  tune.py                       equal-budget hyper-parameter tuning (supplementary)
  explain.py                    Grad-CAM, uncertainty and attention analyses (needs GPU)
  make_tables.py                rebuilds every table in paper/tables
  make_figures.py               rebuilds every figure in paper/figures
docs/DATASETS.md                where to download the datasets and how to lay them out
docs/REPRODUCE.md               step-by-step reproduction with run times
```

## Installation

Tested with Python 3.12, PyTorch 2.11 (CUDA 12.8) on Windows 11 with an NVIDIA RTX 5050 Laptop GPU (8 GB).

```bash
git clone https://github.com/ranjithkumaranandan/TRUST-AV.git
cd TRUST-AV
python -m venv venv
# Windows: venv\Scripts\activate      Linux: source venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

[FFmpeg](https://ffmpeg.org/) must be available on the `PATH`.

## Data

The datasets are **not** redistributed here. Download them from their official sources and place them under `datasets/`, as described in [`docs/DATASETS.md`](docs/DATASETS.md).

## Quick start

```bash
python code/prepare_data.py            # audio + frames
python code/segment_detect.py          # faces, tracks, regions, quality, VAD
python code/extract_features.py        # WavLM + CLIP features (GPU)
python code/run_all.py --with_baselines   # TRUST-AV-E, baselines, report
python code/make_tables.py             # all tables
python code/make_figures.py            # all figures
```

Single runs, for example TRUST-AV on CREMA-D with speaker-independent 5-fold CV:

```bash
python code/train.py --dataset crema --model trustav3_no_distill --protocol cv5 --seeds 1 2 3 4 5 --spk_norm
```

Model names: `trustave` (TRUST-AV-E), `trustav3_no_distill` (TRUST-AV), `b1_audio` … `b13_ens` (baselines), `ablations3` (component ablations). See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for all experiments and expected run times.

## Pre-computed predictions

Per-clip test predictions, probabilities, uncertainties and embeddings of every run (`*_preds.npz`, about 190 MB) are provided as a release asset, so that all tables, figures and significance tests can be recomputed without retraining. Unzip the archive into `results/`.

## Citation

If you use this code, please cite:

```bibtex
@article{anandan2026trustav,
  title   = {Trustworthy Audio-Visual Emotion Recognition via Uncertainty-Steered Attention and Evidential Expert Fusion under Missing and Corrupted Modalities},
  author  = {Anandan, Ranjith Kumar},
  journal = {Under review},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The datasets are **not** covered by this license and remain subject to their own terms: MELD, CREMA-D and RAVDESS must be obtained from their providers and used according to their licences.

## Acknowledgements

This work uses the MELD, CREMA-D and RAVDESS datasets, the WavLM and CLIP pre-trained models (via Hugging Face Transformers) and the YuNet face detector (OpenCV Zoo).
