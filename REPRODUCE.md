# Reproducing the paper

All commands are run from the repository root with the virtual environment activated. Times are for an NVIDIA RTX 5050 Laptop GPU (8 GB) and are approximate.

## 1. Preprocessing (once)

| Step | Command | Time |
|---|---|---|
| Audio + 16 frames per clip | `python code/prepare_data.py` | ~2 h |
| Faces, shots, active speaker, regions, quality, VAD | `python code/segment_detect.py` | ~4 h |
| Frozen WavLM-Large + CLIP ViT-L/14 features | `python code/extract_features.py` | ~5 h |

Outputs are written to `processed/<DATASET>/`. All steps can be stopped and resumed; finished clips are skipped.

## 2. Main experiments

```bash
# all baselines + TRUST-AV, every dataset and protocol used in the paper
python code/train.py --dataset meld --model baselines trustav3_no_distill --protocol official --seeds 1 2 3
python code/train.py --dataset crema ravdess --model baselines trustav3_no_distill --protocol cv5 --seeds 1 2 3 4 5 --spk_norm

# TRUST-AV-E, plain-averaging variant and ensemble baseline on all datasets, plus report
python code/run_all.py
```

`run_all.py` prints, per dataset, all models and McNemar tests of TRUST-AV-E against the strongest competitors, and writes `results/run_all_report.txt`.

## 3. Ablations

```bash
python code/train.py --dataset crema ravdess --model ablations3 --protocol cv5 --seeds 1 2 3 4 5 --spk_norm
python code/train.py --dataset meld --model ablations3 --protocol official --seeds 1 2 3
```

## 4. Supplementary studies

```bash
# without speaker normalisation (Table S11)
python code/train.py --dataset crema ravdess --model b3_early b10_saf b11_mamba trustav3_no_distill --protocol cv5 --seeds 1 2 3 4 5
# equal-budget tuning (Table S13)
python code/tune.py --datasets ravdess
# partial WavLM fine-tuning (Table S12)
python code/extract_wavlm_mid.py --datasets crema
python code/train.py --dataset crema --model trustav3_no_distill b10_saf --protocol cv5 --seeds 1 2 3 4 5 --ft_audio
```

## 5. Explainability (Figs. 11–13)

```bash
python code/explain.py
```

## 6. Tables and figures

```bash
python code/make_tables.py     # paper/tables  (CSV, LaTeX, All_Tables.xlsx)
python code/make_figures.py    # paper/figures (PNG 300 dpi, PDF)
```

These two scripts need only the files in `results/` (no GPU). With the pre-computed predictions from the release asset unzipped into `results/`, every table and figure can be rebuilt without retraining.

## Notes on determinism

Runs are seeded (`--seeds`), but GPU kernels are not fully deterministic, so retrained models may differ slightly from the reported numbers (typically by a few tenths of a point).
