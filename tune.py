"""
tune.py - equal-budget hyper-parameter search for TRUST-AV and the three strongest baselines.

Fairness rules (report these in the paper):
  * every model gets the SAME number of configurations (N_CONFIGS, the first one = defaults)
  * configurations are sampled with a fixed seed from each model's own search space
  * the best configuration is chosen on VALIDATION scores only (mean over the tuning folds)
  * the chosen configuration is then trained and evaluated on all 5 speaker-independent folds;
    test scores are never used for selection

Results
  results/tuning/<DATASET>_trials.csv     every trial (config, fold, validation score)
  results/tuning/<DATASET>_best.json      chosen configuration per model
  results/<DATASET>/cv5_sn_tuned/...      final 5-fold results of the tuned models
  printed summary table at the end

Safe to stop and re-run: finished trials are read back from the trials file and skipped.

Usage (from the repository root):
    .\\venv\\Scripts\\python.exe code\\tune.py                       # CREMA-D then RAVDESS
    .\\venv\\Scripts\\python.exe code\\tune.py --datasets ravdess
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                                    # noqa: E402

N_CONFIGS = 12
TUNE_FOLDS = {"crema": [1, 2], "ravdess": [1, 2, 3, 4, 5]}
MODELS = ["trustav3_no_distill", "b11_mamba", "b10_saf", "b3_early"]
# training-recipe options shared by EVERY model (incl. weight EMA)
COMMON = {"dim": [192, 256, 320], "lr": [1e-4, 3e-4, 6e-4], "wd": [0.01, 0.05, 0.1], "ema": [0.0, 0.999]}
# each model's own architecture hyper-parameters
SPACES = {
    "trustav3_no_distill": {"layers": [1, 2, 3], "p_drop": [0.1, 0.2, 0.3], "lam_edl": [0.25, 0.5, 1.0],
                            "decision": ["ds", "poe"], "mamba_layers": [0, 1]},
    "b10_saf": {"layers": [1, 2, 3], "p_drop": [0.1, 0.2, 0.3]},
    "b11_mamba": {"layers": [1, 2, 3], "d_state": [8, 16, 32]},
    "b3_early": {"layers": [1, 2, 3]},
}
DEFAULTS = {
    "trustav3_no_distill": {"layers": 2, "p_drop": 0.2, "lam_edl": 0.5, "decision": "ds", "mamba_layers": 0},
    "b10_saf": {"layers": 2, "p_drop": 0.2},
    "b11_mamba": {"layers": 2, "d_state": 16},
    "b3_early": {"layers": 2},
}


def configs(model):
    """N_CONFIGS configurations: defaults first, then a seeded random sample of the grid."""
    space = dict(COMMON, **SPACES[model])
    keys = sorted(space)
    grid = [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]
    default = dict({"dim": 256, "lr": 3e-4, "wd": 0.05, "ema": 0.0}, **DEFAULTS[model])
    rest = [g for g in grid if g != default]
    random.Random(1234).shuffle(rest)
    return [default] + rest[:N_CONFIGS - 1]


def make_args(cfg, fold, tag):
    hp = {k: v for k, v in cfg.items() if k not in COMMON}
    return argparse.Namespace(
        protocol="cv5", seeds=[fold], epochs=60, patience=12, bs=64, lr=cfg["lr"], wd=cfg["wd"],
        dim=cfg["dim"], select=None, smoke=False, verbose=False, ft_audio=False, lr_ft=3e-5,
        lam_cons=1.0, spk_norm=True, hp=hp, tag=tag, ema=cfg["ema"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["crema", "ravdess"], choices=["crema", "ravdess"])
    args = ap.parse_args()
    out = T.RESULTS / "tuning"
    out.mkdir(parents=True, exist_ok=True)

    for key in args.datasets:
        name = T.DATASETS[key]
        data = T.load_dataset(key)
        T.speaker_normalise(data[0], data[1])
        log_f = out / f"{name}_trials_v2.csv"
        done = pd.read_csv(log_f) if log_f.exists() else pd.DataFrame()
        best = {}
        print(f"\n######## {name}: tuning {len(MODELS)} models x {N_CONFIGS} configs x "
              f"{len(TUNE_FOLDS[key])} folds (validation only) ########")
        for model in MODELS:
            scores = []
            for ci, cfg in enumerate(configs(model)):
                vals = []
                for fold in TUNE_FOLDS[key]:
                    hit = done[(done.get("model") == model) & (done.get("config") == ci) &
                               (done.get("fold") == fold)] if len(done) else []
                    if len(hit):
                        vals.append(float(hit.iloc[0]["val"]))
                        continue
                    rec = T.run(make_args(cfg, fold, "tune"), key, model, fold, data)
                    vals.append(rec["best_val"])
                    row = dict(model=model, config=ci, fold=fold, val=rec["best_val"],
                               cfg=json.dumps(cfg))
                    pd.DataFrame([row]).to_csv(log_f, mode="a", header=not log_f.exists(), index=False)
                scores.append((float(np.mean(vals)), ci, cfg))
                print(f"  [{model}] config {ci} {cfg} -> mean val {np.mean(vals):.4f}")
            sc, ci, cfg = max(scores, key=lambda x: x[0])
            best[model] = dict(config=ci, cfg=cfg, mean_val=sc)
            print(f"  >>> {model}: best config {ci} (val {sc:.4f}) {cfg}")
        (out / f"{name}_best.json").write_text(json.dumps(best, indent=1))

        print(f"\n######## {name}: final 5-fold evaluation of the tuned configurations ########")
        final = []
        for model in MODELS:
            cfg = best[model]["cfg"]
            for fold in [1, 2, 3, 4, 5]:
                rec = T.run(make_args(cfg, fold, "tuned"), key, model, fold, data)
                r = rec["robust"]
                final.append(dict(model=model, fold=fold, acc=rec["test"]["acc"], wf1=rec["test"]["wf1"],
                                  mf1=rec["test"]["mf1"], no_video=r["video_missing"]["acc"],
                                  no_audio=r["audio_missing"]["acc"], params=rec["params"]))
        df = pd.DataFrame(final)
        df.to_csv(out / f"{name}_tuned_test.csv", index=False)
        g = df.groupby("model").agg(acc=("acc", "mean"), acc_sd=("acc", "std"), wf1=("wf1", "mean"),
                                    mf1=("mf1", "mean"), no_video=("no_video", "mean"),
                                    no_audio=("no_audio", "mean"), params=("params", "first"))
        g["missing_avg"] = (g["no_video"] + g["no_audio"]) / 2
        print(f"\n=== {name}: tuned models, 5-fold test (speaker normalisation) ===")
        print(g.sort_values("acc", ascending=False).round(4).to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
