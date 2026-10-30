"""
run_all.py - one command: TRUST-AV-E and its fair comparisons on all three datasets, then a report.

What it runs (finished runs are skipped, so it is safe to stop and start again):
  0. 2-minute check that the new models run (RAVDESS, 300 clips, 2 epochs)
  1. RAVDESS  speaker-independent 5-fold CV + speaker normalisation
  2. CREMA-D  speaker-independent 5-fold CV + speaker normalisation
  3. MELD     official split, 3 seeds (no speaker normalisation, as for every MELD result)
  models: trustave      TRUST-AV-E (trusted Dempster-Shafer fusion of TRUST-AV core + state-space expert)
          trustave_avg  ablation: same two experts, plain probability averaging
          b13_ens       baseline: plain ensemble of the two strongest baselines (Mamba + SAF)
  --with_baselines also runs the 9 baselines still missing on CREMA-D / RAVDESS (+4-5 h)

At the end it prints, per dataset, every available model (mean over folds/seeds) and
McNemar tests of TRUST-AV-E against Mamba, the plain ensemble and TRUST-AV, and writes
results/run_all_report.txt and results/run_all_table.csv.

Usage (from the repository root):
    .\\venv\\Scripts\\python.exe code\\run_all.py
    .\\venv\\Scripts\\python.exe code\\run_all.py --with_baselines
    .\\venv\\Scripts\\python.exe code\\run_all.py --report_only        # just rebuild the report
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                                    # noqa: E402

NEW = ["trustave", "trustave_avg", "b13_ens"]
MISSING = ["b1_audio", "b2_video", "b4_late", "b5_tfn", "b6_lmf", "b7_gmu", "b8_mult", "b9_jca", "b12_tmc"]
PLAN = [  # dataset key, protocol, speaker normalisation, seeds
    ("ravdess", "cv5", True, [1, 2, 3, 4, 5]),
    ("crema", "cv5", True, [1, 2, 3, 4, 5]),
    ("meld", "official", False, [1, 2, 3]),
]
SHOW = ["trustave", "trustave_avg", "trustav3_no_distill", "b13_ens", "b11_mamba", "b10_saf", "b3_early",
        "b4_late", "b5_tfn", "b6_lmf", "b7_gmu", "b8_mult", "b9_jca", "b12_tmc", "b1_audio", "b2_video"]
NAMES = {"trustave": "TRUST-AV-E (ours)", "trustave_avg": "TRUST-AV-E, plain averaging (ablation)",
         "trustav3_no_distill": "TRUST-AV (ours)", "b13_ens": "B13 Ensemble Mamba+SAF",
         "b11_mamba": "B11 Mamba", "b10_saf": "B10 SAF", "b3_early": "B3 Early", "b4_late": "B4 Late",
         "b5_tfn": "B5 TFN", "b6_lmf": "B6 LMF", "b7_gmu": "B7 GMU", "b8_mult": "B8 MulT", "b9_jca": "B9 JCA",
         "b12_tmc": "B12 TMC", "b1_audio": "B1 Audio", "b2_video": "B2 Video"}


def make_args(protocol, sn, seeds, smoke=False):
    return argparse.Namespace(
        protocol=protocol, seeds=seeds, epochs=60, patience=12, bs=64, lr=3e-4, wd=0.05, dim=256,
        select=None, smoke=smoke, verbose=False, ft_audio=False, lr_ft=3e-5, lam_cons=1.0,
        spk_norm=sn, hp=None, tag="", ema=0.0)


def folder(key, protocol, sn):
    return T.RESULTS / T.DATASETS[key] / (protocol + ("_sn" if sn else ""))


def train_all(models_for):
    for key, protocol, sn, seeds in PLAN:
        out = folder(key, protocol, sn)
        todo = [(m, s) for m in models_for(key) for s in seeds
                if not (out / f"{m}_seed{s}.json").exists()]
        print(f"\n######## {T.DATASETS[key]} | {protocol}{' + speaker norm' if sn else ''} | "
              f"{len(todo)} runs to do ########")
        if not todo:
            continue
        data = T.load_dataset(key)
        if sn:
            T.speaker_normalise(data[0], data[1])
        args = make_args(protocol, sn, seeds)
        for m, s in todo:
            T.run(args, key, m, s, data)


def smoke_check():
    print("\n######## quick check of the new models (2 epochs, 300 clips) ########")
    data = T.load_dataset("ravdess")
    T.speaker_normalise(data[0], data[1])
    args = make_args("cv5", True, [1], smoke=True)
    for m in NEW:
        T.run(args, "ravdess", m, 1, data)
    print("quick check passed\n")


# ----------------------------------------------------------------------------- report
def load_preds(out, model, seeds):
    """correct/incorrect per test clip, pooled over folds/seeds, in a fixed order."""
    keys, corr, spk = [], [], []
    for s in seeds:
        z = np.load(out / f"{model}_seed{s}_preds.npz", allow_pickle=True)
        cid = z["clip_id"].astype(str)
        o = np.argsort(cid)
        keys += [f"{s}|{c}" for c in cid[o]]
        corr.append((z["pred"] == z["y"])[o])
        spk.append(z["speaker"].astype(str)[o])
    return np.array(keys), np.concatenate(corr), np.concatenate(spk)


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n, k = n01 + n10, min(n01, n10)
    if n == 0:
        return n01, n10, 1.0
    if n > 1000:                                         # normal approximation with continuity correction
        z = (abs(n01 - n10) - 1) / math.sqrt(n)
        return n01, n10, math.erfc(z / math.sqrt(2))
    return n01, n10, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def spk_boot_ci(a, b, spk, reps=2000):
    rng, u = np.random.default_rng(0), np.unique(spk)
    groups = {x: np.where(spk == x)[0] for x in u}
    d = []
    for _ in range(reps):
        idx = np.concatenate([groups[x] for x in rng.choice(u, len(u))])
        d.append(a[idx].mean() - b[idx].mean())
    return np.percentile(d, [2.5, 97.5]) * 100


def report():
    lines, rows = [], []
    for key, protocol, sn, seeds in PLAN:
        out = folder(key, protocol, sn)
        tab = []
        for m in SHOW:
            recs = [json.loads((out / f"{m}_seed{s}.json").read_text()) for s in seeds
                    if (out / f"{m}_seed{s}.json").exists()]
            if len(recs) < len(seeds):
                continue
            r = lambda f: [f(x) for x in recs]
            acc = r(lambda x: x["test"]["acc"])
            rob = r(lambda x: np.mean([v["acc"] for v in x["robust"].values()]))
            nov, noa = r(lambda x: x["robust"]["video_missing"]["acc"]), r(lambda x: x["robust"]["audio_missing"]["acc"])
            row = dict(dataset=T.DATASETS[key], model=NAMES.get(m, m), acc=np.mean(acc) * 100,
                       acc_sd=np.std(acc, ddof=1) * 100, wf1=np.mean(r(lambda x: x["test"]["wf1"])) * 100,
                       mf1=np.mean(r(lambda x: x["test"]["mf1"])) * 100, uar=np.mean(r(lambda x: x["test"]["uar"])) * 100,
                       no_video=np.mean(nov) * 100, no_audio=np.mean(noa) * 100,
                       missing_avg=(np.mean(nov) + np.mean(noa)) / 2 * 100, robust10=np.mean(rob) * 100,
                       Mparams=recs[0]["params"] / 1e6, _m=m)
            tab.append(row)
        if not tab:
            continue
        df = pd.DataFrame(tab).sort_values("acc", ascending=False)
        rows.append(df)
        title = f"{T.DATASETS[key]} | {protocol}{' + speaker normalisation' if sn else ''} | {len(seeds)} folds/seeds"
        lines += ["", "=" * len(title), title, "=" * len(title),
                  df.drop(columns=["dataset", "_m"]).round(2).to_string(index=False)]
        have = set(df["_m"])
        if "trustave" in have:
            ka, A, spk = load_preds(out, "trustave", seeds)
            lines.append("\nTRUST-AV-E vs  | acc diff | 95% CI (speaker bootstrap) | only E right | only other right | McNemar p")
            for m in ["b11_mamba", "b13_ens", "trustav3_no_distill", "trustave_avg", "b10_saf"]:
                if m not in have:
                    continue
                kb, Bc, _ = load_preds(out, m, seeds)
                if not (ka == kb).all():
                    lines.append(f"  {m}: test clips differ, skipped")
                    continue
                n01, n10, p = mcnemar(A, Bc)
                lo, hi = spk_boot_ci(A, Bc, spk)
                lines.append(f"  {NAMES[m]:38s} {100 * (A.mean() - Bc.mean()):+6.2f}   [{lo:+.2f}, {hi:+.2f}]"
                             f"            {n01:5d}        {n10:5d}        {p:.3f}")
    text = "\n".join(lines)
    print(text)
    (T.RESULTS / "run_all_report.txt").write_text(text)
    if rows:
        pd.concat(rows).drop(columns=["_m"]).round(3).to_csv(T.RESULTS / "run_all_table.csv", index=False)
    print(f"\nSaved: {T.RESULTS / 'run_all_report.txt'} and run_all_table.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with_baselines", action="store_true", help="also run the 9 missing baselines")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--no_check", action="store_true", help="skip the 2-minute quick check")
    args = ap.parse_args()
    if not args.report_only:
        if not args.no_check:
            smoke_check()
        train_all(lambda key: NEW + (MISSING if args.with_baselines and key != "meld" else []))
    report()
    print("\nDone.")


if __name__ == "__main__":
    main()
