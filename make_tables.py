"""
make_tables.py - every table for the paper, from the saved results (no GPU / torch needed).

Writes to <repo>\\paper\\tables\\
    Txx_<name>.csv          numeric table (for checking / plotting)
    Txx_<name>.tex          LaTeX (booktabs), best value per column in bold
    All_Tables.xlsx         every table as one sheet
    README.txt              list of tables and what each contains

Re-run any time (e.g. after the ablation runs finish): tables are rebuilt from results\\.

Usage (from the repository root):
    .\\venv\\Scripts\\python.exe code\\make_tables.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES, PROC = ROOT / "results", ROOT / "processed"
OUT = ROOT / "paper" / "tables"

DS = ["MELD", "CREMA-D", "RAVDESS"]
MAIN = {"MELD": ("official", [1, 2, 3]), "CREMA-D": ("cv5_sn", [1, 2, 3, 4, 5]),
        "RAVDESS": ("cv5_sn", [1, 2, 3, 4, 5])}
BASE = ["b1_audio", "b2_video", "b3_early", "b4_late", "b5_tfn", "b6_lmf", "b7_gmu", "b8_mult",
        "b9_jca", "b10_saf", "b11_mamba", "b12_tmc", "b13_ens"]
OURS = ["trustav3_no_distill", "trustave"]
NAME = {"b1_audio": "B1 Audio-only (WavLM)", "b2_video": "B2 Video-only (CLIP)", "b3_early": "B3 Early fusion",
        "b4_late": "B4 Late fusion", "b5_tfn": "B5 TFN", "b6_lmf": "B6 LMF", "b7_gmu": "B7 GMU",
        "b8_mult": "B8 MulT", "b9_jca": "B9 JCA", "b10_saf": "B10 SAF", "b11_mamba": "B11 Mamba fusion",
        "b12_tmc": "B12 TMC", "b13_ens": "B13 Ensemble (Mamba+SAF)",
        "trustav3_no_distill": "TRUST-AV (ours)", "trustave": "TRUST-AV-E (ours)",
        "trustave_avg": "TRUST-AV-E w/ plain averaging",
        "trustav": "TRUST-AV v1 (relative-uncertainty routing)", "trustav2": "v2 (uncertainty-biased joint attention)",
        "trustav3": "v3 (v2 + trusted decision fusion + self-distillation)",
        "trustav3_no_trio": "v3 w/o trusted decision fusion (no distill)", "trustav2_no_unc": "v2 w/o uncertainty",
        "trustav4": "v4 (+ trusted experts + degradation consistency)", "trustav4s": "v4-small (d=160)",
        "trustav4_no_exp": "v4 w/o experts", "trustav4_no_cons": "v4 w/o consistency",
        "trustav5": "v5 (state-space stage + trusted attention)"}
ABL3 = {"trustav3_no_distill": "Full TRUST-AV", "trustav3_no_distill_no_unc": "w/o evidential uncertainty",
        "trustav3_no_distill_no_regions": "w/o facial-region tokens (face only)",
        "trustav3_no_distill_no_adv": "w/o speaker adversary", "trustav3_no_distill_no_quality": "w/o face-quality reliability",
        "trustav3_no_distill_no_moddrop": "w/o modality dropout", "trustav3_no_distill_no_contrast": "w/o A-V contrastive loss",
        "trustav3_no_distill_no_trio": "w/o trusted decision fusion"}
ABL1 = {"trustav": "Full TRUST-AV v1", "trustav_no_unc": "w/o uncertainty", "trustav_no_regions": "w/o regions",
        "trustav_no_adv": "w/o speaker adversary", "trustav_no_quality": "w/o quality",
        "trustav_no_moddrop": "w/o modality dropout", "trustav_no_contrast": "w/o contrastive"}
MODES = ["video_missing", "audio_missing", "frame_drop25", "frame_drop50", "frame_drop75", "frame_drop100",
         "audio_noise25", "audio_noise50", "audio_noise75", "audio_noise100"]
MODE_NAME = {"video_missing": "Video missing", "audio_missing": "Audio missing",
             **{f"frame_drop{l}": f"Frame drop {l}%" for l in (25, 50, 75, 100)},
             **{f"audio_noise{l}": f"Audio noise {l}%" for l in (25, 50, 75, 100)}}
TABLES = []                                                    # (id, name, caption, df, lower_is_better cols)


# ----------------------------------------------------------------------------- helpers
def recs(ds, prot, model, seeds):
    out = []
    for s in seeds:
        f = RES / ds / prot / f"{model}_seed{s}.json"
        if not f.exists():
            return None
        out.append(json.loads(f.read_text()))
    return out


def preds(ds, prot, model, seeds):
    keys, y, p, prob, spk = [], [], [], [], []
    for s in seeds:
        f = RES / ds / prot / f"{model}_seed{s}_preds.npz"
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=True)
        o = np.argsort(z["clip_id"].astype(str))
        keys += [f"{s}|{c}" for c in z["clip_id"].astype(str)[o]]
        y.append(z["y"][o]), p.append(z["pred"][o]), spk.append(z["speaker"].astype(str)[o])
        prob.append(z["prob"][o] if "prob" in z.files else None)
    return dict(keys=np.array(keys), y=np.concatenate(y), p=np.concatenate(p), spk=np.concatenate(spk),
                prob=None if any(x is None for x in prob) else np.concatenate(prob))


def ms(v, d=2):
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1):.{d}f}" if len(v) > 1 else f"{np.mean(v):.{d}f}"


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    n, k = n01 + n10, min(n01, n10)
    if n == 0:
        return n01, n10, 1.0
    if n > 1000:
        return n01, n10, math.erfc(((abs(n01 - n10) - 1) / math.sqrt(n)) / math.sqrt(2))
    return n01, n10, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(a, b, spk, reps=1000):
    rng, u = np.random.default_rng(0), np.unique(spk)
    g = {x: np.where(spk == x)[0] for x in u}
    d = [(lambda i: a[i].mean() - b[i].mean())(np.concatenate([g[x] for x in rng.choice(u, len(u))]))
         for _ in range(reps)]
    return np.percentile(d, [2.5, 97.5]) * 100


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def chi2_sf(x, k):
    """survival function of chi-square with k dof (regularised upper gamma, series / continued fraction)."""
    a, x = k / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        s, t, n = 1.0 / a, 1.0 / a, a
        for _ in range(500):
            n += 1
            t *= x / n
            s += t
        return max(0.0, 1 - s * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def ece(prob, y, bins=15):
    c, pr, e = prob.max(1), prob.argmax(1), 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((pr[m] == y[m]).mean() - c[m].mean())
    return e


def add(tid, name, caption, df, lower=(), bold=True, note=""):
    TABLES.append(dict(id=tid, name=name, caption=caption, df=df, lower=set(lower), bold=bold, note=note))


# ----------------------------------------------------------------------------- tables
def t_dataset():
    rows = []
    for ds in DS:
        idx = pd.read_csv(PROC / ds / "features_index.csv")
        idx = idx[idx["has_features"]]
        r0 = None
        prot, seeds = MAIN[ds]
        for m in OURS + ["b11_mamba"]:
            r0 = recs(ds, prot, m, seeds)
            if r0:
                break
        rows.append({
            "Dataset": ds, "Setting": "in-the-wild TV (Friends)" if ds == "MELD" else "acted, studio",
            "Clips used": len(idx), "Speakers": idx["speaker"].nunique(),
            "Classes": len(r0[0]["classes"]) if r0 else idx["label"].nunique(),
            "Class names": ", ".join(r0[0]["classes"]) if r0 else "",
            "Mean duration (s)": round(idx["duration"].mean(), 2),
            "Face found in frames (%)": round(100 * idx["face_ratio"].mean(), 1),
            "Mean face quality": round(idx["mean_quality"].mean(), 3),
            "Speech ratio (%)": round(100 * idx["speech_ratio"].mean(), 1),
            "Multi-face clips (%)": round(100 * (idx["max_faces"] > 1).mean(), 1),
            "Protocol": "official train/dev/test, 3 seeds" if ds == "MELD" else "speaker-independent 5-fold CV",
            "Train / Val / Test (per run)": f"{r0[0]['n_train']} / {r0[0]['n_val']} / {r0[0]['n_test']}" if r0 else ""})
    add("T01", "dataset_statistics", "Dataset statistics after pre-processing (active-speaker face tracking, VAD).",
        pd.DataFrame(rows), bold=False)
    cls = []
    for ds in DS:
        idx = pd.read_csv(PROC / ds / "features_index.csv")
        vc = idx[idx["has_features"]]["label"].value_counts()
        for k, v in vc.items():
            cls.append({"Dataset": ds, "Emotion": k, "Clips": int(v), "Share (%)": round(100 * v / vc.sum(), 1)})
    add("T02", "class_distribution", "Class distribution per dataset.", pd.DataFrame(cls), bold=False)


def t_settings():
    rows = [("Audio encoder", "WavLM-Large (frozen), 13 of 25 hidden layers, learned layer weights, 16 time steps"),
            ("Visual encoder", "CLIP ViT-L/14 vision tower (frozen), face / eyes / mouth crops, 16 frames"),
            ("Face pipeline", "YuNet detection on full-resolution video, shot detection, active-speaker track, aligned crops"),
            ("Model width d", "256"), ("Optimiser", "AdamW, lr 3e-4, weight decay 0.05, gradient clip 1.0"),
            ("Schedule", "1-epoch linear warm-up + cosine decay; max 60 epochs; early stop patience 12"),
            ("Batch size", "64"), ("Loss", "class-weighted (inverse-sqrt) cross-entropy, label smoothing 0.1 (+ model-specific auxiliary terms)"),
            ("Model selection", "best validation accuracy (weighted F1 on MELD); test never used for selection"),
            ("Protocols", "MELD official split, seeds 1-3; CREMA-D / RAVDESS speaker-independent 5-fold CV (fold k = seed k)"),
            ("Speaker normalisation (SAFN)", "per-speaker feature standardisation from unlabelled clips (transductive); CREMA-D / RAVDESS; identical for all models"),
            ("Robustness tests", "video missing, audio missing, frame drop 25-100 %, audio noise 25-100 % (test time only)"),
            ("TRUST-AV hyper-parameters", "2 attention layers, modality dropout 0.2, lambda_EDL 0.5, lambda_adv 0.1, lambda_con 0.1"),
            ("Hardware", "NVIDIA GeForce RTX 5050 Laptop GPU (8 GB), PyTorch 2.11 + CUDA 12.8")]
    add("T03", "training_settings", "Implementation details shared by all models.",
        pd.DataFrame(rows, columns=["Item", "Setting"]), bold=False)


def t_main():
    allrows = []
    for ds in DS:
        prot, seeds = MAIN[ds]
        rows = []
        for m in BASE + OURS:
            r = recs(ds, prot, m, seeds)
            if not r:
                continue
            g = lambda k: [x["test"][k] * 100 for x in r]
            rows.append({"Model": NAME[m], "Acc": np.mean(g("acc")), "Acc SD": np.std(g("acc"), ddof=1),
                         "wF1": np.mean(g("wf1")), "wF1 SD": np.std(g("wf1"), ddof=1), "mF1": np.mean(g("mf1")),
                         "UAR": np.mean(g("uar")), "Params (M)": r[0]["params"] / 1e6, "_m": m})
        df = pd.DataFrame(rows)
        add({"MELD": "T04", "CREMA-D": "T05", "RAVDESS": "T06"}[ds], f"main_results_{ds}",
            f"{ds}: clean test performance (%), mean and SD over {len(seeds)} "
            f"{'seeds' if ds == 'MELD' else 'speaker-independent folds'}"
            f"{' with speaker normalisation' if 'sn' in prot else ''}.",
            df.drop(columns="_m").round(2), lower={"Acc SD", "wF1 SD", "Params (M)"})
        for x in rows:
            allrows.append(dict(Dataset=ds, **{k: v for k, v in x.items()}))
    big = pd.DataFrame(allrows)
    wide = big.pivot_table(index="Model", columns="Dataset", values=["Acc", "wF1"], sort=False)
    wide.columns = [f"{d} {k}" for k, d in wide.columns]
    wide = wide[[f"{d} {k}" for d in DS for k in ("Acc", "wF1")]]
    wide["Mean Acc"] = wide[[f"{d} Acc" for d in DS]].mean(1)
    add("T07", "main_results_summary", "Summary across datasets: accuracy and weighted F1 (%).",
        wide.reset_index().round(2))
    return big


def t_robust():
    summ = []
    for ds in DS:
        prot, seeds = MAIN[ds]
        rows = []
        for m in BASE + OURS:
            r = recs(ds, prot, m, seeds)
            if not r:
                continue
            row = {"Model": NAME[m], "Clean": np.mean([x["test"]["acc"] for x in r]) * 100}
            for md in MODES:
                row[MODE_NAME[md]] = np.mean([x["robust"][md]["acc"] for x in r]) * 100
            row["Missing avg"] = (row["Video missing"] + row["Audio missing"]) / 2
            row["Robust avg (10)"] = np.mean([row[MODE_NAME[md]] for md in MODES])
            row["Drop from clean"] = row["Clean"] - row["Robust avg (10)"]
            rows.append(row)
            summ.append({"Dataset": ds, "Model": NAME[m], "Clean": row["Clean"], "Missing avg": row["Missing avg"],
                         "Robust avg (10)": row["Robust avg (10)"]})
        add({"MELD": "T08", "CREMA-D": "T09", "RAVDESS": "T10"}[ds], f"robustness_{ds}",
            f"{ds}: accuracy (%) under test-time degradation.", pd.DataFrame(rows).round(2),
            lower={"Drop from clean"})
    s = pd.DataFrame(summ).pivot_table(index="Model", columns="Dataset", values=["Missing avg", "Robust avg (10)"],
                                       sort=False)
    s.columns = [f"{d} {k}" for k, d in s.columns]
    s = s[[f"{d} {k}" for d in DS for k in ("Missing avg", "Robust avg (10)")]]
    add("T11", "robustness_summary", "Robustness summary: mean accuracy (%) with one modality missing, and over all 10 degradations.",
        s.reset_index().round(2))


def t_significance():
    rows = []
    for ours in OURS:
        for ds in DS:
            prot, seeds = MAIN[ds]
            A = preds(ds, prot, ours, seeds)
            if A is None:
                continue
            a = A["p"] == A["y"]
            block = []
            for m in BASE + [o for o in OURS if o != ours]:
                B = preds(ds, prot, m, seeds)
                if B is None or not (B["keys"] == A["keys"]).all():
                    continue
                b = B["p"] == B["y"]
                n01, n10, p = mcnemar(a, b)
                lo, hi = boot_ci(a, b, A["spk"])
                block.append({"Ours": NAME[ours], "Dataset": ds, "vs": NAME[m], "Acc diff (pts)": 100 * (a.mean() - b.mean()),
                              "95% CI low": lo, "95% CI high": hi, "Only ours correct": n01, "Only other correct": n10,
                              "McNemar p": p})
            adj = holm([x["McNemar p"] for x in block])
            for x, q in zip(block, adj):
                x["Holm p"] = q
                x["Significant (Holm<0.05)"] = "yes" if q < 0.05 else "no"
            rows += block
    df = pd.DataFrame(rows)
    add("T12", "significance_clean", "Pairwise significance of clean accuracy: McNemar test on pooled test predictions, "
        "95% speaker-cluster bootstrap CI of the accuracy difference, Holm correction per dataset.",
        df.round(4), bold=False)


def t_ranks(big):
    """average ranks over blocks = every (dataset, fold/seed) for accuracy, and for robustness."""
    out = []
    for metric in ["acc", "robust"]:
        blocks = []
        for ds in DS:
            prot, seeds = MAIN[ds]
            models = [m for m in BASE + OURS if recs(ds, prot, m, seeds)]
            for i, s in enumerate(seeds):
                vals = {}
                for m in models:
                    r = recs(ds, prot, m, seeds)[i]
                    vals[m] = r["test"]["acc"] if metric == "acc" else np.mean([r["robust"][x]["acc"] for x in MODES])
                blocks.append(vals)
        common = sorted(set.intersection(*[set(b) for b in blocks]), key=(BASE + OURS).index)
        R = np.array([pd.Series({m: b[m] for m in common}).rank(ascending=False).values for b in blocks])
        N, k = R.shape
        avg = R.mean(0)
        chi = 12 * N / (k * (k + 1)) * (np.sum(avg ** 2) - k * (k + 1) ** 2 / 4)
        p = chi2_sf(chi, k - 1)
        q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
               11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458}.get(k, np.nan)
        cd = q05 * math.sqrt(k * (k + 1) / (6 * N))
        for m, r_ in zip(common, avg):
            out.append({"Criterion": "Clean accuracy" if metric == "acc" else "Robust accuracy (10 degradations)",
                        "Model": NAME[m], "Average rank": r_, "Blocks (dataset x fold)": N,
                        "Friedman chi2": chi, "Friedman p": p, "Nemenyi CD (0.05)": cd})
    df = pd.DataFrame(out).sort_values(["Criterion", "Average rank"])
    add("T13", "average_ranks_friedman", "Average rank of every model over all dataset x fold blocks (1 = best), "
        "Friedman test and Nemenyi critical difference.", df.round(4), lower={"Average rank"}, bold=False)


def t_perclass():
    rows = []
    for ds in DS:
        prot, seeds = MAIN[ds]
        cls = None
        for m in OURS + ["b11_mamba", "b10_saf", "b13_ens", "b4_late"]:
            P = preds(ds, prot, m, seeds)
            r = recs(ds, prot, m, seeds)
            if P is None:
                continue
            cls = r[0]["classes"]
            row = {"Dataset": ds, "Model": NAME[m]}
            for k, c in enumerate(cls):
                tp = ((P["p"] == k) & (P["y"] == k)).sum()
                prec = tp / max(1, (P["p"] == k).sum())
                rec = tp / max(1, (P["y"] == k).sum())
                row[c] = 100 * (2 * prec * rec / max(1e-9, prec + rec))
            rows.append(row)
    add("T14", "per_class_f1", "Per-class F1 (%) of the proposed models and the strongest baselines (pooled test predictions).",
        pd.DataFrame(rows).round(2), bold=False)


def t_calibration():
    rows = []
    for ds in DS:
        prot, seeds = MAIN[ds]
        for m in BASE + OURS:
            P = preds(ds, prot, m, seeds)
            if P is None or P["prob"] is None:
                continue
            q, y = P["prob"], P["y"]
            rows.append({"Dataset": ds, "Model": NAME[m], "ECE (%)": 100 * ece(q, y),
                         "NLL": float(-np.log(q[np.arange(len(y)), y] + 1e-9).mean()),
                         "Brier": float(((q - np.eye(q.shape[1])[y]) ** 2).sum(1).mean()),
                         "Mean confidence (%)": 100 * q.max(1).mean(), "Accuracy (%)": 100 * (q.argmax(1) == y).mean()})
    add("T15", "calibration", "Calibration of predicted probabilities (before any temperature scaling).",
        pd.DataFrame(rows).round(4), lower={"ECE (%)", "NLL", "Brier"}, bold=False)


def t_efficiency():
    rows = []
    for m in BASE + OURS:
        r = recs("CREMA-D", "cv5_sn", m, [1, 2, 3, 4, 5])
        if not r:
            continue
        rows.append({"Model": NAME[m], "Params (M)": r[0]["params"] / 1e6,
                     "Inference (ms / clip, batch 1)": np.mean([x["ms_per_clip"] for x in r]),
                     "Training time (min / fold)": np.mean([x["train_min"] for x in r]),
                     "Epochs to best": np.mean([x["epochs_run"] for x in r])})
    add("T16", "efficiency", "Model size and cost (CREMA-D, mean over folds; features pre-extracted, encoders frozen).",
        pd.DataFrame(rows).round(2), lower={"Params (M)", "Inference (ms / clip, batch 1)", "Training time (min / fold)"})


def summarise(ds, prot, seeds, model, label):
    r = recs(ds, prot, model, seeds)
    if not r:
        return None
    a = lambda k: [x["test"][k] * 100 for x in r]
    nv, na = (np.mean([x["robust"][k]["acc"] for x in r]) * 100 for k in ("video_missing", "audio_missing"))
    return {"Variant": label, "Acc": np.mean(a("acc")), "Acc SD": np.std(a("acc"), ddof=1), "wF1": np.mean(a("wf1")),
            "mF1": np.mean(a("mf1")), "Missing avg": (nv + na) / 2,
            "Robust avg (10)": np.mean([np.mean([x["robust"][k]["acc"] for x in r]) * 100 for k in MODES])}


def t_ablations():
    rows = []
    for ds in DS:
        prot, seeds = MAIN[ds]
        for m, lab in ABL3.items():
            s = summarise(ds, prot, seeds, m, lab)
            if s:
                rows.append({"Dataset": ds, **s})
    if rows:
        df = pd.DataFrame(rows)
        for ds in DS:
            ref = df[(df.Dataset == ds) & (df.Variant == "Full TRUST-AV")]
            if len(ref):
                df.loc[df.Dataset == ds, "Delta Acc"] = df.loc[df.Dataset == ds, "Acc"] - float(ref["Acc"].iloc[0])
                df.loc[df.Dataset == ds, "Delta Robust"] = df.loc[df.Dataset == ds, "Robust avg (10)"] - \
                    float(ref["Robust avg (10)"].iloc[0])
        n = df.groupby("Dataset").size()
        note = "" if (n > 1).all() and len(n) == 3 else \
            "INCOMPLETE: run the ablations3 commands, then re-run make_tables.py"
        add("T17", "ablation_components", "Component ablation of TRUST-AV: each row removes one component from the full model.",
            df.round(2), lower={"Acc SD"}, bold=False, note=note)
    rows = []
    for ds in DS:
        prot, seeds = MAIN[ds]
        for m, lab in [("trustave", "TRUST-AV-E: core + state-space expert, trusted (DS) fusion"),
                       ("trustave_avg", "core + state-space expert, plain averaging"),
                       ("trustav3_no_distill", "core only (TRUST-AV)"), ("b11_mamba", "state-space expert only (= B11)"),
                       ("b13_ens", "two baselines averaged (B13 Mamba+SAF)")]:
            s = summarise(ds, prot, seeds, m, lab)
            if s:
                rows.append({"Dataset": ds, **s})
    add("T18", "ablation_expert_fusion", "Expert-fusion ablation of TRUST-AV-E.", pd.DataFrame(rows).round(2),
        lower={"Acc SD"}, bold=False)
    rows = []
    for ds, prot, seeds in [("CREMA-D", "cv5", [1, 2, 3, 4, 5]), ("MELD", "official", [1, 2, 3])]:
        for m in ["trustav", "trustav2", "trustav2_no_unc", "trustav3", "trustav3_no_trio", "trustav3_no_distill",
                  "trustav4", "trustav4s", "trustav4_no_exp", "trustav4_no_cons"]:
            s = summarise(ds, prot, seeds, m, NAME[m])
            if s:
                rows.append({"Dataset": ds, "Protocol": prot + " (no speaker norm.)" if ds != "MELD" else prot, **s})
    for m in ["trustav3_no_distill", "trustav5"]:
        r = recs("CREMA-D", "cv5_sn", m, [1, 2, 3, 4])
        if r:
            s = summarise("CREMA-D", "cv5_sn", [1, 2, 3, 4], m, NAME[m] + " [folds 1-4]")
            rows.append({"Dataset": "CREMA-D", "Protocol": "cv5_sn, folds 1-4", **s})
    add("T19", "design_evolution", "Development history of the architecture (all variants reported, including those that did not help).",
        pd.DataFrame(rows).round(2), bold=False)
    rows = []
    for m, lab in ABL1.items():
        s = summarise("CREMA-D", "cv5", [1, 2, 3, 4, 5], m, lab)
        if s:
            rows.append({"Dataset": "CREMA-D (cv5, no speaker norm.)", **s})
    add("T20", "ablation_v1_components", "Component ablation of the first TRUST-AV version (supplementary).",
        pd.DataFrame(rows).round(2), bold=False)


def t_safn():
    rows = []
    for ds, a, b, seeds in [("CREMA-D", "cv5", "cv5_sn", [1, 2, 3, 4, 5]), ("RAVDESS", "cv5", "cv5_sn", [1, 2, 3, 4, 5]),
                            ("MELD", "official", "official_sn", [1, 2, 3])]:
        for m in BASE + OURS:
            x, y = recs(ds, a, m, seeds), recs(ds, b, m, seeds)
            if not (x and y):
                continue
            ax, ay = (np.mean([r["test"]["acc"] for r in z]) * 100 for z in (x, y))
            rx, ry = (np.mean([np.mean([r["robust"][k]["acc"] for k in MODES]) for r in z]) * 100 for z in (x, y))
            rows.append({"Dataset": ds, "Model": NAME[m], "Acc without SAFN": ax, "Acc with SAFN": ay, "Delta Acc": ay - ax,
                         "Robust without": rx, "Robust with": ry})
    add("T21", "effect_of_speaker_normalisation", "Effect of speaker-adaptive feature normalisation (SAFN).",
        pd.DataFrame(rows).round(2), bold=False)
    rows = []
    for m in ["trustav3_no_distill", "b10_saf"]:
        for prot in ["cv5", "cv5_ft"]:
            n = 5 if prot == "cv5" or m == "trustav3_no_distill" else 3
            s = summarise("CREMA-D", prot, list(range(1, n + 1)), m, NAME[m])
            if s:
                rows.append({"Audio front": "frozen WavLM" if prot == "cv5" else "WavLM layers 20-23 fine-tuned",
                             "Folds": n, **s})
    add("T22", "wavlm_finetuning", "Frozen vs partially fine-tuned WavLM (CREMA-D, no speaker normalisation).",
        pd.DataFrame(rows).round(2), bold=False)


def t_tuning():
    f = RES / "tuning" / "RAVDESS_tuned_test.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    g = d.groupby("model").agg(Acc=("acc", "mean"), AccSD=("acc", "std"), wF1=("wf1", "mean"), mF1=("mf1", "mean"),
                               NoVideo=("no_video", "mean"), NoAudio=("no_audio", "mean")).reset_index()
    for c in ["Acc", "AccSD", "wF1", "mF1", "NoVideo", "NoAudio"]:
        g[c] *= 100
    best = json.loads((RES / "tuning" / "RAVDESS_best.json").read_text())
    g["Chosen config (validation only)"] = g["model"].map(lambda m: json.dumps(best[m]["cfg"]))
    g["Mean val acc"] = g["model"].map(lambda m: 100 * best[m]["mean_val"])
    g["model"] = g["model"].map(NAME)
    add("T23", "equal_budget_tuning", "Supplementary: equal-budget tuning on RAVDESS (12 configurations per model, "
        "selection on validation only, then 5-fold test).", g.round(2), bold=False)


def t_folds():
    rows = []
    for ds in DS:
        prot, seeds = MAIN[ds]
        for m in BASE + OURS + ["trustave_avg"]:
            r = recs(ds, prot, m, seeds)
            if not r:
                continue
            row = {"Dataset": ds, "Model": NAME[m]}
            for s, x in zip(seeds, r):
                row[f"{'Seed' if ds == 'MELD' else 'Fold'} {s}"] = 100 * x["test"]["acc"]
            rows.append(row)
    add("T24", "per_fold_accuracy", "Supplementary: test accuracy (%) of every fold / seed.", pd.DataFrame(rows).round(2),
        bold=False)


# ----------------------------------------------------------------------------- writing
def to_latex(t):
    df = t["df"]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    best = {}
    if t["bold"]:
        for c in num:
            best[c] = df[c].min() if c in t["lower"] else df[c].max()
    esc = lambda s: str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("±", "$\\pm$")
    cols = [c for c in df.columns if c not in num] + num
    df = df[cols]
    lines = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{esc(t['caption'])}}}", f"\\label{{tab:{t['name']}}}",
             "\\resizebox{\\textwidth}{!}{%", "\\begin{tabular}{" + "l" * (len(cols) - len(num)) + "r" * len(num) + "}",
             "\\toprule", " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in num and pd.notna(v):
                s = f"{v:.2f}" if isinstance(v, float) else str(v)
                if c in best and v == best[c]:
                    s = f"\\textbf{{{s}}}"
                cells.append(s)
            else:
                cells.append(esc(v) if pd.notna(v) else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    steps = [t_dataset, t_settings, t_main, t_robust, t_significance, t_perclass, t_calibration, t_efficiency,
             t_ablations, t_safn, t_tuning, t_folds]
    big = None
    for f in steps:
        try:
            r = f()
            if f is t_main:
                big = r
                t_ranks(big)
        except Exception as e:                                   # one broken table must not stop the rest
            print(f"  ! {f.__name__} failed: {e}")
    TABLES.sort(key=lambda t: t["id"])
    readme = ["Tables for the TRUST-AV paper (generated by code/make_tables.py)", ""]
    with pd.ExcelWriter(OUT / "All_Tables.xlsx", engine="openpyxl") as xw:
        for t in TABLES:
            stem = f"{t['id']}_{t['name']}"
            t["df"].to_csv(OUT / f"{stem}.csv", index=False)
            (OUT / f"{stem}.tex").write_text(to_latex(t), encoding="utf-8")
            sheet = f"{t['id']} {t['name']}"[:31]
            t["df"].to_excel(xw, sheet_name=sheet, index=False, startrow=2)
            ws = xw.sheets[sheet]
            ws.cell(row=1, column=1, value=f"{t['id']}: {t['caption']}" + (f"  [{t['note']}]" if t["note"] else ""))
            readme.append(f"{stem:45s} {len(t['df']):3d} rows  {t['caption']}" + (f"  [{t['note']}]" if t["note"] else ""))
            print(f"  {stem:45s} {len(t['df']):3d} rows" + (f"   <- {t['note']}" if t["note"] else ""))
    (OUT / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    print(f"\n{len(TABLES)} tables written to {OUT}")


if __name__ == "__main__":
    main()
