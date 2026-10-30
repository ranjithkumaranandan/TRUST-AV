"""
make_figures.py - every results figure for the paper, from the saved results (no GPU / torch needed).

Writes to <repo>\\paper\\figures\\  as  Fxx_<name>.png (300 dpi) and .pdf (vector).
Re-run any time after new results (e.g. the ablations) - figures are rebuilt from results\\.
Model-internal figures that need the trained network (Grad-CAM, uncertainty under degradation)
come from explain.py instead.

Usage (from the repository root):
    .\\venv\\Scripts\\python.exe code\\make_figures.py
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap                  # noqa: E402
from matplotlib.patches import FancyBboxPatch                          # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, PROC = ROOT / "results", ROOT / "processed"
OUT = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"

DS = ["MELD", "CREMA-D", "RAVDESS"]
MAIN = {"MELD": ("official", [1, 2, 3]), "CREMA-D": ("cv5_sn", [1, 2, 3, 4, 5]),
        "RAVDESS": ("cv5_sn", [1, 2, 3, 4, 5])}
BASE = ["b1_audio", "b2_video", "b3_early", "b4_late", "b5_tfn", "b6_lmf", "b7_gmu", "b8_mult",
        "b9_jca", "b10_saf", "b11_mamba", "b12_tmc", "b13_ens"]
OURS = ["trustave", "trustav3_no_distill"]
SHORT = {"b1_audio": "Audio", "b2_video": "Video", "b3_early": "Early", "b4_late": "Late", "b5_tfn": "TFN",
         "b6_lmf": "LMF", "b7_gmu": "GMU", "b8_mult": "MulT", "b9_jca": "JCA", "b10_saf": "SAF",
         "b11_mamba": "Mamba", "b12_tmc": "TMC", "b13_ens": "Ens(Mamba+SAF)", "trustave": "TRUST-AV-E",
         "trustav3_no_distill": "TRUST-AV", "trustave_avg": "TRUST-AV-E (avg)"}
# validated categorical slots (light): blue, orange, aqua, yellow, magenta ; baselines recede to grey
KEY = ["trustave", "trustav3_no_distill", "b11_mamba", "b10_saf", "b13_ens"]
COL = {"trustave": "#2a78d6", "trustav3_no_distill": "#eb6834", "b11_mamba": "#1baf7a", "b10_saf": "#eda100",
       "b13_ens": "#e87ba4"}
MARK = {"trustave": "o", "trustav3_no_distill": "s", "b11_mamba": "^", "b10_saf": "D", "b13_ens": "v"}
GREY, INK, INK2, MUTED, GRID = "#b4b2aa", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#ec9c9c", "#f0efec", "#86b6ef", "#1c5cab"])
EMO8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MODES = ["video_missing", "audio_missing"] + [f"frame_drop{l}" for l in (25, 50, 75, 100)] + \
        [f"audio_noise{l}" for l in (25, 50, 75, 100)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.bbox": "tight", "lines.linewidth": 2})
MADE = []


# ----------------------------------------------------------------------------- data helpers
def recs(ds, prot, model, seeds):
    out = []
    for s in seeds:
        f = RES / ds / prot / f"{model}_seed{s}.json"
        if not f.exists():
            return None
        out.append(json.loads(f.read_text()))
    return out


def preds(ds, prot, model, seeds):
    out = []
    for s in seeds:
        f = RES / ds / prot / f"{model}_seed{s}_preds.npz"
        if not f.exists():
            return None
        out.append(dict(np.load(f, allow_pickle=True)))
    cat = lambda k: np.concatenate([z[k] for z in out]) if all(k in z for z in out) else None
    return dict(y=cat("y"), p=cat("pred"), prob=cat("prob"), u_a=cat("u_audio"), u_v=cat("u_video"),
                u_f=cat("u_fused"), region=cat("region_w"), first=out[0])


def M(ds, m):
    prot, seeds = MAIN[ds]
    return recs(ds, prot, m, seeds)


def PM(ds, m):
    prot, seeds = MAIN[ds]
    return preds(ds, prot, m, seeds)


def classes(ds):
    for m in OURS + ["b11_mamba"]:
        r = M(ds, m)
        if r:
            return r[0]["classes"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    MADE.append(name)
    print("  ", name)


def roc_auc(score, pos):
    o = np.argsort(-score)
    pos = pos[o].astype(float)
    tp, fp = np.cumsum(pos), np.cumsum(1 - pos)
    tpr, fpr = np.r_[0, tp / max(pos.sum(), 1)], np.r_[0, fp / max((1 - pos).sum(), 1)]
    return fpr, tpr, float(np.trapz(tpr, fpr))


# ----------------------------------------------------------------------------- figures
def f_training_curves():
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))
    for j, ds in enumerate(DS):
        for m in KEY:
            r = M(ds, m)
            if not r:
                continue
            h = pd.DataFrame(r[0]["history"])
            kw = dict(color=COL[m], label=SHORT[m], lw=1.6 if m != "trustave" else 2.2)
            ax[0, j].plot(h["epoch"], h["train_loss"], **kw)
            vcol = "val_wf1" if ds == "MELD" else "val_acc"
            ax[1, j].plot(h["epoch"], 100 * h[vcol], **kw)
            b = int(np.argmax(h[vcol].values))
            ax[1, j].plot(h["epoch"].iloc[b], 100 * h[vcol].iloc[b], marker=MARK[m], ms=7, color=COL[m],
                          mec="white", mew=1.5, zorder=5)
        ax[0, j].set_title(f"{ds}")
        ax[0, j].set_ylabel("Training loss")
        ax[1, j].set_ylabel("Validation " + ("weighted F1 (%)" if ds == "MELD" else "accuracy (%)"))
        ax[1, j].set_xlabel("Epoch")
    ax[0, 0].legend(loc="upper right")
    fig.suptitle("Training dynamics (fold / seed 1). Loss includes each model's own auxiliary terms, so compare shapes, "
                 "not levels; markers = selected epoch", y=1.0, fontsize=9.5)
    fig.tight_layout()
    save(fig, "F01_training_curves")


def f_confusion():
    for tag, models in [("F02_confusion_TRUST-AV-E", ["trustave"]), ("F03_confusion_Mamba_vs_ours_difference", None)]:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
        for j, ds in enumerate(DS):
            cls = classes(ds)
            K = len(cls)
            P = PM(ds, "trustave")
            if P is None:
                continue
            cm = np.zeros((K, K))
            np.add.at(cm, (P["y"], P["p"]), 1)
            cmn = cm / cm.sum(1, keepdims=True) * 100
            if models is None:
                Q = PM(ds, "b11_mamba")
                cq = np.zeros((K, K))
                np.add.at(cq, (Q["y"], Q["p"]), 1)
                cmn = cmn - cq / cq.sum(1, keepdims=True) * 100
                lim = max(5, np.abs(cmn).max())
                im = ax[j].imshow(cmn, cmap=DIV, vmin=-lim, vmax=lim)
                title = f"{ds}: TRUST-AV-E minus Mamba (pts)"
            else:
                im = ax[j].imshow(cmn, cmap=SEQ, vmin=0, vmax=100)
                title = f"{ds}: TRUST-AV-E (row %)"
            for a in range(K):
                for b in range(K):
                    v = cmn[a, b]
                    dark = (v > 55) if models else abs(v) > 0.6 * lim
                    ax[j].text(b, a, f"{v:.0f}" if models else f"{v:+.1f}", ha="center", va="center", fontsize=7,
                               color="white" if dark else INK)
            ax[j].set_xticks(range(K), cls, rotation=45, ha="right")
            ax[j].set_yticks(range(K), cls)
            ax[j].grid(False)
            ax[j].set_xlabel("Predicted")
            ax[j].set_ylabel("True")
            ax[j].set_title(title)
            fig.colorbar(im, ax=ax[j], fraction=0.046, pad=0.03)
        fig.tight_layout()
        save(fig, tag)


def f_roc():
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.2))
    rows = []
    for j, ds in enumerate(DS):
        cls = classes(ds)
        P = PM(ds, "trustave")
        if P is None or P["prob"] is None:
            continue
        aucs = []
        for k, c in enumerate(cls):
            fpr, tpr, a = roc_auc(P["prob"][:, k], P["y"] == k)
            aucs.append(a)
            ax[j].plot(fpr, tpr, color=EMO8[k % 8], lw=1.4, label=f"{c} ({a:.3f})")
        ax[j].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1)
        ax[j].set_title(f"{ds}: TRUST-AV-E, macro AUC {np.mean(aucs):.3f}")
        ax[j].set_xlabel("False positive rate")
        ax[j].set_ylabel("True positive rate")
        ax[j].legend(loc="lower right", fontsize=7)
        for m in BASE + OURS:
            Q = PM(ds, m)
            if Q is None or Q["prob"] is None:
                continue
            rows.append({"Dataset": ds, "Model": SHORT[m],
                         "Macro AUC": np.mean([roc_auc(Q["prob"][:, k], Q["y"] == k)[2] for k in range(len(cls))])})
    fig.tight_layout()
    save(fig, "F04_roc_curves_TRUST-AV-E")
    df = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(TAB / "T25_macro_auc_all_models.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for j, ds in enumerate(DS):
        d = df[df.Dataset == ds].sort_values("Macro AUC")
        cols = [COL.get(next((k for k, v in SHORT.items() if v == n), ""), GREY) for n in d.Model]
        ax[j].barh(d.Model, d["Macro AUC"], color=cols, height=0.7)
        lo = d["Macro AUC"].min()
        ax[j].set_xlim(max(0.5, lo - 0.03), 1.0 if ds != "MELD" else d["Macro AUC"].max() + 0.02)
        for y, v in enumerate(d["Macro AUC"]):
            ax[j].text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=7, color=INK2)
        ax[j].set_title(f"{ds}: macro one-vs-rest AUC")
        ax[j].grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "F05_macro_auc_all_models")


def f_reliability():
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for j, ds in enumerate(DS):
        for m in ["trustave", "b11_mamba", "b10_saf"]:
            P = PM(ds, m)
            if P is None or P["prob"] is None:
                continue
            c, ok = P["prob"].max(1), P["prob"].argmax(1) == P["y"]
            edges = np.linspace(0, 1, 11)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                s = (c > lo) & (c <= hi)
                if s.sum() >= 10:
                    xs.append(c[s].mean())
                    ys.append(ok[s].mean())
            ax[j].plot(xs, ys, marker=MARK[m], color=COL[m], label=SHORT[m], ms=6, mec="white")
        ax[j].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="perfect calibration")
        ax[j].set_title(f"{ds}: reliability diagram")
        ax[j].set_xlabel("Confidence")
        ax[j].set_ylabel("Accuracy")
        ax[j].set_xlim(0, 1)
        ax[j].set_ylim(0, 1)
    ax[0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, "F06_reliability_diagrams")


def f_robust_curves():
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for j, ds in enumerate(DS):
        for i, kind in enumerate(["frame_drop", "audio_noise"]):
            for m in KEY:
                r = M(ds, m)
                if not r:
                    continue
                y = [np.mean([x["test"]["acc"] for x in r]) * 100] + \
                    [np.mean([x["robust"][f"{kind}{l}"]["acc"] for x in r]) * 100 for l in (25, 50, 75, 100)]
                ax[i, j].plot([0, 25, 50, 75, 100], y, marker=MARK[m], color=COL[m], ms=6, mec="white",
                              label=SHORT[m], lw=2.4 if m == "trustave" else 1.6, zorder=3 if m == "trustave" else 2)
            ax[i, j].set_title(f"{ds}: {'video frames dropped' if kind == 'frame_drop' else 'audio noise'}")
            ax[i, j].set_ylabel("Accuracy (%)")
        ax[1, j].set_xlabel("Degradation level (%)")
    ax[0, 0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "F07_robustness_curves")


def f_acc_vs_robust():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    for j, ds in enumerate(DS):
        for m in BASE + OURS:
            r = M(ds, m)
            if not r:
                continue
            x = np.mean([z["test"]["acc"] for z in r]) * 100
            y = np.mean([(z["robust"]["video_missing"]["acc"] + z["robust"]["audio_missing"]["acc"]) / 2 for z in r]) * 100
            if m in COL:
                ax[j].scatter(x, y, s=90 if m == "trustave" else 60, color=COL[m], marker=MARK[m], edgecolor="white",
                              lw=1.5, zorder=4)
                off = {"trustave": (6, 5), "trustav3_no_distill": (-58, -3)}.get(m, (6, -3))
                ax[j].annotate(SHORT[m], (x, y), xytext=off, textcoords="offset points", fontsize=8,
                               color=INK, fontweight="bold" if m in OURS else "normal")
            else:
                ax[j].scatter(x, y, s=40, color=GREY, edgecolor="white", lw=1, zorder=3)
                ax[j].annotate(SHORT[m], (x, y), xytext=(4, -9), textcoords="offset points", fontsize=7, color=MUTED)
        ax[j].set_title(ds)
        ax[j].set_xlabel("Clean accuracy (%)")
        ax[j].set_ylabel("Accuracy with one modality missing (%)")
    fig.suptitle("Accuracy vs reliability: top-right is better on both", y=1.01, fontsize=10)
    fig.tight_layout()
    save(fig, "F08_accuracy_vs_robustness")


def f_missing_bars():
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    w = 0.16
    conds = [("Clean", None), ("Video missing", "video_missing"), ("Audio missing", "audio_missing")]
    for j, ds in enumerate(DS):
        for k, m in enumerate(KEY):
            r = M(ds, m)
            if not r:
                continue
            vals = [np.mean([z["test"]["acc"] for z in r]) * 100] + \
                   [np.mean([z["robust"][c]["acc"] for z in r]) * 100 for _, c in conds[1:]]
            xs = np.arange(3) + (k - 2) * w
            ax[j].bar(xs, vals, width=w * 0.9, color=COL[m], label=SHORT[m])
        ax[j].set_xticks(range(3), [c for c, _ in conds])
        ax[j].set_ylabel("Accuracy (%)")
        ax[j].set_title(ds)
        ax[j].grid(axis="x", visible=False)
    ax[0].legend(loc="upper right", ncol=1)
    fig.tight_layout()
    save(fig, "F09_missing_modality_bars")


def f_cd():
    f = TAB / "T13_average_ranks_friedman.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, axs = plt.subplots(2, 1, figsize=(10, 7.4))
    titles = {"Clean accuracy": "(a) Clean accuracy",
              "Robust accuracy (10 degradations)": "(b) Accuracy averaged over ten degradations"}
    for ax, crit in zip(axs, d.Criterion.unique()):
        g = d[d.Criterion == crit].sort_values("Average rank").reset_index(drop=True)
        cd = g["Nemenyi CD (0.05)"].iloc[0]
        k = len(g)
        ranks = g["Average rank"].values
        AX = 0.86
        ax.set_xlim(k + 5.5, -4.5)
        ax.set_ylim(0, 1.06)
        ax.plot([1, k], [AX, AX], color=INK2, lw=1)
        for t in range(1, k + 1):
            ax.plot([t, t], [AX, AX + 0.025], color=INK2, lw=1)
            ax.text(t, AX + 0.04, str(t), ha="center", fontsize=7, color=INK2)
        # cliques: maximal groups whose rank range is below the critical difference
        cl = []
        for i in range(k):
            j = i
            while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
                j += 1
            if j > i and not any(a0 <= i and j <= b0 for a0, b0 in cl):
                cl.append((i, j))
        y0 = AX - 0.045
        for n, (i, j) in enumerate(cl):
            yy = y0 - n * 0.028
            ax.plot([ranks[i] - 0.08, ranks[j] + 0.08], [yy, yy], color=INK, lw=2.2, solid_capstyle="round")
        top = y0 - len(cl) * 0.028 - 0.02
        for i, r in g.iterrows():
            left = i < (k + 1) // 2
            row = i if left else k - 1 - i
            yy = top - 0.03 - row * 0.062
            name = r.Model.replace(" (ours)", "*")
            ours = "*" in name
            col = COL["trustave"] if ours else MUTED
            ax.plot([ranks[i]] * 2, [AX, yy], color=col, lw=0.9, zorder=1)
            xe = 0.3 if left else k + 0.7
            ax.plot([ranks[i], xe], [yy, yy], color=col, lw=0.9, zorder=1)
            ax.text(xe - (0.1 if left else -0.1), yy, f"{name} ({ranks[i]:.2f})", va="center",
                    ha="left" if left else "right", fontsize=7.5, color=INK, fontweight="bold" if ours else "normal")
        ax.plot([1, 1 + cd], [1.0, 1.0], color=INK, lw=1.6)
        for xx in (1, 1 + cd):
            ax.plot([xx, xx], [0.985, 1.015], color=INK, lw=1.6)
        ax.text(1 + cd / 2, 1.025, f"CD = {cd:.2f}", ha="center", fontsize=8, color=INK)
        chi = g["Friedman chi2"].iloc[0]
        fp = g["Friedman p"].iloc[0]
        ptxt = "p < 10$^{-10}$" if fp < 1e-10 else f"p = {fp:.1e}"
        ax.set_title(f"{titles.get(crit, crit)}   (Friedman \u03c7$^2$ = {chi:.1f}, {ptxt}; "
                     f"{int(g['Blocks (dataset x fold)'].iloc[0])} dataset\u2013fold blocks)", loc="left", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    save(fig, "F10_critical_difference")


def f_significance():
    f = TAB / "T12_significance_clean.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.Ours.str.contains("-E")]
    base = [b for b in d.vs.unique()]
    mat = np.full((3, len(base)), np.nan)
    sig = np.zeros_like(mat, bool)
    for i, ds in enumerate(DS):
        for j, b in enumerate(base):
            r = d[(d.Dataset == ds) & (d.vs == b)]
            if len(r):
                mat[i, j] = r["Acc diff (pts)"].iloc[0]
                sig[i, j] = r["Holm p"].iloc[0] < 0.05
    fig, ax = plt.subplots(figsize=(12, 2.8))
    lim = np.nanmax(np.abs(np.clip(mat, -6, 6)))
    im = ax.imshow(np.clip(mat, -lim, lim), cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(3):
        for j in range(len(base)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.1f}{'*' if sig[i, j] else ''}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(mat[i, j]) > 0.65 * lim else INK)
    ax.set_xticks(range(len(base)), [b.replace(" (ours)", "") for b in base], rotation=35, ha="right")
    ax.set_yticks(range(3), DS)
    ax.grid(False)
    ax.set_title("TRUST-AV-E minus each model: clean accuracy (pts); * = significant after Holm correction (p < 0.05)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    save(fig, "F11_significance_heatmap")


def f_ablation():
    for fname, out in [("T17_ablation_components", "F12_ablation_components"),
                       ("T18_ablation_expert_fusion", "F13_ablation_expert_fusion")]:
        f = TAB / f"{fname}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if d.groupby("Dataset").size().min() < 2:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 0.45 * d.Variant.nunique() + 1.8), sharey=True)
        vars_ = list(dict.fromkeys(d.Variant))
        w = 0.26
        for k, ds in enumerate(DS):
            g = d[d.Dataset == ds].set_index("Variant").reindex(vars_)
            ys = np.arange(len(vars_)) + (k - 1) * w
            for a, col in zip(ax, ["Acc", "Robust avg (10)"]):
                a.barh(ys, g[col], height=w * 0.9, color=["#2a78d6", "#eb6834", "#1baf7a"][k], label=ds)
        for a, col in zip(ax, ["Clean accuracy (%)", "Robust accuracy, 10 degradations (%)"]):
            a.set_xlabel(col)
            a.grid(axis="y", visible=False)
            lo = np.nanmin(d[["Acc", "Robust avg (10)"]].values)
            a.set_xlim(max(0, lo - 8), None)
        ax[0].set_yticks(range(len(vars_)), vars_)
        ax[0].invert_yaxis()
        ax[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle("Component ablation of TRUST-AV" if "components" in out else "Expert-fusion ablation of TRUST-AV-E",
                     fontsize=10)
        fig.tight_layout()
        save(fig, out)


def f_region_attention():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for j, ds in enumerate(DS):
        P = PM(ds, "trustave")
        cls = classes(ds)
        if P is None or P["region"] is None:
            continue
        mat = np.array([P["region"][P["y"] == k].mean(0) for k in range(len(cls))])
        im = ax[j].imshow(mat, cmap=SEQ, vmin=mat.min() * 0.95, vmax=mat.max(), aspect="auto")
        for a in range(mat.shape[0]):
            for b in range(3):
                ax[j].text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=7.5,
                           color="white" if mat[a, b] > mat.min() + 0.7 * (mat.max() - mat.min()) else INK)
        ax[j].set_xticks(range(3), ["Whole face", "Eyes + brows", "Mouth"])
        ax[j].set_yticks(range(len(cls)), cls)
        ax[j].grid(False)
        ax[j].set_title(f"{ds}: facial-region attention")
        fig.colorbar(im, ax=ax[j], fraction=0.05, pad=0.03)
    fig.tight_layout()
    save(fig, "F14_region_attention_heatmap")


def f_uncertainty():
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.6))
    for j, ds in enumerate(DS):
        P = PM(ds, "trustave")
        if P is None or P["prob"] is None:
            continue
        ok = P["p"] == P["y"]
        u = P["u_f"] if P["u_f"] is not None else 1 - P["prob"].max(1)
        bins = np.linspace(np.percentile(u, 1), np.percentile(u, 99), 30)
        ax[0, j].hist(u[ok], bins=bins, color=COL["trustave"], alpha=0.85, label="correct", density=True)
        ax[0, j].hist(u[~ok], bins=bins, color=COL["trustav3_no_distill"], alpha=0.7, label="wrong", density=True)
        ax[0, j].set_title(f"{ds}: fused evidential uncertainty")
        ax[0, j].set_xlabel("Uncertainty u = K / S")
        ax[0, j].set_ylabel("Density")
        cov = np.linspace(0.1, 1, 19)
        for m in ["trustave", "b11_mamba", "b10_saf"]:
            Q = PM(ds, m)
            if Q is None or Q["prob"] is None:
                continue
            okq = Q["p"] == Q["y"]
            sc = -(Q["u_f"] if (m == "trustave" and Q["u_f"] is not None) else 1 - Q["prob"].max(1))
            o = np.argsort(-sc)
            acc = [okq[o[:max(1, int(c * len(o)))]].mean() * 100 for c in cov]
            ax[1, j].plot(cov * 100, acc, marker=MARK[m], color=COL[m], ms=5, mec="white", label=SHORT[m])
        ax[1, j].set_title(f"{ds}: selective prediction")
        ax[1, j].set_xlabel("Coverage: share of most-certain clips kept (%)")
        ax[1, j].set_ylabel("Accuracy on kept clips (%)")
    ax[0, 0].legend()
    ax[1, 0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "F15_uncertainty_and_selective_prediction")


def tsne(X, perp=30, it=750, seed=0):
    """exact t-SNE (van der Maaten 2008) in numpy; fine for <= 3000 points."""
    rng = np.random.default_rng(seed)
    X = X - X.mean(0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    X = U[:, :50] * S[:50]
    n = len(X)
    D = np.square(X[:, None, :] - X[None, :, :]).sum(-1)
    P = np.zeros((n, n))
    logU = np.log(perp)
    for i in range(n):
        lo, hi, beta = 0.0, np.inf, 1.0
        d = np.delete(D[i], i)
        for _ in range(60):
            p = np.exp(-(d - d.min()) * beta)
            sp = p.sum()
            H = np.log(sp) + beta * (p * (d - d.min())).sum() / sp
            if abs(H - logU) < 1e-4:
                break
            if H > logU:
                lo, beta = beta, beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi, beta = beta, (beta + lo) / 2
        P[i, np.arange(n) != i] = p / sp
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    Y = rng.normal(0, 1e-4, (n, 2))
    V, gains = np.zeros_like(Y), np.ones_like(Y)
    for t in range(it):
        ex = 12.0 if t < 100 else 1.0
        sq = np.square(Y).sum(1)
        num = 1 / (1 + sq[:, None] + sq[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (ex * P - Q) * num
        G = 4 * ((np.diag(PQ.sum(1)) - PQ) @ Y)
        mom = 0.5 if t < 250 else 0.8
        gains = (gains + 0.2) * ((G > 0) != (V > 0)) + gains * 0.8 * ((G > 0) == (V > 0))
        gains = np.maximum(gains, 0.01)
        V = mom * V - 200 * gains * G
        Y = Y + V
        Y -= Y.mean(0)
    return Y


def f_tsne():
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for j, ds in enumerate(DS):
        prot, seeds = MAIN[ds]
        f = RES / ds / prot / f"trustave_seed{seeds[0]}_preds.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        if "emb" not in z.files:
            continue
        E, y = z["emb"].astype(np.float64), z["y"]
        if len(E) > 2000:
            idx = np.random.default_rng(0).choice(len(E), 2000, replace=False)
            E, y = E[idx], y[idx]
        Y = tsne(E)
        cls = classes(ds)
        for k, c in enumerate(cls):
            s = y == k
            ax[j].scatter(Y[s, 0], Y[s, 1], s=7, color=EMO8[k % 8], label=c, alpha=0.8, lw=0)
        ax[j].set_title(f"{ds}: t-SNE, TRUST-AV-E embeddings")
        ax[j].set_xticks([])
        ax[j].set_yticks([])
        ax[j].grid(False)
        ax[j].legend(markerscale=2.5, fontsize=7, loc="best")
    fig.tight_layout()
    save(fig, "F16_tsne_embeddings")


def f_perclass():
    f = TAB / "T14_per_class_f1.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    keep = {"TRUST-AV-E (ours)": "trustave", "TRUST-AV (ours)": "trustav3_no_distill", "B11 Mamba fusion": "b11_mamba",
            "B10 SAF": "b10_saf", "B13 Ensemble (Mamba+SAF)": "b13_ens"}
    for j, ds in enumerate(DS):
        g = d[(d.Dataset == ds) & d.Model.isin(keep)]
        cls = [c for c in classes(ds)]
        w = 0.8 / len(g)
        for k, (_, r) in enumerate(g.iterrows()):
            m = keep[r.Model]
            ax[j].bar(np.arange(len(cls)) + (k - len(g) / 2 + 0.5) * w, [r[c] for c in cls], width=w * 0.92,
                      color=COL[m], label=SHORT[m])
        ax[j].set_xticks(range(len(cls)), cls, rotation=30, ha="right")
        ax[j].set_ylabel("F1 (%)")
        ax[j].set_title(f"{ds}: per-class F1")
        ax[j].grid(axis="x", visible=False)
    ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "F17_per_class_f1")


def f_efficiency():
    f = TAB / "T16_efficiency.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    acc = {}
    for m in BASE + OURS:
        r = M("CREMA-D", m)
        if r:
            acc[m] = np.mean([z["test"]["acc"] for z in r]) * 100
    name2m = {"TRUST-AV-E (ours)": "trustave", "TRUST-AV (ours)": "trustav3_no_distill", "B11 Mamba fusion": "b11_mamba",
              "B10 SAF": "b10_saf", "B13 Ensemble (Mamba+SAF)": "b13_ens"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for _, r in d.iterrows():
        m = name2m.get(r.Model) or next((k for k in BASE if r.Model.startswith(k.split("_")[0].upper() + " ")), None)
        m = m or next((k for k in BASE if SHORT[k] in r.Model), None)
        if m not in acc:
            continue
        for a, col in zip(ax, ["Params (M)", "Inference (ms / clip, batch 1)"]):
            a.scatter(r[col], acc[m], s=70 if m in COL else 40, color=COL.get(m, GREY), marker=MARK.get(m, "o"),
                      edgecolor="white", lw=1.2, zorder=3)
            a.annotate(SHORT[m], (r[col], acc[m]), xytext=(4, -10) if m == "b13_ens" else (4, 3),
                       textcoords="offset points", fontsize=7,
                       color=INK if m in COL else MUTED)
    ax[0].set_xlabel("Parameters (millions, trainable)")
    ax[1].set_xlabel("Inference time (ms per clip, batch 1, RTX 5050 laptop)")
    ax[1].set_xscale("log")
    ax[1].set_xticks([2, 5, 10, 20, 50, 100], ["2", "5", "10", "20", "50", "100"])
    for a in ax:
        a.set_ylabel("CREMA-D accuracy (%)")
    fig.suptitle("Accuracy vs cost (CREMA-D)", fontsize=10)
    fig.tight_layout()
    save(fig, "F18_efficiency")


def f_safn():
    f = TAB / "T21_effect_of_speaker_normalisation.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(d) + 1.2))
    lab = [f"{r.Dataset}: {r.Model.replace(' (ours)', '*')}" for _, r in d.iterrows()]
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r["Acc without SAFN"], r["Acc with SAFN"]], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter(r["Acc without SAFN"], i, color=GREY, s=40, zorder=2, label="without SAFN" if i == 0 else None)
        ax.scatter(r["Acc with SAFN"], i, color=COL["trustave"], s=40, zorder=3, label="with SAFN" if i == 0 else None)
        ax.text(max(r["Acc without SAFN"], r["Acc with SAFN"]) + 0.6, i, f"{r['Delta Acc']:+.1f}", va="center",
                fontsize=7, color=INK2)
    ax.set_yticks(range(len(d)), lab)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Effect of speaker-adaptive feature normalisation (SAFN)")
    fig.tight_layout()
    save(fig, "F19_effect_of_SAFN")


def f_classdist():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for j, ds in enumerate(DS):
        idx = pd.read_csv(PROC / ds / "features_index.csv")
        vc = idx[idx["has_features"]]["label"].value_counts()
        vc = vc.reindex(classes(ds) or vc.index)
        ax[j].bar(vc.index, vc.values, color="#3987e5", width=0.7)
        for x, v in enumerate(vc.values):
            ax[j].text(x, v, f"{v}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax[j].set_title(f"{ds} ({vc.sum()} clips)")
        ax[j].tick_params(axis="x", rotation=30)
        ax[j].set_ylabel("Clips")
        ax[j].grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "F20_class_distribution")


def f_architecture():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#52514e", fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)

    def arr(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1))

    A, V, C, E = "#cde2fb", "#fde0d2", "#d6f1e6", "#fdedc4"
    box(0.1, 4.3, 1.6, 1.0, "Speech\n(16 kHz)", "#f0efec")
    box(0.1, 1.2, 1.6, 1.4, "Video\nactive-speaker\nface track\n(YuNet + shots)", "#f0efec")
    box(2.0, 4.3, 1.9, 1.0, "WavLM-Large\n(frozen, 13 layers)\nlearned layer mix", A)
    box(2.0, 2.2, 1.9, 0.6, "CLIP ViT-L/14: face", V, fs=7.5)
    box(2.0, 1.5, 1.9, 0.6, "CLIP: eyes + brows", V, fs=7.5)
    box(2.0, 0.8, 1.9, 0.6, "CLIP: mouth", V, fs=7.5)
    box(4.2, 0.8, 1.7, 2.0, "Facial-region\nattention\n(per frame)", V)
    box(4.2, 3.2, 1.7, 0.8, "Face-quality\nreliability r_t", V, fs=7.5)
    box(4.2, 4.3, 1.7, 1.0, "Audio tokens\n16 steps", A)
    box(6.2, 4.6, 1.9, 1.2, "Evidential audio head\nDirichlet α_a, u_a", A, fs=7.5)
    box(6.2, 0.4, 1.9, 1.2, "Evidential video head\nDirichlet α_v, u_v", V, fs=7.5)
    box(6.2, 2.0, 1.9, 2.2, "Uncertainty-guided\njoint attention\nbias = log r_t\n+ β·log(relative trust)", C, bold=True)
    box(8.4, 2.0, 1.7, 2.2, "Attention pool\n+ fused head\n(with u_a, u_v,\nconflict)", C)
    box(8.4, 5.95, 1.7, 0.8, "State-space expert\n(bidirectional, A+V)", E, fs=7.5)
    ax.plot([5.05, 5.05, 8.4], [5.3, 6.35, 6.35], color="#52514e", lw=1)
    box(10.4, 2.0, 1.8, 2.2, "Trusted\nDempster-Shafer\nfusion\nfused ⊕ audio ⊕\nvideo ⊕ SSM", "#e6e2fb", bold=True)
    box(12.5, 2.6, 1.4, 1.0, "Emotion +\nuncertainty u", "#f0efec", bold=True)
    box(10.4, 0.2, 3.5, 1.3, "Training only:\nspeaker adversary (GRL)\nA-V contrastive loss\nmodality dropout, evidential loss",
        "#f9f9f7", fs=7.5)
    for (x1, y1, x2, y2) in [(1.7, 4.8, 2.0, 4.8), (1.7, 1.9, 2.0, 2.5), (1.7, 1.9, 2.0, 1.8), (1.7, 1.9, 2.0, 1.1),
                             (3.9, 4.8, 4.2, 4.8), (3.9, 2.5, 4.2, 2.2), (3.9, 1.8, 4.2, 1.8), (3.9, 1.1, 4.2, 1.4),
                             (5.9, 4.8, 6.2, 5.2), (5.9, 1.4, 6.2, 1.0), (5.9, 4.5, 6.2, 3.8), (5.9, 1.8, 6.2, 2.4),
                             (5.9, 3.6, 6.2, 3.1), (7.15, 4.6, 7.15, 4.2), (7.15, 1.6, 7.15, 2.0), (8.1, 3.1, 8.4, 3.1),
                             (8.3, 6.35, 8.4, 6.35), (10.1, 3.1, 10.4, 3.1), (10.1, 6.35, 11.3, 4.2), (8.1, 5.2, 10.6, 4.2),
                             (8.1, 1.0, 10.6, 2.0), (12.2, 3.1, 12.5, 3.1)]:
        arr(x1, y1, x2, y2)
    ax.text(7.0, 6.9, "TRUST-AV-E architecture (TRUST-AV = everything except the state-space expert)",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    save(fig, "F00_architecture")


def main():
    for f in [f_architecture, f_training_curves, f_confusion, f_roc, f_reliability, f_robust_curves, f_acc_vs_robust,
              f_missing_bars, f_cd, f_significance, f_ablation, f_region_attention, f_uncertainty, f_tsne, f_perclass,
              f_efficiency, f_safn, f_classdist]:
        try:
            f()
        except Exception as e:
            import traceback
            print(f"  ! {f.__name__} failed: {e}")
            traceback.print_exc()
    print(f"\n{len(MADE)} figures written to {OUT}")


if __name__ == "__main__":
    main()
