"""
explain.py - model-internal analysis of TRUST-AV-E (needs the GPU; run in the venv on the laptop).

For each dataset it trains TRUST-AV-E once on fold / seed 1 with the same settings as the reported
runs and keeps the weights (results go to a separate "<protocol>_explain" folder, the reported
results are not touched), then produces:

  F21_gradcam_<DATASET>.png        Grad-CAM on the face frames: which facial areas drive the decision
                                   (gradients flow from TRUST-AV-E through the frozen CLIP ViT-L/14)
  F22_trust_under_degradation.png  how the evidential audio / video uncertainty and the video-trust
                                   weight react when audio is noised or video frames are dropped
  F23_attention_to_video.png       share of joint-attention mass on video tokens vs degradation level,
                                   plus per-frame attention for one example clip (clean vs 50 % drop)
  tables T26 / T27 (csv) with the numbers behind F22 / F23

Usage (from the repository root):
    .\\venv\\Scripts\\python.exe code\\explain.py                     # all three datasets (~1 h)
    .\\venv\\Scripts\\python.exe code\\explain.py --datasets ravdess  # one dataset
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                                    # noqa: E402
import extract_features as X                            # noqa: E402

import matplotlib                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG, TAB = ROOT / "paper" / "figures", ROOT / "paper" / "tables"
PLAN = {"ravdess": ("cv5", True), "crema": ("cv5", True), "meld": ("official", False)}
LEVELS = [0, 25, 50, 75, 100]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_A, C_V, C_T = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight", "lines.linewidth": 2})


def make_args(protocol, sn):
    return argparse.Namespace(
        protocol=protocol, seeds=[1], epochs=60, patience=12, bs=64, lr=3e-4, wd=0.05, dim=256, select=None,
        smoke=False, verbose=False, ft_audio=False, lr_ft=3e-5, lam_cons=1.0, spk_norm=sn, hp=None,
        tag="explain", ema=0.0, save_ckpt=True)


def get_model(key, data, raw_face):
    protocol, sn = PLAN[key]
    idx, arr, y, classes = data
    proto = protocol + ("_sn" if sn else "") + "_explain"
    ck = T.RESULTS / T.DATASETS[key] / proto / "trustave_seed1.pt"
    if not ck.exists():
        print(f"  training TRUST-AV-E on {T.DATASETS[key]} fold/seed 1 (weights kept for analysis) ...")
        T.run(make_args(protocol, sn), key, "trustave", 1, data)
    split = T.make_split(idx, y, protocol, 1)
    spk, n_spk = T.speaker_ids(idx, split["train"])
    model = T.build("trustave", len(classes), n_spk, 256).cuda()
    model.load_state_dict(torch.load(ck, map_location="cuda"))
    model.eval()
    return model, split, spk


@torch.no_grad()
def collect(model, arr, y, spk, ix, mode, seed=1, bs=128):
    """predictions + uncertainties + attention share to video tokens under one test-time degradation."""
    rng = np.random.RandomState(seed)
    out = dict(ok=[], ua=[], uv=[], tv=[], att_v=[], att_frames=[])
    for i in range(0, len(ix), bs):
        b = T.get_batch(arr, y, spk, ix[i:i + bs], "cuda")
        if mode:
            b = T.corrupt(b, mode, rng)
        o = model(b)
        w = model.core.blocks[-1].last_weights.float()          # (B, 32 queries, 32 keys), heads averaged
        out["ok"].append((o["logits"].argmax(1) == b["y"]).cpu().numpy())
        out["ua"].append(o["u_audio"].cpu().numpy())
        out["uv"].append(o["u_video"].cpu().numpy())
        out["tv"].append(o["route_av"].cpu().numpy())
        out["att_v"].append(w[:, :, T.T_STEPS:].sum(-1).mean(1).cpu().numpy())
        out["att_frames"].append(w.mean(1).cpu().numpy())        # attention received by each key token
    return {k: np.concatenate(v) for k, v in out.items()}


def degradation_analysis(key, model, arr, y, spk, te, rows_unc, rows_att, examples):
    for kind in ["audio_noise", "frame_drop"]:
        for lv in LEVELS:
            mode = None if lv == 0 else f"{kind}{lv}"
            r = collect(model, arr, y, spk, te, mode)
            rows_unc.append(dict(Dataset=T.DATASETS[key], Degradation=kind, Level=lv, Accuracy=100 * r["ok"].mean(),
                                 u_audio=r["ua"].mean(), u_video=r["uv"].mean(), video_trust=r["tv"].mean()))
            rows_att.append(dict(Dataset=T.DATASETS[key], Degradation=kind, Level=lv,
                                 attention_on_video_tokens=r["att_v"].mean()))
            if kind == "frame_drop" and lv in (0, 50):
                examples.setdefault(T.DATASETS[key], {})[lv] = r["att_frames"][0]


def chunk_stats(x, rows):
    """per-dimension mean / sd over (clips, frames), exactly as SAFN in train.py, without big copies."""
    s1 = np.zeros(x.shape[-1], np.float64)
    s2 = np.zeros(x.shape[-1], np.float64)
    n = 0
    for c in range(0, len(rows), 256):
        xs = x[rows[c:c + 256]].astype(np.float64)
        s1 += xs.sum((0, 1))
        s2 += (xs ** 2).sum((0, 1))
        n += xs.shape[0] * xs.shape[1]
    mu = s1 / n
    return mu.astype(np.float32), (np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-5).astype(np.float32)


def gradcam(key, data, raw_face, model, split, spk, sn, clip):
    idx, arr, y, classes = data
    te = split["test"]
    r = collect(model, arr, y, spk, te, None)
    with torch.no_grad():
        probs = []
        for i in range(0, len(te), 128):
            b = T.get_batch(arr, y, spk, te[i:i + 128], "cuda")
            probs.append(torch.softmax(model(b)["logits"].float(), 1).cpu().numpy())
        probs = np.concatenate(probs)
    pred = probs.argmax(1)
    # one confident, correctly classified clip with a face in every frame for up to 4 classes
    picks = []
    for k in np.argsort(-np.bincount(y[te], minlength=len(classes))):
        cand = [j for j in np.argsort(-probs[:, k]) if pred[j] == k and y[te[j]] == k
                and idx.loc[te[j], "face_ratio"] >= 0.99]
        if cand:
            picks.append(cand[0])
        if len(picks) == 4:
            break
    if sn:
        g_mu, g_sd = chunk_stats(raw_face, np.arange(len(raw_face)))
    spk_all = idx["speaker"].astype(str).values
    fig, ax = plt.subplots(len(picks), 5, figsize=(12, 2.6 * len(picks)))
    ax = np.atleast_2d(ax)
    for row, j in enumerate(picks):
        gi = te[j]
        cid = idx.loc[gi, "clip_id"]
        tiles = X.split_grid(T.PROC / T.DATASETS[key] / "faces" / f"{cid}_face.jpg", X.FACE_SIZE)
        x = torch.from_numpy(((tiles.astype(np.float32) / 255.0 - X.CLIP_MEAN) / X.CLIP_STD)
                             .transpose(0, 3, 1, 2)).cuda().requires_grad_(True)
        out = clip(pixel_values=x, output_hidden_states=True)
        H = out.hidden_states[-2]                                  # input to the last ViT block
        H.retain_grad()
        feat = out.pooler_output
        ii = np.flatnonzero(spk_all == spk_all[gi])
        if sn and len(ii) >= 5:                                    # same speaker normalisation as training
            s_mu, s_sd = chunk_stats(raw_face, ii)
            feat = (feat - torch.from_numpy(s_mu).cuda()) / torch.from_numpy(s_sd).cuda() * \
                torch.from_numpy(g_sd).cuda() + torch.from_numpy(g_mu).cuda()
        b = T.get_batch(arr, y, spk, np.array([gi]), "cuda")
        b["face"] = feat[None].float()
        model.zero_grad(set_to_none=True)
        logit = model(b)["logits"][0, pred[j]]
        logit.backward()
        A, G = H[:, 1:, :].detach(), H.grad[:, 1:, :]
        cam = F.relu((A * G.mean(1, keepdim=True)).sum(-1))         # (16 frames, 256 patches)
        cam = cam / cam.max().clamp_min(1e-8)
        n = int(round(cam.shape[1] ** 0.5))
        cam = cam.reshape(-1, 1, n, n)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
        energy = cam.reshape(16, -1).mean(1)
        frames = np.sort(np.argsort(-energy)[:4])
        ax[row, 0].imshow(tiles[frames[0]])
        ax[row, 0].set_title(f"{classes[y[gi]]} (p={probs[j, pred[j]]:.2f})", fontsize=9)
        for c, t in enumerate(frames):
            ax[row, c + 1].imshow(tiles[t])
            ax[row, c + 1].imshow(cam[t], cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax[row, c + 1].set_title(f"frame {t + 1}", fontsize=8)
        for a in ax[row]:
            a.set_xticks([]), a.set_yticks([]), a.grid(False)
        del out, H
        torch.cuda.empty_cache()
    fig.suptitle(f"{T.DATASETS[key]}: Grad-CAM of TRUST-AV-E on the face stream (first column: input; "
                 f"then the four most relevant frames)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F21_gradcam_{T.DATASETS[key]}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  F21_gradcam_{T.DATASETS[key]}")


def plot_degradation(unc, att, examples):
    ds = list(dict.fromkeys(unc.Dataset))
    fig, ax = plt.subplots(2, len(ds), figsize=(4.2 * len(ds), 6.4), squeeze=False)
    for j, d in enumerate(ds):
        for i, kind in enumerate(["audio_noise", "frame_drop"]):
            g = unc[(unc.Dataset == d) & (unc.Degradation == kind)]
            a = ax[i, j]
            a.plot(g.Level, g.u_audio, marker="o", color=C_A, label="audio uncertainty u_a", mec="white")
            a.plot(g.Level, g.u_video, marker="s", color=C_V, label="video uncertainty u_v", mec="white")
            a.plot(g.Level, g.video_trust / 2, marker="^", color=C_T, label="video trust / 2", mec="white")
            a.set_title(f"{d}: {'audio noise' if kind == 'audio_noise' else 'video frames dropped'}")
            a.set_xlabel("Degradation level (%)")
            a.set_ylabel("Mean value on test set")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Evidential uncertainty follows the corrupted modality, and trust shifts to the clean one", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F22_trust_under_degradation.{ext}", dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for d, col, mk in zip(ds, [C_A, C_V, C_T], "os^"):
        for kind, ls in [("frame_drop", "-"), ("audio_noise", "--")]:
            g = att[(att.Dataset == d) & (att.Degradation == kind)]
            ax[0].plot(g.Level, 100 * g.attention_on_video_tokens, ls=ls, marker=mk, color=col, mec="white",
                       label=f"{d}, {'frame drop' if kind == 'frame_drop' else 'audio noise'}")
    ax[0].set_xlabel("Degradation level (%)")
    ax[0].set_ylabel("Attention mass on video tokens (%)")
    ax[0].set_title("Joint attention re-routes away from the degraded modality")
    ax[0].legend(fontsize=7)
    d0 = ds[0]
    for lv, col in [(0, C_A), (50, C_V)]:
        if lv in examples.get(d0, {}):
            w = examples[d0][lv]
            ax[1].bar(np.arange(32) + (0.2 if lv else -0.2), w, width=0.4, color=col,
                      label="clean" if lv == 0 else "50 % frames dropped")
    ax[1].axvline(15.5, color=MUTED, lw=1, ls=":")
    ax[1].text(7.5, ax[1].get_ylim()[1] * 0.95, "audio tokens", ha="center", color=INK2, fontsize=8)
    ax[1].text(23.5, ax[1].get_ylim()[1] * 0.95, "video tokens", ha="center", color=INK2, fontsize=8)
    ax[1].set_xlabel("Key token (time step)")
    ax[1].set_ylabel("Attention received")
    ax[1].set_title(f"{d0}: one test clip, clean vs 50 % frames dropped")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F23_attention_to_video.{ext}", dpi=300)
    plt.close(fig)
    print("  F22_trust_under_degradation, F23_attention_to_video")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["ravdess", "crema", "meld"], choices=list(PLAN))
    ap.add_argument("--no_gradcam", action="store_true")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    clip = None
    if not args.no_gradcam:
        from transformers import CLIPVisionModel
        clip = CLIPVisionModel.from_pretrained(X.VISION_MODEL).cuda().eval()
        for p in clip.parameters():
            p.requires_grad_(False)
    rows_unc, rows_att, examples = [], [], {}
    for key in args.datasets:
        protocol, sn = PLAN[key]
        print(f"\n=== {T.DATASETS[key]} ===")
        data = T.load_dataset(key)
        raw_face = data[1]["face"].copy() if sn else data[1]["face"]
        if sn:
            T.speaker_normalise(data[0], data[1])
        model, split, spk = get_model(key, data, raw_face)
        degradation_analysis(key, model, data[1], data[2], spk, split["test"], rows_unc, rows_att, examples)
        if clip is not None:
            gradcam(key, data, raw_face, model, split, spk, sn, clip)
        del data, raw_face
    unc, att = pd.DataFrame(rows_unc), pd.DataFrame(rows_att)
    unc.round(4).to_csv(TAB / "T26_uncertainty_under_degradation.csv", index=False)
    att.round(4).to_csv(TAB / "T27_attention_on_video_tokens.csv", index=False)
    plot_degradation(unc, att, examples)
    print("\nDone. Figures in", FIG)


if __name__ == "__main__":
    main()
