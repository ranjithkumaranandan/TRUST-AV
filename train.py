"""
train.py - Step 4: train and evaluate the 12 baselines and TRUST-AV.

All models read the same frozen features (extract_features.py) and are trained
with the same optimiser, schedule, loss weighting, early stopping and metrics,
so the comparison is fair.

Models
  b1_audio     WavLM audio only (layer-weighted, temporal transformer)
  b2_video     CLIP face only
  b3_early     early fusion (concatenate per-step features)
  b4_late      late fusion (average of unimodal logits)
  b5_tfn       Tensor Fusion Network (Zadeh et al., 2017)
  b6_lmf       Low-rank Multimodal Fusion (Liu et al., 2018)
  b7_gmu       Gated Multimodal Unit (Arevalo et al., 2017)
  b8_mult      Multimodal Transformer, cross-modal attention (Tsai et al., 2019)
  b9_jca       joint cross-attention, re-implementation (after Praveen et al., 2022)
  b10_saf      self-attention fusion with modality dropout (after Chumachenko et al., 2022)
  b11_mamba    selective state-space (Mamba-style) fusion, pure PyTorch
  b12_tmc      Trusted Multi-View Classification, evidential late fusion (Han et al., 2021)
  trustav      proposed TRUST-AV
  trustav_no_unc / _no_regions / _no_adv / _no_quality / _no_moddrop / _no_contrast
               ablations (one component removed each)
  trustav5     TRUST-AV v5 (hybrid) = bidirectional state-space temporal stage over interleaved
               audio/video tokens + v3 trusted fusion (ablations: trustav5_no_ssm, trustav5_no_unc,
               trustav5_no_moddrop, trustav5_no_trio, ...)
  trustav4     TRUST-AV v4 = v3 (no distill) + 3 trusted experts (token-subset heads, DS-combined)
               + degradation-consistency training; ablations trustav4_no_exp, trustav4_no_cons;
               trustav4s = same at d=160 (parameter-matched to the baselines)
  trustav3     TRUST-AV v3 = v2 + trusted (Dempster-Shafer) decision fusion of fused/audio/video
               opinions + self-distillation (ablations: trustav3_no_trio, trustav3_no_distill)
  trustav2     TRUST-AV v2: joint self-attention backbone with uncertainty- and
               quality-biased attention logits (ablations: trustav2_no_unc, ... or "ablations2")

Protocols
  official : MELD official split; CREMA-D / RAVDESS fixed speaker-independent split
  cv5      : speaker-independent 5-fold cross-validation (seed k = fold k), CREMA-D / RAVDESS
  random   : stratified random 70/10/20 split (speakers overlap, literature setting)

Metrics: accuracy, weighted-F1, macro-F1, UAR (unweighted average recall), plus
robustness on the test set: video missing, audio missing, 50 % frames dropped.

Results -> <repo>\\results\\<dataset>\\<protocol>\\<model>_seed<k>.json
           <repo>\\results\\summary.csv (one row per run)

Usage (inside the venv, from the repository root):
    python code\\train.py --dataset crema --model all --smoke     # 2-minute check of every model
    python code\\train.py --dataset crema --model b1_audio b2_video
    python code\\train.py --dataset crema --model baselines --seeds 1 2 3 4 5
    python code\\train.py --dataset meld  --model trustav ablations --seeds 1 2 3 4 5
    python code\\train.py --dataset ravdess --model all --protocol random
    python code\\train.py --dataset crema --model trustav3_no_distill b10_saf --protocol cv5 --ft_audio
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
RESULTS = ROOT / "results"
DATASETS = {"meld": "MELD", "crema": "CREMA-D", "ravdess": "RAVDESS"}
T_STEPS, N_LAYERS, FEAT = 16, 13, 1024

BASELINES = ["b1_audio", "b2_video", "b3_early", "b4_late", "b5_tfn", "b6_lmf", "b7_gmu",
             "b8_mult", "b9_jca", "b10_saf", "b11_mamba", "b12_tmc"]
ABLATIONS = ["trustav_no_unc", "trustav_no_regions", "trustav_no_adv",
             "trustav_no_quality", "trustav_no_moddrop", "trustav_no_contrast"]
ABLATIONS2 = [a.replace("trustav", "trustav2") for a in ABLATIONS]
# ablations of the FINAL model (trustav3_no_distill): each removes one more component
ABLATIONS3 = ["trustav3_no_distill_no_" + c for c in
              ["unc", "regions", "adv", "quality", "moddrop", "contrast", "trio"]]


# =============================================================================
# Data
# =============================================================================
def load_dataset(key, skip_audio=False):
    """Load all features into memory; a .npy cache makes later runs start in seconds."""
    name = DATASETS[key]
    idx = pd.read_csv(PROC / name / "features_index.csv")
    idx = idx[idx["has_features"]].reset_index(drop=True)
    cache = PROC / name / "cache"
    cache.mkdir(exist_ok=True)
    files = {k: cache / f"{k}.npy" for k in ["audio", "face", "eyes", "mouth", "mask", "quality"]}
    ids_file = cache / "clip_ids.txt"
    if all(f.exists() for f in files.values()) and ids_file.exists() and \
            ids_file.read_text().split("\n") == idx["clip_id"].tolist():
        arr = {k: np.load(f) for k, f in files.items() if not (skip_audio and k == "audio")}
    else:
        n = len(idx)
        print(f"Building feature cache for {name} ({n} clips, one time) ...")
        arr = {"audio": np.zeros((n, T_STEPS, N_LAYERS, FEAT), np.float16),
               "face": np.zeros((n, T_STEPS, FEAT), np.float16),
               "eyes": np.zeros((n, T_STEPS, FEAT), np.float16),
               "mouth": np.zeros((n, T_STEPS, FEAT), np.float16),
               "mask": np.zeros((n, T_STEPS), np.float16),
               "quality": np.zeros((n, T_STEPS), np.float16)}
        for i, p in enumerate(idx["feature_path"]):
            p = Path(p)
            if not p.exists():                          # path written on another drive letter
                p = PROC / name / "features" / p.name
            z = np.load(p)
            arr["audio"][i], arr["face"][i] = z["audio"], z["face"]
            arr["eyes"][i], arr["mouth"][i] = z["eyes"], z["mouth"]
            arr["mask"][i], arr["quality"][i] = z["face_mask"], z["quality"]
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{n}")
        for k, f in files.items():
            np.save(f, arr[k])
        ids_file.write_text("\n".join(idx["clip_id"].tolist()))
    classes = sorted(idx["label"].unique())
    y = idx["label"].map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)
    return idx, arr, y, classes


def speaker_normalise(idx, arr, min_clips=5):
    """Speaker-adaptive feature normalisation (SAFN): re-centre and re-scale every feature
    stream per speaker (no labels used; test speakers use their own unlabelled clips), then map
    back to the global mean/scale. Speakers with < min_clips clips keep their features."""
    spk = idx["speaker"].astype(str).values
    for k in ["audio", "face", "eyes", "mouth"]:
        if k not in arr:
            continue
        x = arr[k]

        def stats(rows):                                # chunked: never copies the whole array
            s1 = np.zeros(x.shape[2:], np.float64)
            s2 = np.zeros(x.shape[2:], np.float64)
            n = 0
            for c in range(0, len(rows), 256):
                xs = x[rows[c:c + 256]].astype(np.float32)
                s1 += xs.sum((0, 1))
                s2 += (xs.astype(np.float64) ** 2).sum((0, 1))
                n += xs.shape[0] * xs.shape[1]
            mu = s1 / n
            return mu.astype(np.float32), np.sqrt(np.maximum(s2 / n - mu ** 2, 0)).astype(np.float32) + 1e-5

        g_mu, g_sd = stats(np.arange(len(x)))
        for s in np.unique(spk):
            ii = np.flatnonzero(spk == s)
            if len(ii) < min_clips:
                continue
            mu, sd = stats(ii)
            for c in range(0, len(ii), 256):
                rows = ii[c:c + 256]
                x[rows] = ((x[rows].astype(np.float32) - mu) / sd * g_sd + g_mu).astype(x.dtype)
    print(f"SAFN: normalised {len(np.unique(spk))} speakers (min {min_clips} clips)")


def load_mid(key, n):
    """Frame-level WavLM states (extract_wavlm_mid.py) kept in RAM."""
    d = PROC / DATASETS[key] / "wavlm_mid"
    meta = json.loads((d / "meta.json").read_text())
    if meta["clips"] != n:
        raise SystemExit(f"wavlm_mid has {meta['clips']} clips but the index has {n}: re-run extract_wavlm_mid.py")
    off = np.load(d / "offsets.npy")
    mid = np.fromfile(d / "feats.bin", dtype=np.float16).reshape(-1, meta["dim"])
    print(f"Loaded frame-level WavLM states: {mid.shape[0]} frames ({mid.nbytes / 1e9:.1f} GB)")
    return mid, off


def make_split(idx, y, protocol, seed):
    if protocol == "official":
        return {s: np.flatnonzero(idx["split"].values == s) for s in ["train", "val", "test"]}
    if protocol == "cv5":                              # speaker-independent 5-fold CV; seed k -> fold k
        spk = idx["speaker"].astype(str).values
        speakers = np.array(sorted(set(spk)))
        order = np.random.RandomState(0).permutation(len(speakers))   # same folds for every model
        groups = np.array_split(speakers[order], 5)
        fold = (seed - 1) % 5
        test_spk = set(groups[fold])
        rest = [s for s in speakers[order] if s not in test_spk]
        n_val = max(1, int(round(len(rest) / 8)))
        val_spk = set(np.random.RandomState(100 + fold).permutation(rest)[:n_val])
        te = np.flatnonzero(np.isin(spk, list(test_spk)))
        va = np.flatnonzero(np.isin(spk, list(val_spk)))
        tr = np.flatnonzero(~np.isin(spk, list(test_spk | val_spk)))
        return {"train": tr, "val": va, "test": te}
    rng = np.random.RandomState(seed)                  # stratified random 70 / 10 / 20
    parts = {"train": [], "val": [], "test": []}
    for c in np.unique(y):
        ii = rng.permutation(np.flatnonzero(y == c))
        n_tr, n_va = int(round(0.7 * len(ii))), int(round(0.1 * len(ii)))
        parts["train"] += ii[:n_tr].tolist()
        parts["val"] += ii[n_tr:n_tr + n_va].tolist()
        parts["test"] += ii[n_tr + n_va:].tolist()
    return {k: np.array(sorted(v)) for k, v in parts.items()}


def speaker_ids(idx, train_ix, min_count=20):
    """Speaker labels for the adversary (train speakers with >= min_count clips, else 'other')."""
    spk = idx["speaker"].astype(str).values
    counts = pd.Series(spk[train_ix]).value_counts()
    keep = sorted(counts[counts >= min_count].index)
    m = {s: i for i, s in enumerate(keep)}
    other = len(keep)
    ids = np.array([m.get(s, other) for s in spk], np.int64)
    return ids, other + 1


_AMASK = None          # frame mask of the current batch, read by FTAudioFront


def get_batch(arr, y, spk, ix, device):
    global _AMASK
    keys = ["face", "eyes", "mouth", "mask", "quality"] + ([] if "mid" in arr else ["audio"])
    b = {k: torch.from_numpy(arr[k][ix].astype(np.float32)).to(device, non_blocking=True) for k in keys}
    if "mid" in arr:                                     # padded frame-level states + mask
        off, mid = arr["off"], arr["mid"]
        lens = off[ix + 1] - off[ix]
        buf = np.zeros((len(ix), int(lens.max()), mid.shape[1]), np.float32)
        m = np.zeros((len(ix), int(lens.max())), np.float32)
        for j, i in enumerate(ix):
            buf[j, :lens[j]] = mid[off[i]:off[i + 1]]
            m[j, :lens[j]] = 1
        b["audio"] = torch.from_numpy(buf).to(device, non_blocking=True)
        b["amask"] = torch.from_numpy(m).to(device, non_blocking=True)
        _AMASK = b["amask"]
    b["y"] = torch.from_numpy(y[ix]).to(device)
    b["spk"] = torch.from_numpy(spk[ix]).to(device)
    return b


def corrupt(b, mode, rng):
    """Test-time corruptions for robustness evaluation."""
    b = dict(b)
    if mode == "video_missing":
        for k in ["face", "eyes", "mouth", "mask", "quality"]:
            b[k] = torch.zeros_like(b[k])
    elif mode == "audio_missing":
        b["audio"] = torch.zeros_like(b["audio"])
    elif mode.startswith("frame_drop"):                 # frame_drop50 -> drop 50 % of frames
        p = int(mode.replace("frame_drop", "")) / 100.0
        keep = torch.from_numpy((rng.rand(*b["mask"].shape) >= p).astype(np.float32)).to(b["mask"].device)
        for k in ["face", "eyes", "mouth"]:
            b[k] = b[k] * keep[..., None]
        b["mask"], b["quality"] = b["mask"] * keep, b["quality"] * keep
    elif mode.startswith("audio_noise"):                # audio_noise50 -> noise std = 0.5 x feature std
        s = int(mode.replace("audio_noise", "")) / 100.0
        a = b["audio"]
        std = a.std(dim=tuple(range(1, a.dim())), keepdim=True)
        g = torch.from_numpy(rng.randn(*a.shape).astype(np.float32)).to(a.device)
        b["audio"] = a + s * std * g
    return b


CURVE_LEVELS = [0, 25, 50, 75, 100]


# =============================================================================
# Metrics (plain numpy - no scikit-learn needed)
# =============================================================================
def metrics(y, p, K):
    cm = np.zeros((K, K), np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1).astype(float), cm.sum(0).astype(float)
    prec = np.divide(tp, predicted, out=np.zeros(K), where=predicted > 0)
    rec = np.divide(tp, support, out=np.zeros(K), where=support > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    present = support > 0
    return dict(acc=float(tp.sum() / max(len(y), 1)),
                wf1=float((f1 * support).sum() / max(support.sum(), 1)),
                mf1=float(f1[present].mean()) if present.any() else 0.0,
                uar=float(rec[present].mean()) if present.any() else 0.0,
                per_class_f1=f1.round(4).tolist(), confusion=cm.tolist())


# =============================================================================
# Building blocks
# =============================================================================
class AudioFront(nn.Module):
    """Learned softmax weighting of the 13 WavLM layers, then projection."""
    def __init__(self, d, p=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(N_LAYERS))
        self.norm = nn.LayerNorm(FEAT)
        self.proj = nn.Sequential(nn.Linear(FEAT, d), nn.GELU(), nn.Dropout(p))

    def forward(self, a):                              # (B, T, L, F)
        w = torch.softmax(self.w, 0)
        return self.proj(self.norm((a * w[None, None, :, None]).sum(2)))


class FTAudioFront(nn.Module):
    """WavLM-Large layers CUT..23 + final norm, fine-tuned jointly with the model, on frame-level
    states from extract_wavlm_mid.py; learned weighting of the top states; masked pooling to
    T_STEPS tokens so every model sees the same token grid."""
    CUT = 20

    def __init__(self, d, p=0.1):
        super().__init__()
        from transformers import WavLMModel
        w = WavLMModel.from_pretrained("microsoft/wavlm-large")
        enc = w.encoder
        self.top = nn.ModuleList([enc.layers[i] for i in range(self.CUT, len(enc.layers))])
        self.final_norm = enc.layer_norm
        self.bias_attn = enc.layers[0].attention              # only for compute_bias (frozen)
        for q in self.bias_attn.parameters():
            q.requires_grad = False
        self.nh = self.bias_attn.num_heads
        self.lw = nn.Parameter(torch.zeros(len(self.top) + 2))
        self.norm = nn.LayerNorm(FEAT)
        self.proj = nn.Sequential(nn.Linear(FEAT, d), nn.GELU(), nn.Dropout(p))
        del w

    def forward(self, a):                                  # a: (B, T, 1024)
        m = _AMASK
        B, T, _ = a.shape
        with torch.no_grad():
            pb = self.bias_attn.compute_bias(T, T)
        pb = pb.unsqueeze(0).repeat(B, 1, 1, 1).view(B * self.nh, T, T)
        am = m.long()
        h, states = a, [a]
        for layer in self.top:
            h, pb = layer(h, attention_mask=am, position_bias=pb)
            states.append(h)
        states.append(self.final_norm(h))
        w = torch.softmax(self.lw, 0)
        x = sum(wi * si for wi, si in zip(w, states))
        x = self.proj(self.norm(x))                        # (B, T, d)
        L = m.sum(1).clamp_min(1)
        seg = (torch.arange(T, device=a.device)[None, :] * T_STEPS / L[:, None]).long().clamp(max=T_STEPS - 1)
        oh = F.one_hot(seg, T_STEPS).float() * m[..., None]
        return torch.einsum("btk,btd->bkd", oh, x) / oh.sum(1)[..., None].clamp_min(1)


class VisFront(nn.Module):
    def __init__(self, d, p=0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(FEAT), nn.Linear(FEAT, d), nn.GELU(), nn.Dropout(p))

    def forward(self, v):
        return self.proj(v)


class TempEnc(nn.Module):
    def __init__(self, d, layers=1, heads=4, p=0.1, length=T_STEPS):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, length, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, p, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        return self.norm(self.enc(x + self.pos[:, :x.shape[1]]))


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x, bias=None):                   # bias: (B, T) added to the logits
        s = self.score(x).squeeze(-1)
        if bias is not None:
            s = s + bias
        w = torch.softmax(s, -1)
        return (w.unsqueeze(-1) * x).sum(1)


class CrossBlock(nn.Module):
    """Query sequence attends to a context sequence; returns the update (no residual)."""
    def __init__(self, d, heads=4, p=0.1):
        super().__init__()
        self.h = heads
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p),
                                nn.Linear(2 * d, d))

    def forward(self, q, kv, key_bias=None):
        mask = None
        if key_bias is not None:                       # (B, S) -> (B*heads, L, S)
            B, S = key_bias.shape
            mask = key_bias[:, None, None, :].expand(B, self.h, q.shape[1], S)
            mask = mask.reshape(B * self.h, q.shape[1], S).to(q.dtype)
        h, w = self.attn(self.nq(q), self.nk(kv), self.nk(kv), attn_mask=mask,
                         need_weights=True, average_attn_weights=True)
        self.last_weights = w.detach()
        return h + self.ff(h)


def mlp(i, o, d, p=0.2):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, o))


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def edl_loss(alpha, y, K, anneal):
    """Evidential loss (Sensoy et al., 2018) as used in TMC."""
    alpha = alpha.float()
    yo = F.one_hot(y, K).float()
    S = alpha.sum(1, keepdim=True)
    ce = (yo * (torch.digamma(S) - torch.digamma(alpha))).sum(1)
    at = yo + (1 - yo) * alpha
    St = at.sum(1)
    kl = (torch.lgamma(St) - torch.lgamma(at).sum(1) - math.lgamma(K)
          + ((at - 1) * (torch.digamma(at) - torch.digamma(St[:, None]))).sum(1))
    return (ce + anneal * kl).mean()


def ds_combine(a1, a2, K):
    """Dempster-Shafer combination of two Dirichlet opinions (TMC)."""
    S1, S2 = a1.sum(1, keepdim=True), a2.sum(1, keepdim=True)
    b1, b2 = (a1 - 1) / S1, (a2 - 1) / S2
    u1, u2 = K / S1, K / S2
    C = b1.sum(1, keepdim=True) * b2.sum(1, keepdim=True) - (b1 * b2).sum(1, keepdim=True)
    b = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - C).clamp_min(1e-6)
    u = (u1 * u2) / (1 - C).clamp_min(1e-6)
    S = K / u.clamp_min(1e-6)
    return b * S + 1


class MambaLite(nn.Module):
    """Selective state-space block (Mamba-style S6), pure PyTorch, sequential scan."""
    def __init__(self, d, d_state=16, expand=2):
        super().__init__()
        di = expand * d
        self.di, self.ns, self.rank = di, d_state, max(1, d // 16)
        self.inp = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, 3, padding=2, groups=di)
        self.xp = nn.Linear(di, self.rank + 2 * d_state)
        self.dtp = nn.Linear(self.rank, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, d)

    def forward(self, x):
        B, L, _ = x.shape
        x, z = self.inp(x).chunk(2, -1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = torch.split(self.xp(x), [self.rank, self.ns, self.ns], -1)
        dt = F.softplus(self.dtp(dt))
        A = -torch.exp(self.A_log)
        h = x.new_zeros(B, self.di, self.ns)
        ys = []
        for t in range(L):
            h = torch.exp(dt[:, t, :, None] * A) * h + dt[:, t, :, None] * Bm[:, t, None, :] * x[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.D * x[:, t])
        return self.out(torch.stack(ys, 1) * F.silu(z))


# =============================================================================
# Models
# =============================================================================
class Base(nn.Module):
    """Common interface: forward(batch) -> dict(logits, aux, override?)."""
    progress = 0.0                                      # training progress in [0, 1]

    def set_progress(self, p):
        self.progress = p


class Unimodal(Base):
    def __init__(self, K, d, modality):
        super().__init__()
        self.m = modality
        self.front = AudioFront(d) if modality == "audio" else VisFront(d)
        self.enc, self.pool, self.head = TempEnc(d, 2), AttnPool(d), mlp(d, K, d)

    def forward(self, b):
        x = self.front(b["audio"] if self.m == "audio" else b["face"])
        e = self.pool(self.enc(x))
        return dict(logits=self.head(e), emb=e)


class TwoStream(Base):
    """Shared unimodal encoders used by most fusion baselines."""
    def __init__(self, d):
        super().__init__()
        self.af, self.vf = AudioFront(d), VisFront(d)
        self.ae, self.ve = TempEnc(d, 1), TempEnc(d, 1)
        self.ap, self.vp = AttnPool(d), AttnPool(d)

    def seqs(self, b):
        return self.ae(self.af(b["audio"])), self.ve(self.vf(b["face"]))

    def pooled(self, b):
        A, V = self.seqs(b)
        return self.ap(A), self.vp(V)


class EarlyFusion(Base):
    def __init__(self, K, d, layers=2):
        super().__init__()
        self.af, self.vf = AudioFront(d), VisFront(d)
        self.mix = nn.Linear(2 * d, d)
        self.enc, self.pool, self.head = TempEnc(d, layers), AttnPool(d), mlp(d, K, d)

    def forward(self, b):
        x = self.mix(torch.cat([self.af(b["audio"]), self.vf(b["face"])], -1))
        return dict(logits=self.head(self.pool(self.enc(x))))


class LateFusion(TwoStream):
    def __init__(self, K, d):
        super().__init__(d)
        self.ha, self.hv = mlp(d, K, d), mlp(d, K, d)

    def forward(self, b):
        a, v = self.pooled(b)
        la, lv = self.ha(a), self.hv(v)
        out = dict(logits=(la + lv) / 2)
        if self.training:
            out["aux"] = 0.5 * (F.cross_entropy(la, b["y"]) + F.cross_entropy(lv, b["y"]))
        return out


class TFN(TwoStream):
    def __init__(self, K, d, r=64):
        super().__init__(d)
        self.ra, self.rv = nn.Linear(d, r), nn.Linear(d, r)
        self.head = mlp((r + 1) ** 2, K, d, p=0.3)

    def forward(self, b):
        a, v = self.pooled(b)
        one = a.new_ones(a.shape[0], 1)
        a, v = torch.cat([self.ra(a), one], 1), torch.cat([self.rv(v), one], 1)
        z = torch.bmm(a.unsqueeze(2), v.unsqueeze(1)).flatten(1)
        return dict(logits=self.head(z))


class LMF(TwoStream):
    def __init__(self, K, d, rank=4, out=128):
        super().__init__(d)
        self.fa = nn.Parameter(torch.randn(rank, d + 1, out) * 0.02)
        self.fv = nn.Parameter(torch.randn(rank, d + 1, out) * 0.02)
        self.wf = nn.Parameter(torch.ones(1, rank) / rank)
        self.bias = nn.Parameter(torch.zeros(out))
        self.head = mlp(out, K, d)

    def forward(self, b):
        a, v = self.pooled(b)
        one = a.new_ones(a.shape[0], 1)
        za = torch.einsum("bi,rio->rbo", torch.cat([a, one], 1), self.fa)
        zv = torch.einsum("bi,rio->rbo", torch.cat([v, one], 1), self.fv)
        z = torch.einsum("xr,rbo->bo", self.wf, za * zv) + self.bias
        return dict(logits=self.head(z))


class GMU(TwoStream):
    def __init__(self, K, d):
        super().__init__(d)
        self.ha, self.hv, self.z = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(2 * d, d)
        self.head = mlp(d, K, d)

    def forward(self, b):
        a, v = self.pooled(b)
        z = torch.sigmoid(self.z(torch.cat([a, v], 1)))
        return dict(logits=self.head(z * torch.tanh(self.ha(a)) + (1 - z) * torch.tanh(self.hv(v))))


class MulT(TwoStream):
    def __init__(self, K, d, layers=2):
        super().__init__(d)
        self.a2v = nn.ModuleList([CrossBlock(d) for _ in range(layers)])   # audio queries video
        self.v2a = nn.ModuleList([CrossBlock(d) for _ in range(layers)])
        self.sa, self.sv = TempEnc(d, 1), TempEnc(d, 1)
        self.head = mlp(2 * d, K, d)

    def forward(self, b):
        A, V = self.seqs(b)
        A2, V2 = A, V
        for ca, cv in zip(self.a2v, self.v2a):
            A2, V2 = A2 + ca(A2, V), V2 + cv(V2, A)
        A2, V2 = self.sa(A2), self.sv(V2)
        e = torch.cat([self.ap(A2), self.vp(V2)], 1)
        return dict(logits=self.head(e), emb=e)


class JCA(TwoStream):
    """Each modality attends to the joint (audio+video) sequence."""
    def __init__(self, K, d):
        super().__init__(d)
        self.ca, self.cv = CrossBlock(d), CrossBlock(d)
        self.head = mlp(2 * d, K, d)

    def forward(self, b):
        A, V = self.seqs(b)
        J = torch.cat([A, V], 1)
        A2, V2 = A + self.ca(A, J), V + self.cv(V, J)
        return dict(logits=self.head(torch.cat([self.ap(A2), self.vp(V2)], 1)))


class SAF(Base):
    """Joint self-attention over audio+video tokens, with modality dropout."""
    def __init__(self, K, d, p_drop=0.2, layers=2):
        super().__init__()
        self.af, self.vf = AudioFront(d), VisFront(d)
        self.emb = nn.Parameter(torch.zeros(2, 1, 1, d))
        self.enc = TempEnc(d, layers, length=2 * T_STEPS)
        self.pool, self.head, self.p = AttnPool(d), mlp(d, K, d), p_drop

    def forward(self, b):
        A, V = self.af(b["audio"]), self.vf(b["face"])
        if self.training:
            B = A.shape[0]
            r = torch.rand(B, device=A.device)
            A = A * (r >= self.p).float()[:, None, None]                     # drop audio
            V = V * ((r < self.p) | (r >= 2 * self.p)).float()[:, None, None]  # drop video
        x = torch.cat([A + self.emb[0], V + self.emb[1]], 1)
        return dict(logits=self.head(self.pool(self.enc(x))))


class MambaFusion(Base):
    def __init__(self, K, d, layers=2, d_state=16):
        super().__init__()
        self.af, self.vf = AudioFront(d), VisFront(d)
        self.emb = nn.Parameter(torch.zeros(2, 1, 1, d))
        self.fw = nn.ModuleList([MambaLite(d, d_state) for _ in range(layers)])
        self.bw = nn.ModuleList([MambaLite(d, d_state) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.head = mlp(d, K, d)

    def forward(self, b):
        A, V = self.af(b["audio"]) + self.emb[0], self.vf(b["face"]) + self.emb[1]
        x = torch.stack([A, V], 2).flatten(1, 2)                            # interleave a1 v1 a2 v2 ...
        for f, bw, n in zip(self.fw, self.bw, self.norms):
            h = n(x)
            x = x + f(h) + bw(h.flip(1)).flip(1)                            # bidirectional
        return dict(logits=self.head(x.mean(1)))


class TMC(TwoStream):
    def __init__(self, K, d):
        super().__init__(d)
        self.K = K
        self.ha, self.hv = mlp(d, K, d), mlp(d, K, d)

    def forward(self, b):
        a, v = self.pooled(b)
        aa, av = F.softplus(self.ha(a)) + 1, F.softplus(self.hv(v)) + 1
        ac = ds_combine(aa, av, self.K)
        out = dict(logits=torch.log(ac))
        if self.training:
            an = min(1.0, self.progress * 2)
            out["override"] = sum(edl_loss(x, b["y"], self.K, an) for x in (aa, av, ac))
        return out


class TrustAV(Base):
    """TRUST-AV: Trusted Region-aware Uncertainty-guided Speaker-invariant Transformer."""
    def __init__(self, K, d, n_spk, unc=True, regions=True, adv=True, quality=True,
                 moddrop=True, contrast=True, p_drop=0.15, lam_adv=0.1, lam_edl=0.5, lam_con=0.1):
        super().__init__()
        self.K, self.cfg = K, dict(unc=unc, regions=regions, adv=adv, quality=quality,
                                   moddrop=moddrop, contrast=contrast)
        self.p_drop, self.lam_adv, self.lam_edl, self.lam_con = p_drop, lam_adv, lam_edl, lam_con
        self.af, self.ae = AudioFront(d), TempEnc(d, 1)
        self.vf = nn.ModuleDict({r: VisFront(d) for r in (["face", "eyes", "mouth"] if regions else ["face"])})
        self.reg_score = nn.Linear(d, 1)
        self.ve = TempEnc(d, 1)
        self.rel = nn.Linear(2, 1)                                   # frame reliability from quality + mask
        nn.init.constant_(self.rel.weight, 2.0)
        nn.init.constant_(self.rel.bias, -1.0)
        self.ap0, self.vp0, self.ap, self.vp = AttnPool(d), AttnPool(d), AttnPool(d), AttnPool(d)
        self.ca_av, self.ca_va = CrossBlock(d), CrossBlock(d)
        if unc:
            self.ha, self.hv = nn.Linear(d, K), nn.Linear(d, K)
        self.head = mlp(3 * d + (3 if unc else 0), K, d)
        if adv:
            self.spk = mlp(2 * d, n_spk, d)
        if contrast:
            self.pa, self.pv = nn.Linear(d, 128), nn.Linear(d, 128)

    def forward(self, b):
        cfg, B = self.cfg, b["audio"].shape[0]
        audio, mask, qual = b["audio"], b["mask"], b["quality"]
        regs = {r: b[r] for r in self.vf}
        keep_a = keep_v = torch.ones(B, device=audio.device)
        if self.training and cfg["moddrop"]:                         # modality dropout
            r = torch.rand(B, device=audio.device)
            keep_a = (r >= self.p_drop).float()
            keep_v = ((r < self.p_drop) | (r >= 2 * self.p_drop)).float()
            audio = audio * keep_a.view(-1, *([1] * (audio.dim() - 1)))
            regs = {k: x * keep_v[:, None, None] for k, x in regs.items()}
            mask, qual = mask * keep_v[:, None], qual * keep_v[:, None]

        # ---- unimodal sequences ----
        A = self.ae(self.af(audio))
        toks = torch.stack([self.vf[r](regs[r]) for r in self.vf], 2)       # (B, T, R, d)
        rw = torch.softmax(self.reg_score(torch.tanh(toks)).squeeze(-1), -1)  # region weights
        V = self.ve((rw.unsqueeze(-1) * toks).sum(2))
        if cfg["quality"]:
            rel = torch.sigmoid(self.rel(torch.stack([qual, mask], -1))).squeeze(-1)
        else:
            rel = torch.ones_like(mask)
        logrel = torch.log(rel.clamp_min(1e-4))

        # ---- uncertainty-guided cross-modal attention ----
        out = {}
        if cfg["unc"]:
            a0, v0 = self.ap0(A), self.vp0(V, logrel)
            ea, ev = F.softplus(self.ha(a0)), F.softplus(self.hv(v0))
            aa, av = ea + 1, ev + 1
            Sa, Sv = aa.sum(1, keepdim=True), av.sum(1, keepdim=True)
            ua, uv = self.K / Sa, self.K / Sv                          # (B, 1)
            ba, bv = ea / Sa, ev / Sv
            conflict = ba.sum(1, keepdim=True) * bv.sum(1, keepdim=True) - (ba * bv).sum(1, keepdim=True)
            # relative-uncertainty routing: a stream reads more from the other modality when it is
            # the less certain of the two; equal uncertainty -> weight 1 (plain cross-attention)
            w_av = (2 * ua / (ua + uv)).detach()                       # audio <- video
            w_va = (2 * uv / (ua + uv)).detach()                       # video <- audio
            A2 = A + w_av[:, :, None] * self.ca_av(A, V, logrel)
            V2 = V + w_va[:, :, None] * self.ca_va(V, A)
            ap, vp = self.ap(A2), self.vp(V2, logrel)
            z = torch.cat([ap, vp, ap * vp, ua, uv, conflict], 1)
            out.update(u_audio=ua.squeeze(1), u_video=uv.squeeze(1), conflict=conflict.squeeze(1),
                       route_av=w_av.squeeze(1))
        else:
            A2, V2 = A + self.ca_av(A, V), V + self.ca_va(V, A)
            ap, vp = self.ap(A2), self.vp(V2)
            z = torch.cat([ap, vp, ap * vp], 1)
        out["logits"] = self.head(z)
        out["region_w"] = rw.mean(1)
        out["emb"] = torch.cat([ap, vp], 1)

        if self.training:
            aux = 0.0
            if cfg["unc"]:
                an = min(1.0, self.progress * 2)
                aux = aux + self.lam_edl * 0.5 * (edl_loss(aa, b["y"], self.K, an) +
                                                  edl_loss(av, b["y"], self.K, an))
            if cfg["contrast"]:
                both = (keep_a * keep_v) > 0
                if both.sum() > 2:
                    za = F.normalize(self.pa(ap[both]), dim=1)
                    zv = F.normalize(self.pv(vp[both]), dim=1)
                    sim = za @ zv.t() / 0.07
                    tgt = torch.arange(sim.shape[0], device=sim.device)
                    aux = aux + self.lam_con * 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.t(), tgt))
            if cfg["adv"]:
                lam = self.lam_adv * (2 / (1 + math.exp(-10 * self.progress)) - 1)
                h = GradReverse.apply(torch.cat([ap, vp], 1), lam)
                aux = aux + F.cross_entropy(self.spk(h), b["spk"])
            out["aux"] = aux
        return out


class TrustAV2(Base):
    """TRUST-AV v2: joint audio-video self-attention (strongest backbone) in which every
    attention logit is biased by (i) the modality's evidential uncertainty and (ii) per-frame
    face quality, so unreliable tokens are attended to less, token by token."""
    def __init__(self, K, d, n_spk, unc=True, regions=True, adv=True, quality=True,
                 moddrop=True, contrast=True, trio=False, distill=False, n_exp=1, consistency=False,
                 mamba_layers=0,
                 layers=2, p_drop=0.2, lam_adv=0.1, lam_edl=0.5, lam_con=0.1, lam_kd=0.5, kd_t=2.0,
                 tok_drop=0.15, decision="ds"):
        super().__init__()
        trio, distill = trio and unc, distill and unc                # both need the evidential heads
        self.K, self.cfg = K, dict(unc=unc, regions=regions, adv=adv, quality=quality,
                                   moddrop=moddrop, contrast=contrast, trio=trio, distill=distill)
        self.lam_kd, self.kd_t = lam_kd, kd_t
        self.p_drop, self.lam_adv, self.lam_edl, self.lam_con = p_drop, lam_adv, lam_edl, lam_con
        self.af = AudioFront(d)
        self.vf = nn.ModuleDict({r: VisFront(d) for r in (["face", "eyes", "mouth"] if regions else ["face"])})
        self.reg_score = nn.Linear(d, 1)
        self.pos = nn.Parameter(torch.zeros(1, T_STEPS, d))
        self.emb = nn.Parameter(torch.zeros(2, 1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.rel = nn.Linear(2, 1)
        nn.init.constant_(self.rel.weight, 2.0)
        nn.init.constant_(self.rel.bias, -1.0)
        if unc or contrast:                                          # light unimodal summaries
            self.ue_a, self.ue_v = TempEnc(d, 1), TempEnc(d, 1)
            self.up_a, self.up_v = AttnPool(d), AttnPool(d)
        if unc:
            self.ha, self.hv = nn.Linear(d, K), nn.Linear(d, K)
            self.beta = nn.Parameter(torch.tensor(1.0))              # strength of the uncertainty bias
        # optional state-space temporal stage (bidirectional selective SSM over interleaved a/v tokens)
        self.m_fw = nn.ModuleList([MambaLite(d) for _ in range(mamba_layers)])
        self.m_bw = nn.ModuleList([MambaLite(d) for _ in range(mamba_layers)])
        self.m_norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(mamba_layers)])
        self.blocks = nn.ModuleList([CrossBlock(d) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.n_exp, self.tok_drop, self.consistency = n_exp, tok_drop, consistency
        self.decision = decision           # "ds": Dempster-Shafer opinion only; "poe": product of the
                                           # discriminative fused head and the DS opinion
        self.pools = nn.ModuleList([AttnPool(d) for _ in range(n_exp)])          # trusted experts
        self.heads = nn.ModuleList([mlp(d + (3 if unc else 0), K, d) for _ in range(n_exp)])
        if adv:
            self.spk = mlp(d, n_spk, d)
        if contrast:
            self.pa, self.pv = nn.Linear(d, 128), nn.Linear(d, 128)

    def forward(self, b):
        cfg, B = self.cfg, b["audio"].shape[0]
        audio, mask, qual = b["audio"], b["mask"], b["quality"]
        regs = {r: b[r] for r in self.vf}
        keep_a = keep_v = torch.ones(B, device=audio.device)
        if self.training and cfg["moddrop"]:
            r = torch.rand(B, device=audio.device)
            keep_a = (r >= self.p_drop).float()
            keep_v = ((r < self.p_drop) | (r >= 2 * self.p_drop)).float()
            audio = audio * keep_a.view(-1, *([1] * (audio.dim() - 1)))
            regs = {k: x * keep_v[:, None, None] for k, x in regs.items()}
            mask, qual = mask * keep_v[:, None], qual * keep_v[:, None]

        A = self.af(audio) + self.pos
        toks = torch.stack([self.vf[r](regs[r]) for r in self.vf], 2)
        rw = torch.softmax(self.reg_score(torch.tanh(toks)).squeeze(-1), -1)
        V = (rw.unsqueeze(-1) * toks).sum(2) + self.pos
        if cfg["quality"]:
            rel = torch.sigmoid(self.rel(torch.stack([qual, mask], -1))).squeeze(-1)
        else:
            rel = torch.ones_like(mask)
        bias_a = torch.zeros(B, T_STEPS, device=A.device)
        bias_v = torch.log(rel.clamp_min(1e-4))

        out = {}
        if cfg["unc"] or cfg["contrast"]:
            a0 = self.up_a(self.ue_a(A))
            v0 = self.up_v(self.ue_v(V), bias_v)
        if cfg["unc"]:
            ea, ev = F.softplus(self.ha(a0)), F.softplus(self.hv(v0))
            aa, av = ea + 1, ev + 1
            Sa, Sv = aa.sum(1, keepdim=True), av.sum(1, keepdim=True)
            ua, uv = self.K / Sa, self.K / Sv
            ba, bv = ea / Sa, ev / Sv
            conflict = ba.sum(1, keepdim=True) * bv.sum(1, keepdim=True) - (ba * bv).sum(1, keepdim=True)
            # relative trust of each modality (1 = equal); detached so the evidential heads
            # are trained only by their own loss
            ta = (2 * uv / (ua + uv)).detach()
            tv = (2 * ua / (ua + uv)).detach()
            bias_a = bias_a + self.beta * torch.log(ta)
            bias_v = bias_v + self.beta * torch.log(tv)
            out.update(u_audio=ua.squeeze(1), u_video=uv.squeeze(1), conflict=conflict.squeeze(1),
                       route_av=tv.squeeze(1))

        if len(self.m_fw):
            # hybrid: state-space temporal modelling of the interleaved a1 v1 a2 v2 ... sequence,
            # then trusted (uncertainty-biased) joint attention on top
            Xi = torch.stack([A + self.emb[0], V + self.emb[1]], 2).flatten(1, 2)
            for f, bw, n in zip(self.m_fw, self.m_bw, self.m_norm):
                h = n(Xi)
                Xi = Xi + f(h) + bw(h.flip(1)).flip(1)
            X = torch.cat([Xi[:, 0::2], Xi[:, 1::2]], 1)
        else:
            X = torch.cat([A + self.emb[0], V + self.emb[1]], 1)
        bias = torch.cat([bias_a, bias_v], 1)                        # one bias per key token
        for blk in self.blocks:
            X = X + blk(X, X, bias)
        X = self.norm(X)
        expert_logits, zs = [], []
        for e in range(self.n_exp):
            be = bias
            if self.training and self.n_exp > 1:                     # each expert sees a different token subset
                drop = (torch.rand_like(bias) < self.tok_drop).float()
                be = bias - 1e4 * drop
            ze = self.pools[e](X, be)
            zs.append(ze)
            if cfg["unc"]:
                ze = torch.cat([ze, ua, uv, conflict], 1)
            expert_logits.append(self.heads[e](ze))
        out["emb"] = torch.stack(zs).mean(0)
        fused = torch.stack(expert_logits).mean(0)
        if cfg["trio"]:
            # trusted decision fusion: Dempster-Shafer combination of every expert opinion with the
            # audio and video opinions; softmax(log alpha) = expected class probabilities
            ac = None
            for f in expert_logits:
                af = F.softplus(f) + 1
                ac = af if ac is None else ds_combine(ac, af, self.K)
            ac = ds_combine(ds_combine(ac, aa, self.K), av, self.K)
            if self.decision == "poe":
                out["logits"] = F.log_softmax(fused, 1) + torch.log(ac / ac.sum(1, keepdim=True))
            else:
                out["logits"] = torch.log(ac)
            out["u_fused"] = (self.K / ac.sum(1))
        else:
            out["logits"] = fused
        out["region_w"] = rw.mean(1)

        if self.training:
            aux = 0.0
            if cfg["unc"]:
                an = min(1.0, self.progress * 2)
                aux = aux + self.lam_edl * 0.5 * (edl_loss(aa, b["y"], self.K, an) +
                                                  edl_loss(av, b["y"], self.K, an))
            if cfg["contrast"]:
                both = (keep_a * keep_v) > 0
                if both.sum() > 2:
                    za = F.normalize(self.pa(a0[both]), dim=1)
                    zv = F.normalize(self.pv(v0[both]), dim=1)
                    sim = za @ zv.t() / 0.07
                    tgt = torch.arange(sim.shape[0], device=sim.device)
                    aux = aux + self.lam_con * 0.5 * (F.cross_entropy(sim, tgt) + F.cross_entropy(sim.t(), tgt))
            if cfg["adv"]:
                lam = self.lam_adv * (2 / (1 + math.exp(-10 * self.progress)) - 1)
                aux = aux + F.cross_entropy(self.spk(GradReverse.apply(out["emb"], lam)), b["spk"])
            if cfg["trio"]:                                          # keep every expert head sharp
                aux = aux + 0.5 * sum(F.cross_entropy(f, b["y"]) for f in expert_logits) / len(expert_logits)
            if cfg["distill"]:
                # self-distillation: fused prediction teaches each unimodal branch
                t = self.kd_t
                teacher = F.softmax(fused.detach() / t, 1)
                kd = 0.0
                for alpha, keep in ((aa, keep_a), (av, keep_v)):
                    student = F.log_softmax(torch.log(alpha) / t, 1)
                    kl = F.kl_div(student, teacher, reduction="none").sum(1)
                    kd = kd + (kl * keep).sum() / keep.sum().clamp_min(1)
                aux = aux + self.lam_kd * (t * t) * 0.5 * kd
            out["aux"] = aux
        return out


class TrustAVE(Base):
    """TRUST-AV-E: trusted fusion of heterogeneous experts. The TRUST-AV core (uncertainty-biased
    joint attention + evidential audio/video heads + trio Dempster-Shafer fusion) and a
    bidirectional state-space (Mamba-style) expert each produce a Dirichlet opinion; the two
    opinions are combined with Dempster-Shafer (ds=True) so every clip relies on whichever expert
    carries more evidence. ds=False = ablation with plain probability averaging."""
    def __init__(self, K, d, n_spk, ds=True, p_drop=0.2):
        super().__init__()
        self.K, self.ds, self.p = K, ds, p_drop
        self.core = TrustAV2(K, d, n_spk, trio=True, distill=False)
        self.ssm = MambaFusion(K, d)

    def set_progress(self, p):
        self.progress = p
        self.core.set_progress(p)
        self.ssm.set_progress(p)

    def forward(self, b):
        out = self.core(b)
        bm = b
        if self.training:                                   # the SSM expert is also trained for reliability
            B = b["audio"].shape[0]
            r = torch.rand(B, device=b["audio"].device)
            ka = (r >= self.p).float()
            kv = ((r < self.p) | (r >= 2 * self.p)).float()
            bm = dict(b)
            bm["audio"] = b["audio"] * ka.view(-1, *([1] * (b["audio"].dim() - 1)))
            bm["face"] = b["face"] * kv[:, None, None]
        lm = self.ssm(bm)["logits"]
        lc = out["logits"]                                  # log of the core's combined Dirichlet
        if self.ds:
            am = F.softplus(lm.float()) + 1
            ac = torch.exp(lc.float()).clamp_min(1.0)
            a = ds_combine(ac, am, self.K)
            out["logits"] = torch.log(a)
            out["u_fused"] = self.K / a.sum(1)
            out["u_ssm"] = self.K / am.sum(1)
        else:
            out["logits"] = torch.log(0.5 * F.softmax(lc.float(), 1) + 0.5 * F.softmax(lm.float(), 1) + 1e-8)
        if self.training:                                   # keep both experts individually sharp
            out["aux"] = out.get("aux", 0.0) + 0.5 * (F.cross_entropy(lc, b["y"]) + F.cross_entropy(lm, b["y"]))
        return out


class Ensemble(Base):
    """B13: plain ensemble of the two strongest baselines (Mamba + SAF), probabilities averaged."""
    def __init__(self, K, d):
        super().__init__()
        self.m1, self.m2 = MambaFusion(K, d), SAF(K, d)

    def forward(self, b):
        l1, l2 = self.m1(b)["logits"], self.m2(b)["logits"]
        out = dict(logits=torch.log(0.5 * F.softmax(l1.float(), 1) + 0.5 * F.softmax(l2.float(), 1) + 1e-8))
        if self.training:
            out["aux"] = 0.5 * (F.cross_entropy(l1, b["y"]) + F.cross_entropy(l2, b["y"]))
        return out


TRUST_HP = ["layers", "p_drop", "lam_edl", "lam_adv", "lam_con", "decision", "mamba_layers"]


def build(name, K, n_spk, d, hp=None):
    hp = hp or {}
    if name == "b1_audio":  return Unimodal(K, d, "audio")
    if name == "b2_video":  return Unimodal(K, d, "video")
    if name == "b3_early":  return EarlyFusion(K, d, layers=hp.get("layers", 2))
    if name == "b4_late":   return LateFusion(K, d)
    if name == "b5_tfn":    return TFN(K, d)
    if name == "b6_lmf":    return LMF(K, d)
    if name == "b7_gmu":    return GMU(K, d)
    if name == "b8_mult":   return MulT(K, d)
    if name == "b9_jca":    return JCA(K, d)
    if name == "b10_saf":   return SAF(K, d, p_drop=hp.get("p_drop", 0.2), layers=hp.get("layers", 2))
    if name == "b11_mamba": return MambaFusion(K, d, layers=hp.get("layers", 2), d_state=hp.get("d_state", 16))
    if name == "b12_tmc":   return TMC(K, d)
    if name == "b13_ens":   return Ensemble(K, d)
    if name == "trustave":     return TrustAVE(K, d, n_spk, ds=True)
    if name == "trustave_avg": return TrustAVE(K, d, n_spk, ds=False)
    if name.startswith("trustav5"):     # hybrid: state-space temporal stage + trusted fusion (v3 core)
        off = name.replace("trustav5", "").strip("_").replace("no_", "")
        core = ["unc", "regions", "adv", "quality", "moddrop", "contrast", "trio"]
        flags = {k: k != off for k in core}
        return TrustAV2(K, d, n_spk, distill=False, mamba_layers=0 if off == "ssm" else 2, **flags)
    if name.startswith("trustav4"):     # v3 (no distill) + trusted experts + degradation consistency
        small = name.startswith("trustav4s")
        off = name.replace("trustav4s" if small else "trustav4", "").strip("_").replace("no_", "")
        core = ["unc", "regions", "adv", "quality", "moddrop", "contrast", "trio"]
        flags = {k: k != off for k in core}
        return TrustAV2(K, 160 if small else d, n_spk, distill=False, **flags,
                        n_exp=1 if off == "exp" else 3, consistency=off != "cons")
    if name.startswith("trustav3"):                       # v2 + trusted decision fusion + self-distillation
        offs = set(name.replace("trustav3", "").replace("no_", "").strip("_").split("_")) - {""}
        flags = {k: k not in offs for k in ["unc", "regions", "adv", "quality", "moddrop", "contrast",
                                            "trio", "distill"]}
        return TrustAV2(K, d, n_spk, **flags, **{k: hp[k] for k in TRUST_HP if k in hp})
    if name.startswith("trustav2"):
        off = name.replace("trustav2", "").strip("_").replace("no_", "")
        flags = {k: k != off for k in ["unc", "regions", "adv", "quality", "moddrop", "contrast"]}
        return TrustAV2(K, d, n_spk, **flags)
    if name.startswith("trustav"):
        off = name.replace("trustav", "").strip("_").replace("no_", "")
        flags = {k: k != off for k in ["unc", "regions", "adv", "quality", "moddrop", "contrast"]}
        return TrustAV(K, d, n_spk, **flags)
    raise ValueError(name)


# =============================================================================
# Training
# =============================================================================
def set_seed(s):
    random.seed(s), np.random.seed(s), torch.manual_seed(s), torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, arr, y, spk, ix, device, bs, mode=None, seed=0):
    model.eval()
    rng = np.random.RandomState(seed)
    preds, extras = [], {}
    for i in range(0, len(ix), bs):
        b = get_batch(arr, y, spk, ix[i:i + bs], device)
        if mode:
            b = corrupt(b, mode, rng)
        out = model(b)
        preds.append(out["logits"].argmax(1).cpu().numpy())
        extras.setdefault("prob", []).append(torch.softmax(out["logits"].float(), 1).cpu().numpy())
        for k in ["u_audio", "u_video", "u_fused", "u_ssm", "conflict", "route_av", "region_w", "emb"]:
            if k in out:
                extras.setdefault(k, []).append(out[k].float().cpu().numpy())
    return np.concatenate(preds), {k: np.concatenate(v) for k, v in extras.items()}


def run(args, key, name, seed, data):
    idx, arr, y, classes = data
    K = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    split = make_split(idx, y, args.protocol, seed)
    if args.smoke:
        rs = np.random.RandomState(0)
        split = {k: rs.permutation(v)[:300] for k, v in split.items()}
    spk, n_spk = speaker_ids(idx, split["train"])
    model = build(name, K, n_spk, args.dim, getattr(args, "hp", None))
    if args.ft_audio:                                  # swap the frozen-feature audio front
        if isinstance(model, Unimodal):
            if model.m != "audio":
                raise SystemExit("--ft_audio does not apply to b2_video")
            model.front = FTAudioFront(args.dim)
        elif hasattr(model, "af"):
            model.af = FTAudioFront(args.dim)
        else:
            raise SystemExit(f"--ft_audio not supported for {name}")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    counts = np.bincount(y[split["train"]], minlength=K).astype(float)
    cw = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32, device=device)
    cw = cw / cw.mean()                                    # inverse-sqrt class weights (all models)
    ft_ids = {id(p) for mod in model.modules() if isinstance(mod, FTAudioFront)
              for p in list(mod.top.parameters()) + list(mod.final_norm.parameters())}
    groups = [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in ft_ids],
               "lr": args.lr, "weight_decay": args.wd}]
    if ft_ids:
        groups.append({"params": [p for p in model.parameters() if id(p) in ft_ids],
                       "lr": args.lr_ft, "weight_decay": 0.01})
    opt = torch.optim.AdamW(groups)
    epochs = 2 if args.smoke else args.epochs
    steps_per_epoch = math.ceil(len(split["train"]) / args.bs)
    total = epochs * steps_per_epoch
    warm = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    select = args.select or ("wf1" if key == "meld" else "acc")

    ema_d = float(getattr(args, "ema", 0) or 0)       # exponential moving average of the weights
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if ema_d else None

    def swap_in_ema():
        cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v.to(cur[k].dtype) for k, v in ema.items()})
        return cur

    best, best_state, bad, step, t0 = -1, None, 0, 0, time.time()
    history = []
    for ep in range(epochs):
        model.train()
        ep_loss, n_b = 0.0, 0
        perm = np.random.permutation(split["train"])
        for i in range(0, len(perm), args.bs):
            ix = perm[i:i + args.bs]
            if len(ix) < 2:
                continue
            b = get_batch(arr, y, spk, ix, device)
            model.set_progress(step / max(total, 1))
            out = model(b)
            if getattr(model, "consistency", False):
                # degradation consistency: same prediction for a degraded copy of the batch
                mode = ["frame_drop50", "audio_noise50", "video_missing", "audio_missing"][np.random.randint(4)]
                out2 = model(corrupt(b, mode, np.random))
                p_clean = F.softmax(out["logits"].detach().float(), 1)
                kl = F.kl_div(F.log_softmax(out2["logits"].float(), 1), p_clean, reduction="batchmean")
                out["aux"] = out.get("aux", 0.0) + args.lam_cons * kl
            if "override" in out:
                loss = out["override"]
            else:
                loss = F.cross_entropy(out["logits"], b["y"], weight=cw, label_smoothing=0.1)
                if "aux" in out:
                    loss = loss + out["aux"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                with torch.no_grad():
                    dd = min(ema_d, (1 + step) / (10 + step))
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(dd).add_(v.detach().float(), alpha=1 - dd)
                        else:
                            ema[k].copy_(v)
            ep_loss, n_b = ep_loss + loss.item(), n_b + 1
        cur = swap_in_ema() if ema is not None else None
        pv, _ = predict(model, arr, y, spk, split["val"], device, args.bs * 2)
        vm = metrics(y[split["val"]], pv, K)
        score = vm[select]
        history.append(dict(epoch=ep + 1, train_loss=ep_loss / max(n_b, 1), val_acc=vm["acc"],
                            val_wf1=vm["wf1"], val_mf1=vm["mf1"], lr=sched.get_last_lr()[0]))
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if cur is not None:
            model.load_state_dict(cur)
        if args.verbose:
            print(f"    ep {ep + 1:3d}  loss {loss.item():.3f}  val {select} {score:.4f}  best {best:.4f}")
        if bad >= args.patience:
            break
    train_min = (time.time() - t0) / 60
    model.load_state_dict(best_state)

    te = split["test"]
    pt, ext = predict(model, arr, y, spk, te, device, args.bs * 2)
    res = metrics(y[te], pt, K)
    robust = {}
    modes = ["video_missing", "audio_missing"] + [f"frame_drop{l}" for l in CURVE_LEVELS[1:]] + \
            [f"audio_noise{l}" for l in CURVE_LEVELS[1:]]
    if args.smoke:
        modes = ["video_missing", "audio_missing", "frame_drop50"]
    for mode in modes:
        pr, _ = predict(model, arr, y, spk, te, device, args.bs * 2, mode=mode, seed=seed)
        m = metrics(y[te], pr, K)
        robust[mode] = {k: m[k] for k in ["acc", "wf1", "mf1", "uar"]}
    # inference speed (ms per clip, batch 1)
    b1 = get_batch(arr, y, spk, te[:1], device)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            model(b1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(20):
            model(b1)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.time() - t1) / 20 * 1000

    proto = args.protocol + ("_ft" if args.ft_audio else "") + ("_sn" if args.spk_norm else "") + \
        (f"_{args.tag}" if getattr(args, "tag", "") else "")
    record = dict(dataset=DATASETS[key], protocol=proto, model=name, seed=seed, ft_audio=bool(args.ft_audio),
                  spk_norm=bool(args.spk_norm), hp=getattr(args, "hp", None), lr=args.lr, wd=args.wd,
                  dim=args.dim, ema=ema_d,
                  classes=classes, n_train=int(len(split["train"])), n_val=int(len(split["val"])),
                  n_test=int(len(te)), params=int(n_params), epochs_run=ep + 1,
                  best_val=float(best), select=select, train_min=round(train_min, 2),
                  ms_per_clip=round(ms, 2), test=res, robust=robust, history=history,
                  smoke=bool(args.smoke))
    out_dir = RESULTS / DATASETS[key] / proto
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    (out_dir / f"{name}_seed{seed}{tag}.json").write_text(json.dumps(record, indent=1))
    if seed != args.seeds[0]:
        ext.pop("emb", None)                            # embeddings only for the first seed (size)
    np.savez_compressed(out_dir / f"{name}_seed{seed}{tag}_preds.npz", y=y[te], pred=pt,
                        clip_id=idx["clip_id"].values[te],
                        speaker=idx["speaker"].astype(str).values[te], **ext)
    if getattr(args, "save_ckpt", False):                  # weights of the selected epoch (for explain.py)
        torch.save(best_state, out_dir / f"{name}_seed{seed}{tag}.pt")
    if not args.smoke and not getattr(args, "tag", ""):
        row = dict(dataset=DATASETS[key], protocol=proto, model=name, seed=seed,
                   acc=res["acc"], wf1=res["wf1"], mf1=res["mf1"], uar=res["uar"],
                   rob_video_missing_acc=robust["video_missing"]["acc"],
                   rob_audio_missing_acc=robust["audio_missing"]["acc"],
                   rob_frame_drop50_acc=robust["frame_drop50"]["acc"],
                   params=n_params, epochs=ep + 1, train_min=round(train_min, 2), ms_per_clip=round(ms, 2))
        summ = RESULTS / "summary.csv"
        pd.DataFrame([row]).to_csv(summ, mode="a", header=not summ.exists(), index=False)
    print(f"  {name:22s} seed {seed}  acc {res['acc']:.4f}  wF1 {res['wf1']:.4f}  mF1 {res['mf1']:.4f}  "
          f"UAR {res['uar']:.4f}  | no-video {robust['video_missing']['acc']:.3f}  "
          f"no-audio {robust['audio_missing']['acc']:.3f}  | {n_params / 1e6:.2f}M params, "
          f"{ep + 1} ep, {train_min:.1f} min")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=["crema"], choices=list(DATASETS))
    ap.add_argument("--model", nargs="+", default=["b1_audio"],
                    help="model names, or: baselines | ablations | all")
    ap.add_argument("--protocol", default="official", choices=["official", "cv5", "random"],
                    help="cv5 = speaker-independent 5-fold CV (CREMA-D/RAVDESS); --seeds 1 2 3 4 5 = folds")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--select", default=None, choices=[None, "acc", "wf1", "mf1", "uar"])
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 300 clips: checks every model runs")
    ap.add_argument("--ft_audio", action="store_true",
                    help="fine-tune WavLM layers 20-23 on frame-level states (run extract_wavlm_mid.py first)")
    ap.add_argument("--lr_ft", type=float, default=3e-5, help="learning rate of the fine-tuned WavLM layers")
    ap.add_argument("--lam_cons", type=float, default=1.0, help="weight of the degradation-consistency loss")
    ap.add_argument("--spk_norm", action="store_true",
                    help="speaker-adaptive feature normalisation (unlabelled per-speaker statistics)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off), e.g. 0.999")
    args = ap.parse_args()

    if args.ft_audio and args.bs == 64:
        args.bs = 32                                   # fits the fine-tuned WavLM layers in 8 GB
    names = []
    for m in args.model:
        names += BASELINES if m == "baselines" else ABLATIONS if m == "ablations" else \
            ABLATIONS2 if m == "ablations2" else \
            ABLATIONS3 if m == "ablations3" else \
            BASELINES + ["trustav"] + ABLATIONS if m == "all" else [m]
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    for key in args.dataset:
        data = load_dataset(key, skip_audio=args.ft_audio)
        if args.spk_norm:
            speaker_normalise(data[0], data[1])
        if args.ft_audio:
            data[1]["mid"], data[1]["off"] = load_mid(key, len(data[0]))
        print(f"\n=== {DATASETS[key]} | {args.protocol} | classes: {data[3]} ===")
        for name in names:
            for s in args.seeds:
                run(args, key, name, s, data)
    print("\nDone. Results in", RESULTS)


if __name__ == "__main__":
    main()
