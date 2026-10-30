"""
extract_features.py - Step 3 of the pipeline: frozen foundation-model features.

For every usable clip it runs two frozen pretrained encoders on the GPU once
and saves compact features, so all later models (12 baselines + TRUST-AV +
ablations) train in minutes on an 8 GB GPU.

  AUDIO  : WavLM-Large (microsoft/wavlm-large)
           - audio trimmed to the detected speech segment (max 15 s)
           - hidden states of 13 layers (every 2nd of 25) kept, so models can
             learn their own layer weighting
           - pooled to 16 time steps                  -> audio  (16, 13, 1024)
  VISUAL : CLIP ViT-L/14 (openai/clip-vit-large-patch14), vision tower
           - 16 frames x 3 regions (face, eyes+brows, mouth)
                                                      -> face   (16, 1024)
                                                         eyes   (16, 1024)
                                                         mouth  (16, 1024)
  ALSO   : face_mask (16), quality (16), speech_ratio, n_audio_frames

Saved as processed/<DATASET>/features/<clip_id>.npz (float16), plus
processed/<DATASET>/features_index.csv joining labels, splits, speakers,
segment info and feature paths - the single table every training script reads.

Safe to stop and re-run: finished clips are skipped.

Usage (inside the venv, from the repository root):
    python code\\extract_features.py --limit 5            # quick test
    python code\\extract_features.py                      # all datasets
    python code\\extract_features.py --datasets crema     # one dataset
"""
import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
DATASETS = {"meld": "MELD", "crema": "CREMA-D", "ravdess": "RAVDESS"}

AUDIO_MODEL = "microsoft/wavlm-large"
VISION_MODEL = "openai/clip-vit-large-patch14"
N_STEPS = 16
AUDIO_LAYERS = list(range(0, 25, 2))          # 13 of 25 hidden states
MAX_AUDIO_S = 15.0
MIN_AUDIO_S = 0.5
SR = 16000
FACE_SIZE, REGION_SIZE = 224, 112
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


# ----------------------------------------------------------------------------
# Data loading (CPU workers)
# ----------------------------------------------------------------------------
def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != SR:                                   # should not happen (prepare_data -> 16 kHz)
        x = cv2.resize(x[None, :], (int(len(x) * SR / sr), 1))[0]
    return x


def split_grid(path, size):
    g = cv2.imread(str(path))
    if g is None:
        return np.zeros((N_STEPS, 224, 224, 3), np.uint8)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
    tiles = []
    for i in range(N_STEPS):
        r, c = divmod(i, 4)
        t = g[r * size:(r + 1) * size, c * size:(c + 1) * size]
        if size != 224:
            t = cv2.resize(t, (224, 224), interpolation=cv2.INTER_CUBIC)
        tiles.append(t)
    return np.stack(tiles)


class ClipSet(Dataset):
    def __init__(self, ds_name, rows):
        self.ds, self.rows = ds_name, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        cid = r["clip_id"]
        base = PROC / self.ds
        # ---- audio, trimmed to the speech segment ----
        wav = base / "audio" / f"{cid}.wav"
        x = read_wav(wav) if wav.exists() else np.zeros(int(SR * MIN_AUDIO_S), np.float32)
        s0, s1 = r.get("speech_start"), r.get("speech_end")
        if pd.notna(s0) and pd.notna(s1) and s1 - s0 >= 0.2:
            x = x[int(s0 * SR):int(s1 * SR)]
        if len(x) > MAX_AUDIO_S * SR:              # very long (mis-cut) clips: keep the centre
            mid, half = len(x) // 2, int(MAX_AUDIO_S * SR / 2)
            x = x[mid - half:mid + half]
        if len(x) < MIN_AUDIO_S * SR:
            x = np.pad(x, (0, int(MIN_AUDIO_S * SR) - len(x)))
        x = (x - x.mean()) / (x.std() + 1e-7)      # WavLM-Large expects normalised input
        # ---- face regions ----
        fdir = base / "faces"
        imgs = np.concatenate([
            split_grid(fdir / f"{cid}_face.jpg", FACE_SIZE),
            split_grid(fdir / f"{cid}_eyes.jpg", REGION_SIZE),
            split_grid(fdir / f"{cid}_mouth.jpg", REGION_SIZE),
        ])                                          # (48, 224, 224, 3)
        imgs = (imgs.astype(np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
        imgs = imgs.transpose(0, 3, 1, 2)
        meta_p = fdir / f"{cid}.json"
        mask, qual = np.zeros(N_STEPS, np.float32), np.zeros(N_STEPS, np.float32)
        if meta_p.exists():
            m = json.loads(meta_p.read_text())
            mask[:] = m.get("face_mask", [0] * N_STEPS)[:N_STEPS]
            qual[:] = m.get("quality", [0] * N_STEPS)[:N_STEPS]
        return cid, torch.from_numpy(x), torch.from_numpy(imgs), mask, qual


def collate(batch):
    return batch                                    # variable-length audio: keep as list


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------
def load_models(device):
    from transformers import CLIPVisionModel, WavLMModel
    print(f"Loading {AUDIO_MODEL} ...")
    wavlm = WavLMModel.from_pretrained(AUDIO_MODEL).to(device).eval().half()
    print(f"Loading {VISION_MODEL} (vision tower) ...")
    clip = CLIPVisionModel.from_pretrained(VISION_MODEL).to(device).eval().half()
    return wavlm, clip


@torch.no_grad()
def encode_audio(wavlm, x, device):
    out = wavlm(x[None].to(device).half(), output_hidden_states=True)
    hs = torch.stack([out.hidden_states[i][0] for i in AUDIO_LAYERS])   # (13, T, 1024)
    n_frames = hs.shape[1]
    pooled = F.adaptive_avg_pool1d(hs.permute(0, 2, 1).float(), N_STEPS)  # (13, 1024, 16)
    return pooled.permute(2, 0, 1).cpu().numpy().astype(np.float16), n_frames


@torch.no_grad()
def encode_images(clip, imgs, device, chunk=96):
    feats = []
    for i in range(0, len(imgs), chunk):
        x = imgs[i:i + chunk].to(device, non_blocking=True).half()
        feats.append(clip(pixel_values=x).pooler_output.float().cpu())
    return torch.cat(feats).numpy().astype(np.float16)                  # (N, 1024)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def build_index(name):
    man = pd.read_csv(PROC / name / "manifest.csv")
    seg = pd.read_csv(PROC / name / "segments.csv")
    keep = ["clip_id", "status", "face_ratio", "n_tracks", "n_shots", "max_faces",
            "mean_quality", "speech_start", "speech_end", "speech_ratio", "audio_dur"]
    seg = seg[[c for c in keep if c in seg.columns]].rename(columns={"status": "face_status"})
    man = man[(man["status"].isin(["ok", "few_frames"])) & (man["n_frames"] > 0)]
    df = man.merge(seg, on="clip_id", how="left")
    return df


def run_dataset(key, limit, workers, wavlm, clip, device):
    name = DATASETS[key]
    df = build_index(name)
    if limit:
        df = df.head(limit)
    out_dir = PROC / name / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in df.to_dict("records") if not (out_dir / f"{r['clip_id']}.npz").exists()]
    print(f"\n=== {name}: {len(df)} clips, {len(todo)} to extract ===")

    if todo:
        loader = DataLoader(ClipSet(name, todo), batch_size=4, shuffle=False,
                            num_workers=workers, collate_fn=collate,
                            persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
        t0 = time.time()
        for batch in tqdm(loader, desc=name, unit="batch"):
            cids = [b[0] for b in batch]
            imgs = torch.cat([b[2] for b in batch])                     # (4*48, 3, 224, 224)
            vfeat = encode_images(clip, imgs, device).reshape(len(batch), 3, N_STEPS, -1)
            for j, (cid, x, _, mask, qual) in enumerate(batch):
                afeat, n_af = encode_audio(wavlm, x, device)
                np.savez(out_dir / f"{cid}.npz",
                         audio=afeat, face=vfeat[j, 0], eyes=vfeat[j, 1], mouth=vfeat[j, 2],
                         face_mask=mask.astype(np.float16), quality=qual.astype(np.float16),
                         n_audio_frames=np.int32(n_af))
        print(f"{name}: {len(todo)} clips in {(time.time() - t0) / 60:.1f} min")

    df["feature_path"] = [str(out_dir / f"{c}.npz") for c in df["clip_id"]]
    df["has_features"] = [os.path.exists(p) for p in df["feature_path"]]
    df.to_csv(PROC / name / "features_index.csv", index=False)
    print(f"{name}: {df['has_features'].sum()} / {len(df)} clips have features "
          f"-> {PROC / name / 'features_index.csv'}")

    # quick sanity check on one file
    ok = df[df["has_features"]]
    if len(ok):
        z = np.load(ok.iloc[0]["feature_path"])
        print("  shapes:", {k: z[k].shape for k in ["audio", "face", "eyes", "mouth", "face_mask"]})
        for k in ["audio", "face"]:
            a = z[k].astype(np.float32)
            if not np.isfinite(a).all():
                print(f"  WARNING: non-finite values in {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["crema", "ravdess", "meld"],
                    choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="CPU data-loading workers")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU not found. Activate the venv (python with torch+cu128).")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    wavlm, clip = load_models(device)
    for k in args.datasets:
        run_dataset(k, args.limit, args.workers, wavlm, clip, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
