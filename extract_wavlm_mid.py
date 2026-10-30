"""
extract_wavlm_mid.py - Stage 4 preparation: frame-level WavLM states for partial fine-tuning.

Runs WavLM-Large up to (not including) transformer layer CUT on every clip and stores the
frame-level hidden states (50 frames / s, 1024-d, float16). train.py --ft_audio then runs and
fine-tunes WavLM layers CUT..23 (+ final layer norm) jointly with each model.

Audio is trimmed to the detected speech segment (as in extract_features.py) and centre-cropped
to MAX_SEC seconds.

Output per dataset (row order = features_index.csv):
    processed/<DATASET>/wavlm_mid/feats.bin     float16, shape (total_frames, 1024)
    processed/<DATASET>/wavlm_mid/offsets.npy   int64, start row of clip i (len N+1)
    processed/<DATASET>/wavlm_mid/meta.json

Usage (from the repository root):
    .\\venv\\Scripts\\python.exe code\\extract_wavlm_mid.py
    .\\venv\\Scripts\\python.exe code\\extract_wavlm_mid.py --datasets crema
"""
import argparse
import json
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "processed"
DATASETS = {"meld": "MELD", "crema": "CREMA-D", "ravdess": "RAVDESS"}
MODEL = "microsoft/wavlm-large"
CUT = 20
MAX_SEC, MIN_SEC, SR = 8.0, 0.5, 16000


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        n, ch = w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    return x.reshape(-1, ch).mean(1) if ch > 1 else x


def load_audio(name, r):
    wav = PROC / name / "audio" / f"{r['clip_id']}.wav"
    x = read_wav(wav) if wav.exists() else np.zeros(int(SR * MIN_SEC), np.float32)
    s0, s1 = r.get("speech_start"), r.get("speech_end")
    if pd.notna(s0) and pd.notna(s1) and s1 - s0 >= 0.2:
        x = x[int(s0 * SR):int(s1 * SR)]
    if len(x) > MAX_SEC * SR:
        mid, half = len(x) // 2, int(MAX_SEC * SR / 2)
        x = x[mid - half:mid + half]
    if len(x) < MIN_SEC * SR:
        x = np.pad(x, (0, int(MIN_SEC * SR) - len(x)))
    return (x - x.mean()) / (x.std() + 1e-7)


@torch.no_grad()
def lower_layers(model, x, device):
    """Exactly the WavLMModel / WavLMEncoderStableLayerNorm forward, stopped before layer CUT."""
    v = torch.from_numpy(x)[None].to(device).half()
    feats = model.feature_extractor(v).transpose(1, 2)
    hs, _ = model.feature_projection(feats)
    enc = model.encoder
    hs = hs + enc.pos_conv_embed(hs)
    pb = None
    for i in range(CUT):
        hs, pb = enc.layers[i](hs, position_bias=pb)
    return hs[0].float().cpu().numpy().astype(np.float16)


def run(key, model, device):
    name = DATASETS[key]
    idx = pd.read_csv(PROC / name / "features_index.csv")
    idx = idx[idx["has_features"]].reset_index(drop=True)
    out = PROC / name / "wavlm_mid"
    out.mkdir(exist_ok=True)
    if (out / "meta.json").exists():
        meta = json.loads((out / "meta.json").read_text())
        if meta.get("clips") == len(idx) and meta.get("cut") == CUT:
            print(f"{name}: already extracted, skipping")
            return
    offsets = [0]
    t0 = time.time()
    with open(out / "feats.bin", "wb") as f:
        for r in tqdm(idx.to_dict("records"), desc=name, unit="clip"):
            h = lower_layers(model, load_audio(name, r), device)
            f.write(h.tobytes())
            offsets.append(offsets[-1] + h.shape[0])
    np.save(out / "offsets.npy", np.array(offsets, np.int64))
    (out / "meta.json").write_text(json.dumps(dict(
        model=MODEL, cut=CUT, dim=1024, dtype="float16", clips=len(idx), frames=offsets[-1],
        max_sec=MAX_SEC, clip_ids_head=idx["clip_id"].head(3).tolist())))
    print(f"{name}: {len(idx)} clips, {offsets[-1]} frames "
          f"({offsets[-1] * 2048 / 1e9:.2f} GB) in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["crema", "ravdess", "meld"], choices=list(DATASETS))
    args = ap.parse_args()
    from transformers import WavLMModel
    device = torch.device("cuda")
    model = WavLMModel.from_pretrained(MODEL).to(device).eval().half()
    print(f"GPU: {torch.cuda.get_device_name(0)} | encoder: {type(model.encoder).__name__} | "
          f"layers: {len(model.encoder.layers)} | cut at {CUT}")
    for k in args.datasets:
        run(k, model, device)
    print("Done.")


if __name__ == "__main__":
    main()
