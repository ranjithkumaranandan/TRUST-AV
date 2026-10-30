"""
prepare_data.py - Step 1 of the audio-visual emotion recognition pipeline.

For every labelled clip in MELD, CREMA-D and RAVDESS this script:
  1. extracts the audio track as 16 kHz mono WAV
  2. samples N frames uniformly across the clip (JPEG, short side = FRAME_SIZE)
  3. writes one manifest CSV per dataset with labels, splits, speakers and paths

Splits
  MELD     : official train / dev / test
  CREMA-D  : speaker-independent, by actor id (seeded, 70 / 10 / 20)
  RAVDESS  : speaker-independent, actors 1-16 train, 17-20 val, 21-24 test
             (each split is gender-balanced). Audio-visual files only (modality 01).

It can be stopped and re-run safely: clips that are already done are skipped.

Usage (inside the venv, from the repository root):
  python code\\prepare_data.py --limit 20        # quick test, 20 clips per dataset
  python code\\prepare_data.py                   # full run
  python code\\prepare_data.py --datasets meld   # one dataset only
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent          # repository root
DATA = ROOT / "datasets"
OUT = ROOT / "processed"

N_FRAMES = 16
FRAME_SIZE = 224
SAMPLE_RATE = 16000

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# ----------------------------------------------------------------------------
# Manifests (which clips exist, their labels and splits)
# ----------------------------------------------------------------------------
def find_one(base: Path, pattern: str) -> Path:
    hits = sorted(base.rglob(pattern))
    hits = [h for h in hits if not h.name.startswith("._")]
    if not hits:
        sys.exit(f"Could not find '{pattern}' under {base}")
    return hits[0]


def meld_manifest():
    base = DATA / "MELD"
    splits = {
        "train": ("train_sent_emo.csv", "train_splits"),
        "val": ("dev_sent_emo.csv", "dev_splits_complete"),
        "test": ("test_sent_emo.csv", "output_repeated_splits_test"),
    }
    rows = []
    for split, (csv_name, vid_dir_name) in splits.items():
        csv_path = find_one(base, csv_name)
        vid_dir = find_one(base, vid_dir_name)
        df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")
        for r in df.itertuples(index=False):
            name = f"dia{r.Dialogue_ID}_utt{r.Utterance_ID}"
            rows.append(dict(
                dataset="MELD", split=split, clip_id=f"{split}_{name}",
                video=str(vid_dir / f"{name}.mp4"), source_audio="",
                label=str(r.Emotion).lower(), sentiment=str(r.Sentiment).lower(),
                speaker=str(r.Speaker), dialogue_id=int(r.Dialogue_ID),
                utterance_id=int(r.Utterance_ID), text=str(r.Utterance),
            ))
    return rows


CREMA_EMO = {"ANG": "anger", "DIS": "disgust", "FEA": "fear",
             "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def crema_manifest(seed=42):
    base = DATA / "CREMA-D"
    vids = sorted((base / "VideoFlash").glob("*.flv"))
    actors = sorted({v.stem.split("_")[0] for v in vids})
    rng = random.Random(seed)
    rng.shuffle(actors)
    n = len(actors)
    n_train, n_val = round(0.7 * n), round(0.1 * n)
    split_of = {a: "train" for a in actors[:n_train]}
    split_of.update({a: "val" for a in actors[n_train:n_train + n_val]})
    split_of.update({a: "test" for a in actors[n_train + n_val:]})
    rows = []
    for v in vids:
        actor, sentence, emo, level = v.stem.split("_")
        wav = base / "AudioWAV" / f"{v.stem}.wav"
        rows.append(dict(
            dataset="CREMA-D", split=split_of[actor], clip_id=v.stem,
            video=str(v), source_audio=str(wav) if wav.exists() else "",
            label=CREMA_EMO[emo], sentiment="", speaker=actor,
            dialogue_id="", utterance_id="", text=sentence, intensity=level,
        ))
    return rows


RAVDESS_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
               "05": "anger", "06": "fear", "07": "disgust", "08": "surprise"}


def ravdess_split(actor: int) -> str:
    if actor <= 16:
        return "train"
    if actor <= 20:
        return "val"
    return "test"


def ravdess_manifest():
    base = DATA / "RAVDESS"
    rows = []
    for v in sorted(base.rglob("*.mp4")):
        parts = v.stem.split("-")
        if len(parts) != 7 or parts[0] != "01":   # 01 = full audio-video
            continue
        _, channel, emo, intensity, statement, rep, actor = parts
        actor_i = int(actor)
        rows.append(dict(
            dataset="RAVDESS", split=ravdess_split(actor_i), clip_id=v.stem,
            video=str(v), source_audio="", label=RAVDESS_EMO[emo], sentiment="",
            speaker=f"actor{actor}", dialogue_id="", utterance_id="",
            text=statement, intensity=intensity,
        ))
    return rows


BUILDERS = {"meld": meld_manifest, "crema": crema_manifest, "ravdess": ravdess_manifest}
OUT_NAMES = {"meld": "MELD", "crema": "CREMA-D", "ravdess": "RAVDESS"}


# ----------------------------------------------------------------------------
# Extraction
# ----------------------------------------------------------------------------
def probe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return p.returncode == 0, p.stderr[-400:]


def process_clip(row: dict, out_dir: Path) -> dict:
    res = dict(clip_id=row["clip_id"], audio_path="", frames_dir="",
               n_frames=0, duration=0.0, status="ok", error="")
    video = row["video"]
    if not os.path.exists(video):
        res.update(status="missing_video")
        return res

    wav = out_dir / "audio" / f"{row['clip_id']}.wav"
    fdir = out_dir / "frames" / row["clip_id"]
    res["duration"] = round(probe_duration(video), 3)

    # ---- audio --------------------------------------------------------------
    if not (wav.exists() and wav.stat().st_size > 1000):
        src = row.get("source_audio") or video
        ok, err = run([FFMPEG, "-y", "-v", "error", "-i", src, "-vn",
                       "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16",
                       str(wav)])
        if not ok:
            res.update(status="audio_failed", error=err)
    if wav.exists():
        res["audio_path"] = str(wav)

    # ---- frames -------------------------------------------------------------
    existing = sorted(fdir.glob("*.jpg")) if fdir.exists() else []
    if len(existing) < N_FRAMES:
        fdir.mkdir(parents=True, exist_ok=True)
        for f in existing:
            f.unlink()
        dur = res["duration"] if res["duration"] > 0 else 3.0
        fps = max(N_FRAMES / dur, 0.5)
        ok, err = run([FFMPEG, "-y", "-v", "error", "-i", video, "-an",
                       "-vf", f"fps={fps:.5f},scale=-2:{FRAME_SIZE}",
                       "-frames:v", str(N_FRAMES), "-q:v", "3",
                       str(fdir / "%02d.jpg")])
        existing = sorted(fdir.glob("*.jpg"))
        if not ok and not existing:
            res.update(status="frames_failed", error=err)
    res["n_frames"] = len(existing)
    if existing:
        res["frames_dir"] = str(fdir)
    if res["status"] == "ok" and len(existing) < N_FRAMES:
        res["status"] = "few_frames"
    return res


def process_dataset(key: str, limit: int, workers: int):
    name = OUT_NAMES[key]
    print(f"\n=== {name} ===")
    rows = BUILDERS[key]()
    if limit:
        rows = rows[:limit]
    out_dir = OUT / name
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_clip, r, out_dir): r["clip_id"] for r in rows}
        for f in tqdm(as_completed(futs), total=len(futs), desc=name, unit="clip"):
            r = f.result()
            results[r["clip_id"]] = r

    merged = []
    for r in rows:
        m = dict(r)
        m.update(results[r["clip_id"]])
        merged.append(m)
    df = pd.DataFrame(merged)
    df.to_csv(out_dir / "manifest.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    print(f"Clips: {len(df)}")
    print(df["status"].value_counts().to_string())
    print("\nLabels per split:")
    print(pd.crosstab(df["label"], df["split"]).to_string())
    bad = df[df["status"] != "ok"]
    if len(bad):
        bad.to_csv(out_dir / "problems.csv", index=False)
        print(f"\n{len(bad)} clips need attention -> {out_dir / 'problems.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["meld", "crema", "ravdess"],
                    choices=list(BUILDERS))
    ap.add_argument("--limit", type=int, default=0, help="clips per dataset (testing)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    if not FFMPEG or not FFPROBE:
        sys.exit("ffmpeg/ffprobe not found. Open a NEW PowerShell window "
                 "(after installing FFmpeg) and activate the venv again.")
    print(f"Data:   {DATA}\nOutput: {OUT}\nffmpeg: {FFMPEG}")
    for key in args.datasets:
        process_dataset(key, args.limit, args.workers)
    print("\nDone.")


if __name__ == "__main__":
    main()
