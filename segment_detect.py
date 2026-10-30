"""
segment_detect.py - Step 2 of the pipeline: SEGMENT and DETECT.

For every usable clip (from processed/<DATASET>/manifest.csv) this script:

  DETECT
    1. detects faces in the 16 sampled frames (OpenCV YuNet detector)
    2. links detections across frames into face tracks (IoU tracking)
    3. splits the clip into camera shots and, in each shot, picks the target track
         - MELD     : active-speaker score = mouth motion that follows the audio
                      energy + face size + detection confidence + frontal pose
         - CREMA-D / RAVDESS : single actor -> largest, most confident track
    4. scores each frame's quality: detection confidence, face size,
       sharpness, frontal pose

  SEGMENT
    5. facial regions from the 5 YuNet landmarks:
         face (full aligned crop), eyes+brows, mouth
    6. speech segment: start / end of voiced speech in the audio
       (energy-based voice activity detection), used later to trim
       silence and the few very long, mis-cut MELD clips

Outputs per dataset in processed/<DATASET>/faces/
    <clip_id>_face.jpg    4x4 grid of 16 face crops   (16 x 224x224)
    <clip_id>_eyes.jpg    4x4 grid of eye/brow crops  (16 x 112x112)
    <clip_id>_mouth.jpg   4x4 grid of mouth crops     (16 x 112x112)
    <clip_id>.json        per-frame boxes, quality scores, chosen track, speech segment
and processed/<DATASET>/segments.csv (one row per clip, summary).

Grids instead of single images keep the file count low (Windows handles
~70k files far better than ~1M).

Safe to stop and re-run: finished clips are skipped.

Usage (inside the venv, from the repository root):
    python code\\segment_detect.py --limit 20          # quick test
    python code\\segment_detect.py                     # full run, all datasets
    python code\\segment_detect.py --datasets meld     # one dataset
    python code\\segment_detect.py --limit 20 --redo   # re-do clips already processed
"""
import argparse
import json
import os
import sys
import urllib.request
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent           # repository root
PROC = ROOT / "processed"
MODEL_DIR = ROOT / "models"
YUNET = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")

N_FRAMES = 16
FACE_SIZE = 224
REGION_SIZE = 112
DET_SCORE = 0.6           # minimum detection confidence
DET_WIDTH = 640          # detection runs at this width; crops come from full resolution
MIN_FACE_PX = 24          # ignore detections smaller than this (full-res pixels)
DATASETS = {"meld": "MELD", "crema": "CREMA-D", "ravdess": "RAVDESS"}

_det = None               # one detector per worker process


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def ensure_model():
    if YUNET.exists() and YUNET.stat().st_size > 100_000:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face detector to {YUNET} ...")
    urllib.request.urlretrieve(YUNET_URL, YUNET)
    if YUNET.stat().st_size < 100_000:
        sys.exit("Face detector download failed. Download it manually from\n"
                 f"{YUNET_URL}\nand save it as {YUNET}")


def init_worker(model_path):
    global _det
    cv2.setNumThreads(1)
    _det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320),
                                     score_threshold=DET_SCORE,
                                     nms_threshold=0.3, top_k=20)


def detect(img):
    """Return array (n, 15): x, y, w, h, 5 landmarks (x,y), score - in original pixels."""
    h, w = img.shape[:2]
    scale = DET_WIDTH / w
    small = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    _det.setInputSize((small.shape[1], small.shape[0]))
    _, faces = _det.detect(small)
    if faces is None:
        return np.zeros((0, 15), np.float32)
    faces = faces.astype(np.float32)
    faces[:, :14] /= scale
    keep = (faces[:, 2] >= MIN_FACE_PX) & (faces[:, 3] >= MIN_FACE_PX)
    return faces[keep]


def resolve_video(p):
    """Manifest paths are absolute; fall back to ROOT/datasets/... if the drive moved."""
    if isinstance(p, str) and os.path.exists(p):
        return p
    s = str(p).replace("\\", "/")
    i = s.find("datasets/")
    if i >= 0:
        cand = ROOT / s[i:]
        if cand.exists():
            return str(cand)
    return None


def read_video_frames(path, n=N_FRAMES):
    """Read n frames spread evenly through the clip, at full resolution."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return []
        idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
        return [frames[i] for i in idx]
    targets = ((np.arange(n) + 0.5) * total / n).astype(int).clip(0, total - 1)
    out = []
    if total > 900:                               # long clip: seek instead of decoding all
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ok, f = cap.read()
            if ok:
                out.append(f)
    else:
        want, i = set(targets.tolist()), 0
        while i <= targets[-1]:
            ok = cap.grab()
            if not ok:
                break
            if i in want:
                ok, f = cap.retrieve()
                if ok:
                    for _ in range(int((targets == i).sum())):
                        out.append(f)
            i += 1
    cap.release()
    return out


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def same_face(a, b):
    """Link two detections if they overlap or are close relative to face size."""
    if iou(a[:4], b[:4]) > 0.3:
        return True
    ca = np.array([a[0] + a[2] / 2, a[1] + a[3] / 2])
    cb = np.array([b[0] + b[2] / 2, b[1] + b[3] / 2])
    w = (a[2] + b[2]) / 2
    ratio = (b[2] * b[3]) / max(a[2] * a[3], 1e-6)
    return np.linalg.norm(ca - cb) < 0.8 * w and 0.5 < ratio < 2.0


def build_tracks(dets, frames_idx):
    """Greedy linking of per-frame detections into tracks (within one shot)."""
    tracks = []                                   # each: {frame_idx: det}
    for t in frames_idx:
        frame_dets = dets[t]
        used = set()
        for tr in sorted(tracks, key=len, reverse=True):
            last_t = max(tr)
            if t - last_t > 3:                    # allow short gaps
                continue
            best_j, best_d = None, 1e9
            for j, d in enumerate(frame_dets):
                if j in used or not same_face(tr[last_t], d):
                    continue
                c1 = tr[last_t][:2] + tr[last_t][2:4] / 2
                dist = float(np.linalg.norm(c1 - (d[:2] + d[2:4] / 2)))
                if dist < best_d:
                    best_j, best_d = j, dist
            if best_j is not None:
                tr[t] = frame_dets[best_j]
                used.add(best_j)
        for j, d in enumerate(frame_dets):
            if j not in used:
                tracks.append({t: d})
    return tracks


def shot_boundaries(frames, thr=0.55):
    """Split the 16 sampled frames into camera shots (colour-histogram change)."""
    hists = []
    for f in frames:
        hsv = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hists.append(cv2.normalize(h, h).flatten())
    shots, cur = [], [0]
    for t in range(1, len(frames)):
        if cv2.compareHist(hists[t - 1], hists[t], cv2.HISTCMP_CORREL) < thr:
            shots.append(cur)
            cur = []
        cur.append(t)
    shots.append(cur)
    return shots


def square_crop(img, cx, cy, side, out):
    """Crop a square around (cx, cy), padding with black outside the image."""
    half = side / 2.0
    x1, y1 = int(round(cx - half)), int(round(cy - half))
    x2, y2 = x1 + int(round(side)), y1 + int(round(side))
    h, w = img.shape[:2]
    pad = max(0, -x1, -y1, x2 - w, y2 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out, out, 3), np.uint8)
    return cv2.resize(crop, (out, out), interpolation=cv2.INTER_CUBIC)


def aligned_face(img, d):
    """Rotate so the eyes are level, then crop the face square."""
    rex, rey, lex, ley = d[4], d[5], d[6], d[7]
    angle = np.degrees(np.arctan2(ley - rey, lex - rex))
    if abs(angle) > 25:                           # unreliable landmarks: do not rotate
        angle = 0.0
    cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(angle), 1.0)
    rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         borderMode=cv2.BORDER_CONSTANT)

    def tr(x, y):
        return M @ np.array([x, y, 1.0])

    re, le = tr(rex, rey), tr(lex, ley)
    rm, lm = tr(d[10], d[11]), tr(d[12], d[13])
    side = max(d[2], d[3]) * 1.25
    face = square_crop(rot, cx, cy, side, FACE_SIZE)

    eye_c = (re + le) / 2
    eye_dist = max(np.linalg.norm(le - re), 4.0)
    eyes = square_crop(rot, eye_c[0], eye_c[1] - 0.15 * eye_dist,  # include brows
                       eye_dist * 2.0, REGION_SIZE)
    mouth_c = (rm + lm) / 2
    mouth_w = max(np.linalg.norm(lm - rm), 4.0)
    mouth = square_crop(rot, mouth_c[0], mouth_c[1] + 0.1 * mouth_w,
                        mouth_w * 1.9, REGION_SIZE)
    return face, eyes, mouth


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frontalness(d):
    """1 = frontal, 0 = strong profile, from nose position between the eyes."""
    rex, lex, nx = d[4], d[6], d[8]
    span = lex - rex
    if abs(span) < 1e-3:
        return 0.0
    r = (nx - rex) / span                        # 0.5 for a frontal face
    return float(max(0.0, 1.0 - 2.0 * abs(r - 0.5)))


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def speech_segment(x, sr, win=0.025, hop=0.010):
    """Energy-based VAD -> (start_s, end_s, speech_ratio, rms_envelope, hop)."""
    wl, hl = int(win * sr), int(hop * sr)
    if len(x) < wl:
        return 0.0, len(x) / sr, 0.0, np.zeros(1), hop
    n = 1 + (len(x) - wl) // hl
    idx = np.arange(wl)[None, :] + hl * np.arange(n)[:, None]
    rms = np.sqrt((x[idx] ** 2).mean(1) + 1e-10)
    db = 20 * np.log10(rms + 1e-10)
    floor = np.percentile(db, 10)
    thr = max(floor + 12.0, db.max() - 40.0)     # adaptive threshold
    voiced = db > thr
    # smooth: close gaps < 200 ms, drop blips < 60 ms
    voiced = cv2.morphologyEx(voiced.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                              np.ones((1, 20), np.uint8))[0].astype(bool)
    voiced = cv2.morphologyEx(voiced.astype(np.uint8)[None, :], cv2.MORPH_OPEN,
                              np.ones((1, 6), np.uint8))[0].astype(bool)
    if not voiced.any():
        return 0.0, len(x) / sr, 0.0, rms, hop
    on = np.flatnonzero(voiced)
    start = max(0.0, on[0] * hop - 0.1)
    end = min(len(x) / sr, on[-1] * hop + win + 0.1)
    return round(start, 3), round(end, 3), round(float(voiced.mean()), 3), rms, hop


def mouth_motion(frames, track):
    """Mean absolute difference of the mouth region between consecutive frames."""
    ts = sorted(track)
    motion = np.full(N_FRAMES, np.nan)
    prev = None
    for t in ts:
        d = track[t]
        img = frames[t]
        mw = max(abs(d[12] - d[10]), 4.0)
        cx, cy = (d[10] + d[12]) / 2, (d[11] + d[13]) / 2 + 0.1 * mw
        m = cv2.cvtColor(square_crop(img, cx, cy, mw * 1.9, 48), cv2.COLOR_BGR2GRAY)
        m = m.astype(np.float32)
        if prev is not None and t - prev[0] <= 2:
            motion[t] = np.abs(m - prev[1]).mean()
        prev = (t, m)
    return motion


def grid(images, size):
    g = np.zeros((4 * size, 4 * size, 3), np.uint8)
    for i, im in enumerate(images[:N_FRAMES]):
        r, c = divmod(i, 4)
        g[r * size:(r + 1) * size, c * size:(c + 1) * size] = im
    return g


# ----------------------------------------------------------------------------
# One clip
# ----------------------------------------------------------------------------
def process_clip(ds_name, clip_id, duration, video, out_dir, multi_speaker, redo=False):
    meta_path = out_dir / f"{clip_id}.json"
    if meta_path.exists() and not redo:
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass

    fdir = PROC / ds_name / "frames" / clip_id
    wav = PROC / ds_name / "audio" / f"{clip_id}.wav"
    files = sorted(fdir.glob("*.jpg"))
    meta = dict(clip_id=clip_id, n_frames=len(files), status="ok")

    vpath = resolve_video(video)
    frames = read_video_frames(vpath) if vpath else []
    meta["source"] = "video" if len(frames) >= 4 else "jpg"
    if meta["source"] == "jpg":                   # fall back to the 224-px frames
        frames = [cv2.imread(str(f)) for f in files]
        frames = [f for f in frames if f is not None]
    if not frames:
        meta["status"] = "no_frames"
        meta_path.write_text(json.dumps(meta))
        return meta
    base = len(frames)
    while len(frames) < N_FRAMES:                 # short clips: repeat frames
        frames.append(frames[len(frames) % base])
    frames = frames[:N_FRAMES]

    # ---- speech segment ---------------------------------------------------
    rms, hop = np.zeros(1), 0.01
    if wav.exists():
        x, sr = read_wav(wav)
        s0, s1, ratio, rms, hop = speech_segment(x, sr)
        meta.update(speech_start=s0, speech_end=s1, speech_ratio=ratio,
                    audio_dur=round(len(x) / sr, 3))
    else:
        meta.update(speech_start=0.0, speech_end=0.0, speech_ratio=0.0, audio_dur=0.0)

    # ---- detection + tracking ---------------------------------------------
    dets = [detect(f) for f in frames]
    meta["faces_per_frame"] = [int(len(d)) for d in dets]
    shots = shot_boundaries(frames)
    meta["n_shots"] = len(shots)

    # audio energy sampled at each frame time
    dur = duration if duration and duration > 0 else len(rms) * hop
    t_frames = (np.arange(N_FRAMES) + 0.5) * dur / N_FRAMES
    a_idx = np.clip((t_frames / hop).astype(int), 0, len(rms) - 1)
    a_env = rms[a_idx]
    img_area = frames[0].shape[0] * frames[0].shape[1]

    # one target track per camera shot; shots are joined into the clip's face sequence
    track, all_scores, n_tracks = {}, [], 0
    for shot in shots:
        tracks = build_tracks(dets, shot)
        n_tracks += len(tracks)
        if not tracks:
            continue
        scores = []
        for tr in tracks:
            coverage = len(tr) / len(shot)
            size = np.mean([d[2] * d[3] for d in tr.values()]) / img_area
            conf = np.mean([d[14] for d in tr.values()])
            front = np.mean([frontalness(d) for d in tr.values()])
            sc = 1.0 * coverage + 2.0 * np.sqrt(size) + 1.0 * conf + 1.0 * front
            av_corr = 0.0
            if multi_speaker and len(tracks) > 1:
                mot = mouth_motion(frames, tr)
                ok = ~np.isnan(mot)
                if ok.sum() >= 4 and np.std(mot[ok]) > 1e-6 and np.std(a_env[ok]) > 1e-9:
                    av_corr = float(np.corrcoef(mot[ok], a_env[ok])[0, 1])
                motion_level = float(np.nanmean(mot)) / 20.0 if ok.any() else 0.0
                sc += 1.5 * av_corr + 1.0 * min(motion_level, 1.0)
            scores.append(dict(shot=shot[0], score=float(sc), coverage=float(coverage),
                               size=float(size), conf=float(conf), front=float(front),
                               av_corr=av_corr))
        best = int(np.argmax([x["score"] for x in scores]))
        track.update(tracks[best])
        all_scores.append(scores[best])
    meta["n_tracks"] = n_tracks
    meta["chosen_per_shot"] = all_scores

    if not track:
        meta["status"] = "no_face"
        blank_f = np.zeros((FACE_SIZE, FACE_SIZE, 3), np.uint8)
        blank_r = np.zeros((REGION_SIZE, REGION_SIZE, 3), np.uint8)
        cv2.imwrite(str(out_dir / f"{clip_id}_face.jpg"), grid([blank_f] * 16, FACE_SIZE))
        cv2.imwrite(str(out_dir / f"{clip_id}_eyes.jpg"), grid([blank_r] * 16, REGION_SIZE))
        cv2.imwrite(str(out_dir / f"{clip_id}_mouth.jpg"), grid([blank_r] * 16, REGION_SIZE))
        meta.update(face_mask=[0] * N_FRAMES, quality=[0.0] * N_FRAMES)
        meta_path.write_text(json.dumps(meta))
        return meta

    # ---- crops + quality --------------------------------------------------
    known = sorted(track)
    crops, quality_k = {}, {}
    for t in known:                               # crop only where the face was detected
        d = track[t]
        f, e, mo = aligned_face(frames[t], d)
        crops[t] = (f, e, mo)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        sharp = min(sharpness(g) / 300.0, 1.0)
        size_q = min(np.sqrt(d[2] * d[3] / img_area) / 0.35, 1.0)
        q = 0.35 * float(d[14]) + 0.25 * sharp + 0.2 * size_q + 0.2 * frontalness(d)
        quality_k[t] = round(float(q), 3)

    faces, eyes, mouths, mask, quality, boxes = [], [], [], [], [], []
    for t in range(N_FRAMES):
        src = t if t in track else min(known, key=lambda k: abs(k - t))
        f, e, mo = crops[src]                     # gaps reuse the nearest real crop
        faces.append(f), eyes.append(e), mouths.append(mo)
        mask.append(1 if t in track else 0)
        quality.append(quality_k[t] if t in track else 0.0)
        boxes.append([round(float(v), 1) for v in track[src][:4]])

    cv2.imwrite(str(out_dir / f"{clip_id}_face.jpg"), grid(faces, FACE_SIZE),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / f"{clip_id}_eyes.jpg"), grid(eyes, REGION_SIZE),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / f"{clip_id}_mouth.jpg"), grid(mouths, REGION_SIZE),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    meta.update(face_mask=mask, quality=quality, boxes=boxes,
                face_ratio=round(sum(mask) / N_FRAMES, 3))
    if sum(mask) < 4:
        meta["status"] = "few_faces"
    meta_path.write_text(json.dumps(meta))
    return meta


# ----------------------------------------------------------------------------
# Dataset loop
# ----------------------------------------------------------------------------
def run_dataset(key, limit, workers, only, redo=False):
    name = DATASETS[key]
    man = pd.read_csv(PROC / name / "manifest.csv")
    man = man[(man["n_frames"] > 0)].copy()
    if only:
        man = man[man["clip_id"].isin(only)]
    if limit:
        man = man.head(limit)
    out_dir = PROC / name / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    multi = key == "meld"
    print(f"\n=== {name}: {len(man)} clips ===")

    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(YUNET,)) as ex:
        futs = [ex.submit(process_clip, name, r.clip_id, float(r.duration), str(r.video),
                          out_dir, multi, redo)
                for r in man.itertuples()]
        for f in tqdm(as_completed(futs), total=len(futs), desc=name, unit="clip"):
            try:
                results.append(f.result())
            except Exception as e:                # never let one clip stop the run
                results.append(dict(clip_id="?", status=f"error: {e}"))

    rows = []
    for m in results:
        rows.append(dict(
            clip_id=m.get("clip_id"), status=m.get("status"),
            face_ratio=m.get("face_ratio", 0.0), n_tracks=m.get("n_tracks", 0),
            n_shots=m.get("n_shots", 1),
            max_faces=max(m.get("faces_per_frame", [0]) or [0]),
            mean_quality=round(float(np.mean(m.get("quality", [0.0]))), 3),
            speech_start=m.get("speech_start"), speech_end=m.get("speech_end"),
            speech_ratio=m.get("speech_ratio"), audio_dur=m.get("audio_dur"),
        ))
    seg = pd.DataFrame(rows)
    seg.to_csv(PROC / name / "segments.csv", index=False)

    print(seg["status"].value_counts().to_string())
    print(f"Frames with a face (mean): {seg['face_ratio'].mean():.1%}")
    print(f"Clips with 2+ faces on screen: {(seg['max_faces'] >= 2).mean():.1%}")
    print(f"Mean face quality: {seg['mean_quality'].mean():.2f}")
    print(f"Mean speech ratio: {seg['speech_ratio'].mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--only", nargs="*", default=None, help="specific clip ids (testing)")
    ap.add_argument("--redo", action="store_true", help="re-process clips that are already done")
    args = ap.parse_args()

    ensure_model()
    print(f"Processed data: {PROC}\nWorkers: {args.workers}")
    for k in args.datasets:
        run_dataset(k, args.limit, args.workers, args.only, args.redo)
    print("\nDone.")


if __name__ == "__main__":
    main()
