#Convert CelebV-HQ clips to event streams using Vid2e (esim_torch).
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

import esim_torch


def video_to_log_intensity(video_path):
    #Reads video and returns log-intensity frames, timestamps, (H, W)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    frames = []
    target_h, target_w = None, None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if target_h is None:
            target_h, target_w = gray.shape
            target_h -= target_h % 2
            target_w -= target_w % 2
        if gray.shape != (target_h, target_w):
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.float32) / 255.0
        log_img = np.log(gray + 1e-3)
        frames.append(log_img)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    log_frames = np.stack(frames, axis=0)
    H, W = log_frames.shape[1:]
    timestamps_ns = (np.arange(len(frames), dtype=np.int64) * (1e9 / fps)).astype(np.int64)
    return torch.from_numpy(log_frames).cuda(), timestamps_ns, (H, W)

def simulate_events(video_path: Path, out_path: Path, esim) -> bool:
    try:
        log_frames, ts_ns, (H, W) = video_to_log_intensity(video_path)
        ts_tensor = torch.from_numpy(ts_ns).cuda()

        events = esim.forward(log_frames, ts_tensor)
        if events is None or events["t"].numel() == 0:
            return False

        #save as h5
        t = events["t"].cpu().numpy()
        x = events["x"].cpu().numpy().astype(np.int16)
        y = events["y"].cpu().numpy().astype(np.int16)
        p = events["p"].cpu().numpy().astype(np.int8)

        with h5py.File(str(out_path), "w") as f:
            f.create_dataset("t", data=t, compression="gzip")
            f.create_dataset("x", data=x, compression="gzip")
            f.create_dataset("y", data=y, compression="gzip")
            f.create_dataset("p", data=p, compression="gzip")
            f.attrs["height"] = H
            f.attrs["width"] = W
        return True
    except Exception as e:
        print(f"  ERROR on {video_path.name}: {e}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--contrast-threshold", type=float, default=0.2,
                   help="ESIM contrast threshold (default 0.2; lower=more events)")
    args = p.parse_args()

    clips_dir = Path(args.clips)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ct = args.contrast_threshold
    clips = sorted(clips_dir.glob("*.mp4"))
    print(f"Found {len(clips)} clips")
    print(f"Contrast threshold: {ct}")
    ok_count = 0
    fail_count = 0
    for clip in tqdm(clips, desc="Simulating events"):
        out_path = out_dir / f"{clip.stem}.h5"
        if out_path.exists():
            ok_count += 1
            continue
        esim = esim_torch.ESIM(contrast_threshold_neg=ct,
                               contrast_threshold_pos=ct,
                               refractory_period_ns=0)
        if simulate_events(clip, out_path, esim):
            ok_count += 1
        else:
            fail_count += 1

    print(f"\nSuccessful: {ok_count}")
    print(f"Failed:     {fail_count}")


if __name__ == "__main__":
    main()
