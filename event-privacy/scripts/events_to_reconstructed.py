#Reconstruct grayscale frames using E2VID via its run_reconstruction.py script.

#Pipeline per clip:
#  1. Read .h5 event file
#  2. Pad voxel-grid dimensions to multiple of 16
#  3. Write a temporary zip in E2VID's expected text format
#  4. Call run_reconstruction.py
#  5. Pick middle reconstructed frame, save as RGB PNG, crop back
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

E2VID_ROOT = Path.home() / "event-privacy/repos/rpg_e2vid"


def round_up(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


def write_e2vid_zip(h5_path, zip_path, H_pad, W_pad):
    #Write events in E2VID's expected zip format.
    with h5py.File(str(h5_path), "r") as f:
        H = int(f.attrs["height"])
        W = int(f.attrs["width"])
        t_ns = f["t"][:]
        x = f["x"][:]
        y = f["y"][:]
        p = f["p"][:]

    #Convert to seconds and remap polarity from {-1, 1} to {0, 1}
    t_s = t_ns.astype(np.float64) / 1e9
    p01 = (p > 0).astype(np.int8)

    #Filter events that fall outside the padded grid
    mask = (x >= 0) & (x < W_pad) & (y >= 0) & (y < H_pad)
    t_s = t_s[mask]
    x = x[mask]
    y = y[mask]
    p01 = p01[mask]

    txt_path = Path(tempfile.mkdtemp()) / "events.txt"
    with open(txt_path, "w") as f:
        f.write(f"{W_pad} {H_pad}\n")
        for ti, xi, yi, pi in zip(t_s, x, y, p01):
            f.write(f"{ti:.6f} {xi} {yi} {pi}\n")

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(txt_path), arcname="events.txt")
    shutil.rmtree(txt_path.parent)


def run_e2vid(zip_path, out_dir, model_path):
    cmd = [
        sys.executable, str(E2VID_ROOT / "run_reconstruction.py"),
        "-c", str(model_path),
        "-i", str(zip_path),
        "--output_folder", str(out_dir),
        "--auto_hdr",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(E2VID_ROOT))
    if r.returncode != 0:
        print(f"\n  STDOUT: {r.stdout[-400:]}")
        print(f"  STDERR: {r.stderr[-400:]}")
        return False
    return True

def reconstruct_one(h5_path, out_png_path, model_path, tmp_root):
    #Run the full reconstruction for one event file. Save the middle frame as RGB PNG.
    with h5py.File(str(h5_path), "r") as f:
        H = int(f.attrs["height"])
        W = int(f.attrs["width"])

    H_pad = round_up(H, 16)
    W_pad = round_up(W, 16)

    work_dir = Path(tempfile.mkdtemp(prefix="e2vid_", dir=str(tmp_root)))
    zip_path = work_dir / "events.zip"
    recon_dir = work_dir / "out"
    recon_dir.mkdir()

    try:
        write_e2vid_zip(h5_path, zip_path, H_pad, W_pad)
        ok = run_e2vid(zip_path, recon_dir, model_path)
        if not ok:
            return False

        #E2VID writes reconstructed frames into recon_dir/<zip_stem>/frame_*.png
        frames = sorted(recon_dir.glob("**/frame_*.png"))
        if not frames:
            print(f"  No frames found in {recon_dir}")
            print(f"  Contents: {list(recon_dir.rglob('*'))[:10]}")
            return False
        middle = frames[len(frames) // 2]
        img = cv2.imread(str(middle), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False

        # Crop back to original H x W
        img = img[:H, :W]
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_rgb = img

        return cv2.imwrite(str(out_png_path), img_rgb)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--events", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=str(E2VID_ROOT / "pretrained/E2VID_lightweight.pth.tar"))
    args = p.parse_args()

    events_dir = Path(args.events)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="e2vid_pipeline_"))

    files = sorted(events_dir.glob("*.h5"))
    print(f"Found {len(files)} event files")
    print(f"Model: {args.model}")
    print(f"Tmp root: {tmp_root}")

    ok = 0
    fail = 0
    for f in tqdm(files, desc="Reconstructing"):
        out_path = out_dir / f"{f.stem}.png"
        if out_path.exists():
            ok += 1
            continue
        try:
            success = reconstruct_one(f, out_path, Path(args.model), tmp_root)
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  FAILED: {f.name}")
        except Exception as e:
            print(f"  ERROR on {f.name}: {e}")
            fail += 1

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\nSuccessful: {ok}")
    print(f"Failed:     {fail}")


if __name__ == "__main__":
    main()
