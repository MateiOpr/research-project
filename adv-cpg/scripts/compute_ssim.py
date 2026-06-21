"""Compute SSIM between source images and protected images."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Folder of source images")
    p.add_argument("--protected", required=True, help="Folder of protected images")
    args = p.parse_args()

    src_dir = Path(args.source)
    prot_dir = Path(args.protected)

    src_files = {f.stem: f for f in src_dir.glob("*.png")}
    prot_files = {f.stem: f for f in prot_dir.glob("*.png")}

    common = sorted(set(src_files.keys()) & set(prot_files.keys()))
    print(f"Comparing {len(common)} pairs")

    scores = []
    for stem in tqdm(common, desc="SSIM"):
        src = cv2.imread(str(src_files[stem]))
        prot = cv2.imread(str(prot_files[stem]))

        # Resize protected to match source dimensions
        h, w = src.shape[:2]
        prot_resized = cv2.resize(prot, (w, h), interpolation=cv2.INTER_AREA)

        # Convert to grayscale for SSIM
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        prot_gray = cv2.cvtColor(prot_resized, cv2.COLOR_BGR2GRAY)

        score = ssim(src_gray, prot_gray, data_range=255)
        scores.append(score)

    print(f"\nMean SSIM: {np.mean(scores):.4f}")
    print(f"Std SSIM:  {np.std(scores):.4f}")
    print(f"Min SSIM:  {np.min(scores):.4f}")
    print(f"Max SSIM:  {np.max(scores):.4f}")


if __name__ == "__main__":
    main()
