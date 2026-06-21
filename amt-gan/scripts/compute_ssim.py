#Compute SSIM between source images and protected images, averaged across all source images.
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
    p.add_argument("--recursive", action="store_true", help="Look in subfolders (for AMT-GAN ref01/ref02/...)")
    args = p.parse_args()

    src_dir = Path(args.source)
    prot_dir = Path(args.protected)

    src_files = {f.stem: f for f in src_dir.glob("*.png")}

    if args.recursive:
        #Each subfolder is one reference style
        for ref_folder in sorted(prot_dir.iterdir()):
            if not ref_folder.is_dir():
                continue
            prot_files = {f.stem: f for f in ref_folder.glob("*.png")}
            common = sorted(set(src_files.keys()) & set(prot_files.keys()))
            if not common:
                continue
            scores = []
            for stem in common:
                src = cv2.imread(str(src_files[stem]))
                prot = cv2.imread(str(prot_files[stem]))
                h, w = src.shape[:2]
                prot_resized = cv2.resize(prot, (w, h), interpolation=cv2.INTER_AREA)
                src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                prot_gray = cv2.cvtColor(prot_resized, cv2.COLOR_BGR2GRAY)
                score = ssim(src_gray, prot_gray, data_range=255)
                scores.append(score)
            print(f"{ref_folder.name}: SSIM mean={np.mean(scores):.4f}  n={len(scores)}")
    else:
        prot_files = {f.stem: f for f in prot_dir.glob("*.png")}
        common = sorted(set(src_files.keys()) & set(prot_files.keys()))
        scores = []
        for stem in tqdm(common, desc="SSIM"):
            src = cv2.imread(str(src_files[stem]))
            prot = cv2.imread(str(prot_files[stem]))
            h, w = src.shape[:2]
            prot_resized = cv2.resize(prot, (w, h), interpolation=cv2.INTER_AREA)
            src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            prot_gray = cv2.cvtColor(prot_resized, cv2.COLOR_BGR2GRAY)
            score = ssim(src_gray, prot_gray, data_range=255)
            scores.append(score)
        print(f"Mean SSIM: {np.mean(scores):.4f}  (n={len(scores)})")


if __name__ == "__main__":
    main()
