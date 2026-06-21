"""Compute Attack Success Rate (ASR) using InsightFace's ArcFace.

ASR = % of protected images for which FR identifies them as the TARGET
(instead of the original source identity).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm


def get_embedding(app, image_path):
    """Return the FR embedding for an image, or None if no face detected."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    # Pick the largest face
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Folder of source (original) images")
    p.add_argument("--protected", required=True, help="Folder of protected images")
    p.add_argument("--target", required=True, help="Path to target identity image (e.g. albert_einstein.jpg)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Cosine similarity threshold for 'same identity' (default 0.5)")
    args = p.parse_args()

    print("Loading FR model (ArcFace via InsightFace)...")
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(512, 512))  # ctx_id=-1 = CPU; use 0 if GPU avail

    # Compute target embedding (Einstein)
    target_emb = get_embedding(app, args.target)
    if target_emb is None:
        print("ERROR: no face detected in target image")
        return

    src_dir = Path(args.source)
    prot_dir = Path(args.protected)
    src_files = {f.stem: f for f in src_dir.glob("*.png")}
    prot_files = {f.stem: f for f in prot_dir.glob("*.png")}
    common = sorted(set(src_files.keys()) & set(prot_files.keys()))
    print(f"Evaluating {len(common)} pairs")

    target_hits = 0       # protected misidentified as target
    source_hits = 0       # protected still identified as source
    no_face_count = 0     # no face detected in protected

    sim_to_source_list = []
    sim_to_target_list = []

    for stem in tqdm(common, desc="ASR"):
        prot_emb = get_embedding(app, prot_files[stem])
        if prot_emb is None:
            no_face_count += 1
            continue

        src_emb = get_embedding(app, src_files[stem])
        if src_emb is None:
            continue

        sim_to_source = cosine_sim(prot_emb, src_emb)
        sim_to_target = cosine_sim(prot_emb, target_emb)

        sim_to_source_list.append(sim_to_source)
        sim_to_target_list.append(sim_to_target)

        # An attack "succeeds" if protected is closer to target than to source
        # AND the similarity to target is high enough to claim "same identity"
        if sim_to_target > args.threshold and sim_to_target > sim_to_source:
            target_hits += 1
        if sim_to_source > args.threshold:
            source_hits += 1

    total = len(sim_to_source_list)
    print(f"\nTotal evaluated: {total}")
    print(f"No face in protected: {no_face_count}")
    print(f"\nMean similarity to source:  {np.mean(sim_to_source_list):.4f}")
    print(f"Mean similarity to target:  {np.mean(sim_to_target_list):.4f}")
    print(f"\nAttack Success Rate (ASR):  {target_hits/total*100:.1f}%  ({target_hits}/{total})")
    print(f"Source still recognized:    {source_hits/total*100:.1f}%  ({source_hits}/{total})")


if __name__ == "__main__":
    main()
