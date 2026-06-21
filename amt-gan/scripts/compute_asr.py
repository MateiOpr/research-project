#Compute ASR (Attack Success Rate) against a target identity using InsightFace ArcFace."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def get_embedding(app, path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
    return faces[0].normed_embedding


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--protected", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--recursive", action="store_true")
    args = p.parse_args()

    print("Loading ArcFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(512, 512))

    target_emb = get_embedding(app, args.target)
    if target_emb is None:
        print("ERROR: no face in target")
        return

    src_dir = Path(args.source)
    prot_dir = Path(args.protected)
    src_files = {f.stem: f for f in src_dir.glob("*.png")}

    def evaluate(prot_files, label):
        common = sorted(set(src_files.keys()) & set(prot_files.keys()))
        sim_target, sim_source, hits, no_face = [], [], 0, 0
        for stem in common:
            p_emb = get_embedding(app, prot_files[stem])
            s_emb = get_embedding(app, src_files[stem])
            if p_emb is None:
                no_face += 1
                continue
            if s_emb is None:
                continue
            st = cosine(p_emb, target_emb)
            ss = cosine(p_emb, s_emb)
            sim_target.append(st)
            sim_source.append(ss)
            if st > args.threshold and st > ss:
                hits += 1
        total = len(sim_target)
        if total == 0:
            print(f"{label}: no valid pairs")
            return
        asr = hits / total * 100
        print(f"{label}: ASR={asr:.1f}%  sim_target={np.mean(sim_target):.3f}  sim_source={np.mean(sim_source):.3f}  n={total} no_face={no_face}")

    if args.recursive:
        for ref_folder in sorted(prot_dir.iterdir()):
            if not ref_folder.is_dir():
                continue
            prot_files = {f.stem: f for f in ref_folder.glob("*.png")}
            evaluate(prot_files, ref_folder.name)
    else:
        prot_files = {f.stem: f for f in prot_dir.glob("*.png")}
        evaluate(prot_files, "Protected")


if __name__ == "__main__":
    main()
