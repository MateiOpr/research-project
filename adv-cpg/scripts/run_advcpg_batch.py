"""Run Adv-CPG on all images in an input folder with configurable target/prompt/naming."""
import argparse
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

ADV_CPG_ROOT = Path.home() / "event-privacy/Adv-CPG"


def run_one(source_path, target_path, prompt, out_path):
    cmd = [
        sys.executable, str(ADV_CPG_ROOT / "infer_adv_cpg.py"),
        "--base_model", "/scratch/moprescu/event-privacy/models/stable-diffusion-xl-base-1.0",
        "--advcpg_ckpt", "/scratch/moprescu/event-privacy/models/consistentid/ConsistentID_SDXL-v1.bin",
        "--image_encoder", "/scratch/moprescu/event-privacy/models/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "--ori_path", str(source_path),
        "--target_path", str(target_path),
        "--prompt", prompt,
        "--out", str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ADV_CPG_ROOT))
    if r.returncode != 0:
        print(f"  ERROR on {source_path.name}:")
        print(r.stderr[-500:])
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", required=True, help="Path to target identity image")
    p.add_argument("--prompt", required=True, help="Text prompt for generation")
    p.add_argument("--suffix", required=True, help="Naming suffix, e.g. einstein or minimal_prompt_einstein")
    args = p.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpg")))
    print(f"Found {len(images)} images in {in_dir}", flush=True)
    print(f"Target: {args.target}", flush=True)
    print(f"Prompt: {args.prompt}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Suffix: {args.suffix}", flush=True)

    ok = 0
    fail = 0
    for idx, img in enumerate(tqdm(images, desc=f"Adv-CPG-{args.suffix}"), start=1):
        out_path = out_dir / f"{idx}_{args.suffix}.png"
        if out_path.exists():
            ok += 1
            continue
        if run_one(img, args.target, args.prompt, out_path):
            ok += 1
        else:
            fail += 1

    print(f"\nSuccessful: {ok}", flush=True)
    print(f"Failed:     {fail}", flush=True)


if __name__ == "__main__":
    main()
