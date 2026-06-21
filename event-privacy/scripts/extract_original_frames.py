#Extract the middle frame of each CelebV-HQ clip as a PNG.
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def extract_middle_frame(video_path: Path, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return False
    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    return cv2.imwrite(str(out_path), frame)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", required=True, help="Folder of mp4 clips")
    p.add_argument("--out", required=True, help="Output folder for PNGs")
    args = p.parse_args()

    clips_dir = Path(args.clips)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(clips_dir.glob("*.mp4"))
    print(f"Found {len(clips)} clips")

    ok_count = 0
    fail_count = 0
    for clip in tqdm(clips, desc="Extracting"):
        out_path = out_dir / f"{clip.stem}.png"
        if out_path.exists():
            ok_count += 1
            continue
        if extract_middle_frame(clip, out_path):
            ok_count += 1
        else:
            fail_count += 1

    print(f"\nSuccessful: {ok_count}")
    print(f"Failed:     {fail_count}")


if __name__ == "__main__":
    main()
