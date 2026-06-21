import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path

import ffmpeg
from tqdm import tqdm


def download_youtube_video(ytb_id, out_path):
    cmd = [
        "yt-dlp",
        "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "--no-warnings",
        "--no-playlist",
        "--remux-video", "mp4", 
        "-o", str(out_path),
        f"https://www.youtube.com/watch?v={ytb_id}",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180)
        if r.returncode != 0:
            print(f"\n    [yt-dlp error for {ytb_id}] {r.stderr.decode('utf-8').strip()}", flush=True)
            return False
        return out_path.exists()
    except Exception as e:
        print(f"\n    [Download exception for {ytb_id}] {e}", flush=True)
        return False


def crop_clip(in_video, out_clip, start, end, bbox):
    try:
        probe = ffmpeg.probe(str(in_video))
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        W, H = int(v["width"]), int(v["height"])
        
        x = int(bbox["left"] * W)
        y = int(bbox["top"] * H)
        cw = int((bbox["right"] - bbox["left"]) * W)
        ch = int((bbox["bottom"] - bbox["top"]) * H)
        
        #Dimensions need to be even (required by libx264)
        cw -= cw % 2
        ch -= ch % 2
        
        (
            ffmpeg.input(str(in_video), ss=start, to=end)
            .filter("crop", cw, ch, x, y)
            .output(str(out_clip), vcodec="libx264", preset="fast", crf=18, an=None)
            .overwrite_output()
            .run(quiet=True)
        )
        return out_clip.exists() and out_clip.stat().st_size > 1024
    
    except ffmpeg.Error as e:
        err_msg = e.stderr.decode('utf-8').strip() if e.stderr else 'Unknown'
        print(f"\n    [ffmpeg error] {err_msg}", flush=True)
        return False
    except Exception as e:
        print(f"\n    [Crop exception] {e}", flush=True)
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", type=int, default=50)
    p.add_argument("--max-attempts", type=int, default=300)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metadata) as f:
        data = json.load(f)
        
    clip_ids = list(data["clips"].keys())

    #Clips are shuffled, this prevents the script from burning through attempts on dead channels.
    random.shuffle(clip_ids)

    yt_cache = {}
    tmp_root = Path(tempfile.mkdtemp(prefix="celebv_"))
    successes = 0
    failures = 0

    print(f"Will attempt up to {args.max_attempts} clips, target {args.target} successful")
    pbar = tqdm(total=args.target, desc="Successful")

    for i, clip_id in enumerate(clip_ids):
        if successes >= args.target or i >= args.max_attempts:
            break

        info = data["clips"][clip_id]
        ytb_id = info["ytb_id"]
        start = info["duration"]["start_sec"]
        end = info["duration"]["end_sec"]
        bbox = info["bbox"]

        out_clip = out_dir / f"{clip_id}.mp4"
        if out_clip.exists():
            successes += 1
            pbar.update(1)
            continue

        if ytb_id not in yt_cache:
            yt_path = tmp_root / f"{ytb_id}.mp4"
            print(f"\n[{i+1}] Downloading YouTube ID {ytb_id} ...", flush=True)
            ok = download_youtube_video(ytb_id, yt_path)
            yt_cache[ytb_id] = yt_path if ok else None
            
            if not ok:
                print(f"    FAILED to download {ytb_id}", flush=True)
                failures += 1
                continue

        if yt_cache[ytb_id] is None:
            failures += 1
            continue

        ok = crop_clip(yt_cache[ytb_id], out_clip, start, end, bbox)
        if ok:
            successes += 1
            pbar.update(1)
        else:
            failures += 1

    pbar.close()
    
    #Cleanup
    for f in tmp_root.glob("*"):
        try:
            f.unlink()
        except Exception:
            pass
    try:
        tmp_root.rmdir()
    except Exception:
        pass

    print(f"\nSuccessful: {successes}")
    print(f"Failed:     {failures}")


if __name__ == "__main__":
    main()


