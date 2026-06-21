import argparse, cv2
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--size", type=int, default=512)
args = p.parse_args()

src = Path(args.input)
dst = Path(args.output)
dst.mkdir(parents=True, exist_ok=True)

for f in src.glob("*.png"):
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    scale = args.size / max(h, w)
    upscaled = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(dst / f.name), upscaled)
print(f"Upscaled {len(list(src.glob('*.png')))} images")
