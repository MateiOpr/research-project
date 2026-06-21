
# Research Project

Scripts and datasets from a research project on facial-image privacy.

## Structure

- `amt-gan/` — AMT-GAN experiments
  - `scripts/` — ASR, SSIM, and upscaling utilities
  - `datasets/` — original, reconstructed, and thresholded image sets (200 samples)
  - `eval/` — evaluation outputs at varying thresholds
  - `test.py` — test entry point
- `event-privacy/` — event-camera privacy experiments
  - `scripts/` — clip/event conversion, dataset download, frame extraction, rendering
