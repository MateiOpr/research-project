
# Research Project

Scripts and datasets used for the research project "Privacy Preservation in Event-Based Vision: Risks, Methods, and Trade-offs", A Thesis Submitted to EEMCS Faculty Delft University of Technology,
In Partial Fulfilment of the Requirements
For the Bachelor of Computer Science and Engineering for the course CSE3000 Research Projec.

## Structure

- `amt-gan/` — AMT-GAN experiments
  - `scripts/` — ASR, SSIM, and upscaling utilities
  - `datasets/` — original, reconstructed, and thresholded image sets (200 samples)
  - `eval/` — evaluation outputs at varying thresholds
  - `test.py` — test entry point
- `event-privacy/` — event-camera privacy experiments
  - `scripts/` — clip/event conversion, dataset download, frame extraction, rendering
