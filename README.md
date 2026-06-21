# Research Project

Scripts and datasets used for the research project **"Privacy Preservation in Event-Based Vision: Risks, Methods, and Trade-offs"**, a thesis submitted to the EEMCS Faculty, Delft University of Technology, in partial fulfilment of the requirements for the Bachelor of Computer Science and Engineering, course CSE3000 Research Project 2026.

## Structure

- `amt-gan/` — AMT-GAN experiments
  - `scripts/` — ASR, SSIM, and upscaling utilities
  - `datasets/` — original, reconstructed, and thresholded image sets (200 samples)
  - `eval/` — evaluation scripts at varying thresholds
  - `test.py` — test entry point
- `adv-cpg/` — Adv-CPG experiments
  - `scripts/` — ASR, SSIM, and batch-run utilities
  - `jobs/` — SLURM batch and test job scripts
- `event-privacy/` — event-camera privacy experiments
  - `scripts/` — clip/event conversion, dataset download, frame extraction, rendering
