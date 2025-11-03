# CAR Testing Code

This folder contains analysis utilities that replay the InfinityPilot preprocessing
pipeline with the same configuration used in `train_pilot.sh`. Each script loads the
pretrained Infinity 2B checkpoint (`/mnt/syndata/VAR_ckpt/infinity_2b_reg.pth`) together
with the BSQ VAE (`weights/infinity_vae_d32_reg.pth`), runs the BitwiseSelfCorrection
latents, and prints per-stage statistics so you can inspect CAR ↔ VAR interactions.

## Files

- `run_current_method.py` – compares the production CAR control convs (current method)
  against a hypothetical VAE-shared control branch within the same run.
- `run_shared_vqvae_method.py` – focuses on the shared-VAE control branch only, printing
  detailed per-scale token statistics.
- `run_infinity_only.py` – disables CAR entirely and traces the base Infinity transformer
  under the pretrained weights.
- `common.py` – shared helpers for loading checkpoints, encoding latents, and replaying
  the forward pass while collecting tensor summaries.

## Usage

From the repository root:

```bash
python -m car_testing_code.run_current_method
python -m car_testing_code.run_shared_vqvae_method
python -m car_testing_code.run_infinity_only
```

Both scripts print per-stage statistics (mean, std, min, max) for:

1. Raw inputs and VAE encoded features.
2. Per-stage control/VAR token preparation.
3. Outputs of CAR blocks and the residual fusion inside the Infinity transformer.

Since the scripts rely on the pretrained checkpoints, make sure the following files are
available before running:

- `/mnt/syndata/VAR_ckpt/infinity_2b_reg.pth`
- `weights/infinity_vae_d32_reg.pth`
