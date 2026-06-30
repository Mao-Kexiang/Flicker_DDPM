# Flicker_DDPM

Colored-noise DDPM on CIFAR-10. Demonstrates that replacing white Gaussian noise with spectrally colored noise (power-law kernel) improves sample quality at reduced diffusion steps.

## Project Structure

```
├── main.py              # CLI entry point (train / sample / FID eval)
├── models/              # Network architectures
│   ├── unet.py          # Unconditional UNet
│   └── unet_cfg.py      # Classifier-free guidance UNet
├── core/                # Core components
│   ├── diffusion.py     # DiffusionTrainer & DiffusionSampler
│   ├── noise.py         # NoiseModule (white / colored via Cholesky or FFT)
│   ├── train.py         # Training & sampling loops
│   ├── eval_fid.py      # FID evaluation pipeline
│   ├── scheduler.py     # Learning rate warmup scheduler
│   └── visualizer.py    # Training diagnostics & visualization
├── analysis/            # Theory verification & spectral analysis
│   ├── gamma_ode.py     # Measure γ(k,t) and validate linear-theory ODE
│   ├── fit_eta.py       # Fit power-law exponent from CIFAR-10 spectrum
│   ├── verify_matern.py # Matérn kernel theory verification
│   └── ...
├── scripts/             # Plotting scripts
│   ├── plot_fid_combined.py
│   ├── plot_fid_comparison_final.py
│   ├── plot_noise_comparison.py
│   └── plot_prediction2.py
└── docs/                # Reports & documentation
```

## Requirements

```
torch >= 2.0
torchvision
numpy
scipy
matplotlib
tqdm
pytorch-fid
```

## Usage

All commands are run from the project root directory.

### Training

```bash
# White noise baseline, T=500
python main.py --noise_type white --T 500 --epoch 200

# Colored noise (η=0.2), T=150
python main.py --noise_type colored --eta 0.2 --T 150 --epoch 200

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 main.py --noise_type colored --eta 0.2 --T 150
```

Checkpoints are saved to `./Checkpoints_{tag}/` automatically.

### Sampling

```bash
# Sample from a trained colored-noise model
python main.py --eval --noise_type colored --eta 0.2 --T 150 --ckpt ckpt_199_.pt
```

Generated images are saved to `./SampledImgs_{tag}/`.

### FID Evaluation

```bash
# Evaluate FID across training epochs
python main.py --fid --noise_type colored --eta 0.2 --T 150
```

Results (per-epoch FID curve) are saved to `./Checkpoints_{tag}/fid_data.npz`.

### Analysis Scripts

```bash
# Measure γ(k,t) and validate ODE predictions
python analysis/gamma_ode.py --noise_type colored --eta 0.2 --T 500

# Fit power-law exponent α from CIFAR-10
python analysis/fit_eta.py

# Generate comparison plots
python scripts/plot_fid_comparison_final.py
```
