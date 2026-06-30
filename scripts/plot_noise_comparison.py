import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from core.noise import NoiseModule

torch.manual_seed(42)

img_size = 32
eta = 0.2

nm = NoiseModule("colored", img_size=img_size, eta=eta, method="fft")

white = torch.randn(1, 3, img_size, img_size)
colored = nm.colorize(white.clone())

def to_image(tensor):
    img = tensor[0].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    return img

def radial_power_spectrum(img_tensor):
    img = img_tensor[0].numpy()
    N = img.shape[-1]
    ps_sum = np.zeros(N // 2)
    counts = np.zeros(N // 2)

    freqs = np.fft.fftfreq(N, d=1.0)
    kx, ky = np.meshgrid(freqs, freqs)
    k_mag = np.sqrt(kx**2 + ky**2)
    k_bins = (k_mag * N).astype(int)

    for c in range(img.shape[0]):
        fft2 = np.fft.fft2(img[c], norm='ortho')
        power = np.abs(fft2) ** 2
        for i in range(1, N // 2):
            mask = (k_bins == i)
            if mask.sum() > 0:
                ps_sum[i] += power[mask].mean()
                counts[i] += 1

    valid = counts > 0
    ps_avg = np.zeros_like(ps_sum)
    ps_avg[valid] = ps_sum[valid] / counts[valid]
    return ps_avg

ps_white = radial_power_spectrum(white)
ps_colored = radial_power_spectrum(colored)

# === Combined figure: 3 panels ===
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1.1], left=0.02, right=0.98, bottom=0.20, top=0.88, wspace=0.35)
gs_left = gs[0].subgridspec(1, 2, wspace=0.15)

# Panel (a): White noise
ax0 = fig.add_subplot(gs_left[0])
ax0.imshow(to_image(white))
ax0.set_title(r'White Noise ($\Sigma = $ I)', fontsize=28)
ax0.axis('off')

# Panel (b): Colored noise
ax1 = fig.add_subplot(gs_left[1])
ax1.imshow(to_image(colored))
ax1.set_title(r'Colored Noise ($\eta$=0.2)', fontsize=28)
ax1.axis('off')

# Panel (c): Power spectrum
ax2 = fig.add_subplot(gs[1])
k = np.arange(len(ps_white))
valid = k >= 1
ax2.loglog(k[valid], ps_white[valid], 'o-', color='#1f77b4', label=r'$\Sigma = $ I', markersize=4)
ax2.loglog(k[valid], ps_colored[valid], 's-', color='#d62728', label=r'$\eta$=0.2', markersize=4)

k_fit = k[valid].astype(float)
log_k = np.log(k_fit)
log_ps = np.log(ps_colored[valid] + 1e-30)
finite_mask = np.isfinite(log_ps)
if finite_mask.sum() > 2:
    slope, intercept = np.polyfit(log_k[finite_mask], log_ps[finite_mask], 1)
    fit_line = np.exp(intercept) * k_fit**slope
    ax2.loglog(k_fit, fit_line, '--', color='#d62728', alpha=0.4,
               label=r'$P(k) \sim k^{-\alpha}$', linewidth=2)
    mid_idx = len(k_fit) // 2
    ax2.annotate(rf'$\alpha = {-slope:.2f}$',
                 xy=(k_fit[mid_idx], fit_line[mid_idx]),
                 xytext=(10, 10), textcoords='offset points',
                 fontsize=18, color='#d62728')

ax2.set_xlabel('Spatial Frequency k', fontsize=28)
ax2.set_ylabel('Power Spectral Density', fontsize=28)
ax2.set_title('Power Spectrum', fontsize=28)
ax2.legend(fontsize=20, loc='upper right')
ax2.tick_params(labelsize=28)
ax2.grid(True, alpha=0.3, which='both')

fig.savefig(os.path.join(PROJECT_ROOT, 'noise_combined.png'), dpi=150)
print("Saved noise_combined.png")
