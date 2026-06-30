import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_cifar10_batch(path):
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return d[b'data']  # (10000, 3072) uint8

data_dir = os.path.join(PROJECT_ROOT, 'CIFAR10/cifar-10-batches-py')
raw = load_cifar10_batch(f'{data_dir}/data_batch_1')
imgs = raw.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
imgs = (imgs - 0.5) / 0.5  # [-1, 1]

n = 10000
imgs = imgs[:n]
B, C, H, W = imgs.shape

acc_power = np.zeros((H, W), dtype=np.float64)
acc_mean = np.zeros((H, W), dtype=np.complex128)
count = 0

for b in range(B):
    for c in range(C):
        f_shift = np.fft.fftshift(np.fft.fft2(imgs[b, c]))
        acc_power += np.abs(f_shift) ** 2
        acc_mean += f_shift
        count += 1

acc_power /= count
acc_mean /= count

mean_sq = np.abs(acc_mean) ** 2
variance = acc_power - mean_sq

cy, cx = H // 2, W // 2
y, x = np.indices((H, W))
r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
nr = np.bincount(r.ravel())

radial_power = np.bincount(r.ravel(), acc_power.ravel()) / np.maximum(nr, 1)
radial_mean_sq = np.bincount(r.ravel(), mean_sq.ravel()) / np.maximum(nr, 1)
radial_var = np.bincount(r.ravel(), variance.ravel()) / np.maximum(nr, 1)

k_vals = np.arange(len(radial_power))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.semilogy(k_vals[1:], radial_power[1:], 'b-o', label=r'$\langle |x_k|^2 \rangle$', markersize=4)
ax.semilogy(k_vals[1:], radial_mean_sq[1:], 'r-s', label=r'$|\langle x_k \rangle|^2$', markersize=4)
ax.semilogy(k_vals[1:], radial_var[1:], 'g-^', label=r'$\mathrm{Var}[x_k]$', markersize=4)
ax.set_xlabel('k')
ax.set_ylabel('Spectral density')
ax.set_title('CIFAR-10 (10k images)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax = axes[1]
ratio = radial_mean_sq[1:] / radial_power[1:]
ax.bar(k_vals[1:], ratio, color='red', alpha=0.7)
ax.set_xlabel('k')
ax.set_ylabel(r'$|\langle x_k \rangle|^2 \,/\, \langle |x_k|^2 \rangle$')
ax.set_title('Fraction of total power from mean')
ax.grid(True, alpha=0.3)

ax = axes[2]
diff = radial_power[1:] - radial_var[1:]
ax.bar(k_vals[1:], diff, color='orange', alpha=0.7, label=r'$\langle|x_k|^2\rangle - \mathrm{Var}[x_k]$')
ax.set_xlabel('k')
ax.set_ylabel('Absolute difference')
ax.set_title(r'$|\langle x_k \rangle|^2$ (absolute value)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(PROJECT_ROOT, 'mean_sq_term_check.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")

print(f"\n{'k':>3s} {'<|x_k|^2>':>12s} {'|<x_k>|^2':>12s} {'Var[x_k]':>12s} {'ratio':>8s}")
print("-" * 50)
for k in range(0, min(23, len(k_vals))):
    r = radial_mean_sq[k] / radial_power[k] if radial_power[k] > 0 else 0
    print(f"{k:3d} {radial_power[k]:12.4f} {radial_mean_sq[k]:12.4f} {radial_var[k]:12.4f} {r:8.4f}")
