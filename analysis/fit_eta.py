"""
fit_eta.py — 从 CIFAR-10 的功率谱直接拟合幂律指数。

方法：
  1. 对所有训练图片做 2D FFT (norm='ortho')
  2. 取 |amplitude|^2，对 |k| 做径向平均得到 D(k)
  3. 在 log k - log D 空间做线性回归，斜率即为幂律指数
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_radial_power_spectrum(data_root='./CIFAR10', n_samples=50000):
    """计算 CIFAR-10 训练集的径向平均功率谱。"""
    dataset = CIFAR10(
        root=data_root, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=4)

    H = W = 32
    acc_power = torch.zeros(H, W, dtype=torch.float64)
    count = 0

    for images, _ in loader:
        # 2D FFT, 取振幅平方
        x_fft = torch.fft.fft2(images.float(), norm='ortho')  # (B, 3, H, W)
        power = (x_fft.real ** 2 + x_fft.imag ** 2).to(torch.float64)
        # 对 batch 和 channel 求平均
        acc_power += power.sum(dim=(0, 1))
        count += images.shape[0] * 3
        if count >= n_samples * 3:
            break

    avg_power = acc_power / count

    # 径向平均：按 |k| 分 bin
    kh = torch.arange(H, dtype=torch.float64)
    kw = torch.arange(W, dtype=torch.float64)
    kh_freq = torch.where(kh <= H // 2, kh, kh - H)
    kw_freq = torch.where(kw <= W // 2, kw, kw - W)
    kr = torch.sqrt(kh_freq[:, None] ** 2 + kw_freq[None, :] ** 2)

    kr_int = kr.int().flatten().numpy()
    power_flat = avg_power.flatten().numpy()

    k_max = int(kr.max().item()) + 1
    radial_sum = np.bincount(kr_int, weights=power_flat, minlength=k_max)
    radial_count = np.bincount(kr_int, minlength=k_max)
    radial_power = radial_sum / np.maximum(radial_count, 1)

    print(f"Computed from {count // 3} images, {count} channel-images total.")
    return np.arange(k_max), radial_power


def fit_power_law(k_arr, power_arr, k_min=1, k_max=14):
    """在 log-log 空间做线性回归: log D(k) = -alpha * log k + const."""
    mask = (k_arr >= k_min) & (k_arr <= k_max)
    k_fit = k_arr[mask].astype(float)
    p_fit = power_arr[mask]

    log_k = np.log(k_fit)
    log_p = np.log(p_fit)

    slope, intercept, r_value, p_value, std_err = linregress(log_k, log_p)
    alpha = -slope  # D(k) ∝ k^{-alpha}

    return alpha, slope, intercept, r_value ** 2, std_err


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'CIFAR10'))
    parser.add_argument('--k_min', type=int, default=1)
    parser.add_argument('--k_max', type=int, default=14)
    parser.add_argument('--out', type=str, default=os.path.join(PROJECT_ROOT, 'fit_eta_results.png'))
    args = parser.parse_args()

    print("=" * 60)
    print("  CIFAR-10 Power Spectrum: Power-Law Exponent Fitting")
    print("=" * 60)

    print("\nComputing radial power spectrum...")
    k_arr, radial_power = compute_radial_power_spectrum(args.data_root)

    print(f"\nRadial power spectrum D(k):")
    print(f"  {'k':>3s}  {'D(k)':>12s}  {'log k':>8s}  {'log D':>8s}")
    print("  " + "-" * 40)
    for k in range(min(17, len(k_arr))):
        if k == 0:
            print(f"  {k:3d}  {radial_power[k]:12.6f}  {'—':>8s}  {'—':>8s}")
        else:
            print(f"  {k:3d}  {radial_power[k]:12.6f}  {np.log(k):8.4f}  {np.log(radial_power[k]):8.4f}")

    print(f"\nFitting power law in range k=[{args.k_min}, {args.k_max}]...")
    alpha, slope, intercept, R2, std_err = fit_power_law(
        k_arr, radial_power, k_min=args.k_min, k_max=args.k_max)

    print(f"\n{'=' * 60}")
    print(f"  RESULT:")
    print(f"    D(k) ∝ k^{{-α}}")
    print(f"    α = {alpha:.4f} ± {std_err:.4f}")
    print(f"    R² = {R2:.6f}")
    print(f"{'=' * 60}")

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) log-log 图 + 拟合线
    ax = axes[0]
    k_plot = k_arr[1:17]
    p_plot = radial_power[1:17]
    ax.plot(np.log(k_plot), np.log(p_plot), 'bo', ms=7, label='CIFAR-10 data')

    # 拟合线
    k_line = np.linspace(np.log(args.k_min), np.log(args.k_max), 100)
    ax.plot(k_line, slope * k_line + intercept, 'r-', lw=2,
            label=rf'Fit: $\alpha = {alpha:.3f}$, $R^2 = {R2:.4f}$')

    # 标记拟合范围
    ax.axvline(np.log(args.k_min), color='gray', ls=':', alpha=0.5)
    ax.axvline(np.log(args.k_max), color='gray', ls=':', alpha=0.5)

    ax.set_xlabel(r'$\ln k$', fontsize=12)
    ax.set_ylabel(r'$\ln D(k)$', fontsize=12)
    ax.set_title(r'(a) Power spectrum: $D(k) \propto k^{-\alpha}$', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # (b) 原始 log-log scale
    ax = axes[1]
    ax.loglog(k_plot, p_plot, 'bo', ms=7, label='CIFAR-10 data')

    # 拟合幂律
    k_ref = np.linspace(1, 16, 100)
    ax.loglog(k_ref, np.exp(intercept) * k_ref ** slope, 'r-', lw=2,
              label=rf'$D(k) = {np.exp(intercept):.2f} \cdot k^{{-{alpha:.3f}}}$')

    ax.set_xlabel(r'$k$', fontsize=12)
    ax.set_ylabel(r'$D(k)$', fontsize=12)
    ax.set_title('(b) Power spectrum (log-log scale)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    fig.suptitle(rf'CIFAR-10 Power-Law Fit: $D(k) \propto k^{{-{alpha:.3f}}}$',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {args.out}")


if __name__ == '__main__':
    main()
