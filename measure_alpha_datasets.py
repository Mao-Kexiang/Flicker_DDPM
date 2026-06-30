"""
measure_alpha_datasets.py — 测量多个数据集的功率谱幂律指数 alpha。

对每个数据集：
  1. 2D FFT (norm='ortho')
  2. |amplitude|^2 径向平均
  3. log-log 线性回归得到 alpha
  4. 预测最优 eta = (3 - alpha_eff) / 2
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def compute_radial_spectrum(dataloader, img_size, max_samples=50000):
    """计算数据集的径向平均功率谱。"""
    H = W = img_size
    acc_power = torch.zeros(H, W, dtype=torch.float64)
    count = 0

    for images, _ in dataloader:
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        x_fft = torch.fft.fft2(images.float(), norm='ortho')
        power = (x_fft.real ** 2 + x_fft.imag ** 2).to(torch.float64)
        acc_power += power.sum(dim=(0, 1))
        count += images.shape[0] * images.shape[1]
        if count >= max_samples * 3:
            break

    avg_power = acc_power / count

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

    return np.arange(k_max), radial_power, count // 3


def fit_alpha(k_arr, power, k_min=1, k_max=None):
    """Log-log 线性回归。"""
    if k_max is None:
        k_max = len(k_arr) - 2
    mask = (k_arr >= k_min) & (k_arr <= k_max)
    k_fit = k_arr[mask].astype(float)
    p_fit = power[mask]
    valid = p_fit > 0
    k_fit = k_fit[valid]
    p_fit = p_fit[valid]

    slope, intercept, r_val, _, std_err = linregress(np.log(k_fit), np.log(p_fit))
    return -slope, r_val ** 2, std_err


def get_dataset(name, img_size=32):
    """加载数据集。"""
    t = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    t_gray = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    if name == 'cifar10':
        ds = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=t)
    elif name == 'fashion_mnist':
        ds = datasets.FashionMNIST(root='./FashionMNIST', train=True, download=True, transform=t_gray)
    elif name == 'mnist':
        ds = datasets.MNIST(root='./MNIST', train=True, download=True, transform=t_gray)
    elif name == 'celeba':
        ds = datasets.CelebA(root='./CelebA', split='train', download=False,
                             transform=t)
    elif name == 'imagenet32':
        # 使用 CIFAR-100 作为 ImageNet32 的替代（类别更多，更接近 ImageNet 分布）
        ds = datasets.CIFAR100(root='./CIFAR100', train=True, download=True, transform=t)
    elif name == 'svhn':
        ds = datasets.SVHN(root='./SVHN', split='train', download=True, transform=t)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return ds


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['cifar10', 'fashion_mnist', 'mnist', 'svhn', 'imagenet32'])
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--out', type=str, default='./alpha_multi_datasets.png')
    args = parser.parse_args()

    print("=" * 70)
    print("  Multi-Dataset Power Spectrum Analysis")
    print("  Formula: eta_opt = (3 - alpha_eff) / 2")
    print("=" * 70)

    results = {}

    for name in args.datasets:
        print(f"\n{'—' * 50}")
        print(f"  Dataset: {name}")
        print(f"{'—' * 50}")

        try:
            ds = get_dataset(name, args.img_size)
        except Exception as e:
            print(f"  [SKIP] Failed to load: {e}")
            continue

        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
        k_arr, power, n_imgs = compute_radial_spectrum(loader, args.img_size)

        # 全局拟合
        k_max_fit = min(14, len(k_arr) - 2)
        alpha_global, R2_global, se_global = fit_alpha(k_arr, power, k_min=1, k_max=k_max_fit)

        # 中频拟合 (k=3-7)
        k_mid_max = min(7, k_max_fit)
        alpha_mid, R2_mid, _ = fit_alpha(k_arr, power, k_min=3, k_max=k_mid_max)

        eta_leading = (3 - alpha_global) / 2
        eta_opt = (3 - alpha_mid) / 2

        print(f"  Images: {n_imgs}")
        print(f"  Global fit k=[1,{k_max_fit}]: alpha = {alpha_global:.3f} ± {se_global:.3f}, R² = {R2_global:.4f}")
        print(f"  Mid-freq  k=[3,{k_mid_max}]:  alpha_eff = {alpha_mid:.3f}, R² = {R2_mid:.4f}")
        print(f"  eta (leading)   = (3 - {alpha_global:.2f}) / 2 = {eta_leading:.3f}")
        print(f"  eta (corrected) = (3 - {alpha_mid:.2f}) / 2 = {eta_opt:.3f}")

        results[name] = {
            'k_arr': k_arr, 'power': power,
            'alpha_global': alpha_global, 'R2_global': R2_global,
            'alpha_mid': alpha_mid, 'R2_mid': R2_mid,
            'eta_leading': eta_leading, 'eta_opt': eta_opt,
        }

    # 汇总表
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Dataset':<15} {'alpha_global':<13} {'alpha_mid':<11} {'eta_lead':<10} {'eta_opt':<10}")
    print(f"  {'-'*60}")
    for name, r in results.items():
        print(f"  {name:<15} {r['alpha_global']:<13.3f} {r['alpha_mid']:<11.3f} "
              f"{r['eta_leading']:<10.3f} {r['eta_opt']:<10.3f}")

    # 画图
    n_ds = len(results)
    fig, axes = plt.subplots(1, n_ds, figsize=(4.5 * n_ds, 4.5), squeeze=False)

    for ax, (name, r) in zip(axes[0], results.items()):
        k_plot = r['k_arr'][1:16]
        p_plot = r['power'][1:16]
        valid = p_plot > 0

        ax.semilogy(k_plot[valid], p_plot[valid], 'bo', ms=6)

        # 全局拟合线
        k_line = np.linspace(1, 14, 100)
        ax.semilogy(k_line, np.exp(np.log(r['power'][1]) - r['alpha_global'] * np.log(k_line)),
                    'r-', lw=2, alpha=0.7,
                    label=rf"$\alpha={r['alpha_global']:.2f}$" + "\n"
                          + rf"$\eta_{{opt}}={r['eta_opt']:.2f}$")

        ax.set_xlabel('k', fontsize=11)
        ax.set_ylabel('D(k)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0.8, 16)

    fig.suptitle(r'Power spectrum $D(k) \propto k^{-\alpha}$ across datasets', fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {args.out}")


if __name__ == '__main__':
    main()
