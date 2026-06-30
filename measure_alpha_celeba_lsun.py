"""
measure_alpha_celeba_lsun.py — 测量 CelebA 和 LSUN Church 的功率谱幂律指数 alpha。

使用 HuggingFace datasets 加载数据。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class HFImageDataset(Dataset):
    """Wrap a HuggingFace dataset for PyTorch DataLoader."""

    def __init__(self, hf_dataset, transform, img_key='image'):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.img_key = img_key

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        img = self.hf_dataset[idx][self.img_key]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img), 0


def compute_radial_spectrum(dataloader, img_size, max_samples=50000):
    """计算数据集的径向平均功率谱。"""
    H = W = img_size
    acc_power = torch.zeros(H, W, dtype=torch.float64)
    count = 0

    for images, _ in dataloader:
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--celeba_cache', type=str,
                        default='/opt/data/bcmdata/ZONES/home/PROJECTS/homefile/PRIVATE/milksang/DDPM_trial/CelebA_HF')
    parser.add_argument('--lsun_cache', type=str,
                        default='/opt/data/bcmdata/ZONES/home/PROJECTS/homefile/PRIVATE/milksang/DDPM_trial/LSUN_Church_HF')
    parser.add_argument('--out', type=str, default='./alpha_celeba_lsun.png')
    args = parser.parse_args()

    t = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    print("=" * 70)
    print("  CelebA & LSUN Church Power Spectrum Analysis")
    print("  Formula: eta_opt = (3 - alpha_eff) / 2")
    print("=" * 70)

    results = {}

    # CelebA
    print(f"\n{'—' * 50}")
    print(f"  Loading CelebA...")
    print(f"{'—' * 50}")
    try:
        hf_celeba = load_dataset('flwrlabs/celeba', split='train', cache_dir=args.celeba_cache)
        ds_celeba = HFImageDataset(hf_celeba, transform=t, img_key='image')
        loader = DataLoader(ds_celeba, batch_size=256, shuffle=False, num_workers=4)
        k_arr, power, n_imgs = compute_radial_spectrum(loader, args.img_size, args.max_samples)

        k_max_fit = min(14, len(k_arr) - 2)
        alpha_global, R2_global, se_global = fit_alpha(k_arr, power, k_min=1, k_max=k_max_fit)
        k_mid_max = min(7, k_max_fit)
        alpha_mid, R2_mid, _ = fit_alpha(k_arr, power, k_min=3, k_max=k_mid_max)

        eta_leading = (3 - alpha_global) / 2
        eta_opt = (3 - alpha_mid) / 2

        print(f"  Images: {n_imgs}")
        print(f"  Global fit k=[1,{k_max_fit}]: alpha = {alpha_global:.3f} ± {se_global:.3f}, R² = {R2_global:.4f}")
        print(f"  Mid-freq  k=[3,{k_mid_max}]:  alpha_eff = {alpha_mid:.3f}, R² = {R2_mid:.4f}")
        print(f"  eta (leading)   = (3 - {alpha_global:.2f}) / 2 = {eta_leading:.3f}")
        print(f"  eta (corrected) = (3 - {alpha_mid:.2f}) / 2 = {eta_opt:.3f}")

        results['celeba'] = {
            'k_arr': k_arr, 'power': power,
            'alpha_global': alpha_global, 'R2_global': R2_global,
            'alpha_mid': alpha_mid, 'R2_mid': R2_mid,
            'eta_leading': eta_leading, 'eta_opt': eta_opt,
        }
    except Exception as e:
        print(f"  [SKIP] CelebA failed: {e}")

    # LSUN Church
    print(f"\n{'—' * 50}")
    print(f"  Loading LSUN Church...")
    print(f"{'—' * 50}")
    try:
        hf_lsun = load_dataset('tglcourse/lsun_church_train', split='train', cache_dir=args.lsun_cache)
        ds_lsun = HFImageDataset(hf_lsun, transform=t, img_key='image')
        loader = DataLoader(ds_lsun, batch_size=256, shuffle=False, num_workers=4)
        k_arr, power, n_imgs = compute_radial_spectrum(loader, args.img_size, args.max_samples)

        k_max_fit = min(14, len(k_arr) - 2)
        alpha_global, R2_global, se_global = fit_alpha(k_arr, power, k_min=1, k_max=k_max_fit)
        k_mid_max = min(7, k_max_fit)
        alpha_mid, R2_mid, _ = fit_alpha(k_arr, power, k_min=3, k_max=k_mid_max)

        eta_leading = (3 - alpha_global) / 2
        eta_opt = (3 - alpha_mid) / 2

        print(f"  Images: {n_imgs}")
        print(f"  Global fit k=[1,{k_max_fit}]: alpha = {alpha_global:.3f} ± {se_global:.3f}, R² = {R2_global:.4f}")
        print(f"  Mid-freq  k=[3,{k_mid_max}]:  alpha_eff = {alpha_mid:.3f}, R² = {R2_mid:.4f}")
        print(f"  eta (leading)   = (3 - {alpha_global:.2f}) / 2 = {eta_leading:.3f}")
        print(f"  eta (corrected) = (3 - {alpha_mid:.2f}) / 2 = {eta_opt:.3f}")

        results['lsun_church'] = {
            'k_arr': k_arr, 'power': power,
            'alpha_global': alpha_global, 'R2_global': R2_global,
            'alpha_mid': alpha_mid, 'R2_mid': R2_mid,
            'eta_leading': eta_leading, 'eta_opt': eta_opt,
        }
    except Exception as e:
        print(f"  [SKIP] LSUN Church failed: {e}")

    # 汇总
    if results:
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
        fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), squeeze=False)

        for ax, (name, r) in zip(axes[0], results.items()):
            k_plot = r['k_arr'][1:16]
            p_plot = r['power'][1:16]
            valid = p_plot > 0

            ax.semilogy(k_plot[valid], p_plot[valid], 'bo', ms=6)

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

        fig.suptitle(r'Power spectrum $D(k) \propto k^{-\alpha}$: CelebA & LSUN Church', fontsize=13)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\nFigure saved to {args.out}")


if __name__ == '__main__':
    main()
