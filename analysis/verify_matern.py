"""
verify_matern.py — 验证 Matérn 协方差族能否解释 CIFAR-10 的 α=2.7。

理论：
  当前核 C(r) = (r+1)^{-η} 是纯代数衰减，FT 给出 Σ(k) ∝ k^{-(2-η)}，上限 k^{-2}。

  缺失的物理：自然图像有有限关联长度 ξ（物体边界、纹理尺度）。
  正确描述是 Matérn 协方差族：
    C(r) ∝ (r/ξ)^ν K_ν(r/ξ)
  对应功率谱：
    Σ(k) ∝ (k² + κ²)^{-(ν + d/2)}，  κ = 1/ξ
  大 k 极限：Σ(k) ∝ k^{-2(ν+1)} （2D）

  匹配 α=2.7：2(ν+1) = 2.7 → ν = 0.35

本脚本验证 Matérn 谱对 CIFAR-10 数据的拟合质量。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import linregress
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_radial_power_spectrum(data_root=None, n_samples=50000):
    if data_root is None:
        data_root = os.path.join(PROJECT_ROOT, 'CIFAR10')
    """计算 CIFAR-10 的径向平均功率谱。"""
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
        x_fft = torch.fft.fft2(images.float(), norm='ortho')
        power = (x_fft.real ** 2 + x_fft.imag ** 2).to(torch.float64)
        acc_power += power.sum(dim=(0, 1))
        count += images.shape[0] * 3
        if count >= n_samples * 3:
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

    return np.arange(k_max), radial_power


def matern_spectrum(k, sigma2, kappa, nu):
    """Matérn 功率谱 (2D): Σ(k) = σ² * (k² + κ²)^{-(ν+1)}"""
    return sigma2 * (k ** 2 + kappa ** 2) ** (-(nu + 1))


def pure_powerlaw(k, A, alpha):
    """纯幂律谱: Σ(k) = A * k^{-α}"""
    return A * k ** (-alpha)


def old_kernel_spectrum(k, eta):
    """旧核的理论谱 (连续极限): Σ(k) ∝ k^{-(2-η)}"""
    alpha_eff = 2 - eta
    return k ** (-alpha_eff)


def fit_matern(k_arr, D_data, k_min=1, k_max=14):
    """拟合 Matérn 参数 (σ², κ, ν)."""
    mask = (k_arr >= k_min) & (k_arr <= k_max)
    k_fit = k_arr[mask].astype(float)
    D_fit = D_data[mask]
    log_D = np.log(D_fit)

    def residual(params):
        log_sigma2, log_kappa, nu = params
        sigma2 = np.exp(log_sigma2)
        kappa = np.exp(log_kappa)
        if nu <= 0:
            return 1e10
        pred = matern_spectrum(k_fit, sigma2, kappa, nu)
        return np.sum((np.log(pred) - log_D) ** 2)

    # 初始猜测: α=2.7 → ν=0.35, κ~0.5, σ²~D(k=1)
    x0 = [np.log(D_data[1] * 2), np.log(0.3), 0.35]
    from scipy.optimize import minimize
    res = minimize(residual, x0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-6})

    sigma2 = np.exp(res.x[0])
    kappa = np.exp(res.x[1])
    nu = res.x[2]

    pred = matern_spectrum(k_fit, sigma2, kappa, nu)
    ss_res = np.sum((np.log(pred) - log_D) ** 2)
    ss_tot = np.sum((log_D - log_D.mean()) ** 2)
    R2 = 1 - ss_res / ss_tot

    return sigma2, kappa, nu, R2


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=os.path.join(PROJECT_ROOT, 'CIFAR10'))
    parser.add_argument('--out', type=str, default=os.path.join(PROJECT_ROOT, 'theory_alpha_gt2.png'))
    args = parser.parse_args()

    print("=" * 60)
    print("  Why α > 2: Matérn vs Power-Law Kernel Analysis")
    print("=" * 60)

    print("\nComputing CIFAR-10 power spectrum...")
    k_arr, D_data = compute_radial_power_spectrum(args.data_root)

    # 1. 纯幂律拟合
    mask = (k_arr >= 1) & (k_arr <= 14)
    k_fit = k_arr[mask].astype(float)
    D_fit = D_data[mask]
    slope, intercept, r_val, _, std_err = linregress(np.log(k_fit), np.log(D_fit))
    alpha_pl = -slope
    R2_pl = r_val ** 2

    print(f"\n[1] Pure power-law fit: D(k) = A·k^{{-α}}")
    print(f"    α = {alpha_pl:.4f} ± {std_err:.4f},  R² = {R2_pl:.6f}")

    # 2. Matérn 拟合
    sigma2, kappa, nu, R2_m = fit_matern(k_arr, D_data)
    xi = 1.0 / kappa
    alpha_matern_hk = 2 * (nu + 1)

    print(f"\n[2] Matérn fit: Σ(k) = σ²·(k² + κ²)^{{-(ν+1)}}")
    print(f"    ν = {nu:.4f}")
    print(f"    κ = {kappa:.4f}  (ξ = 1/κ = {xi:.2f} pixels)")
    print(f"    σ² = {sigma2:.4f}")
    print(f"    High-k exponent: 2(ν+1) = {alpha_matern_hk:.4f}")
    print(f"    R² = {R2_m:.6f}")

    # 3. 旧核能达到的最大指数
    print(f"\n[3] Old kernel C(r) = (r+1)^{{-η}} limits:")
    print(f"    Max spectral exponent (η→0): α = 2.0")
    print(f"    At η=0.2: α = 1.8")
    print(f"    Gap to data: {alpha_pl:.2f} - 2.0 = {alpha_pl - 2:.2f}")

    # 4. 物理解释
    print(f"\n{'=' * 60}")
    print(f"  THEORETICAL EXPLANATION")
    print(f"{'=' * 60}")
    print(f"""
  缺失的物理：有限关联长度 ξ ≈ {xi:.1f} pixels

  (r+1)^{{-η}} 假设无穷远处仍有代数关联，但自然图像中：
    - 物体有边界 → 关联在物体尺度处截断
    - 不同物体之间统计独立 → 指数衰减
    - CIFAR-10 (32×32) 中 ξ ≈ {xi:.1f} px 对应物体特征尺度

  正确的协方差函数应包含指数截断：
    C(r) ∝ (r/ξ)^ν · K_ν(r/ξ)   [Matérn 族]

  频域：Σ(k) = σ² · (k² + κ²)^{{-(ν+1)}}
    - k ≫ κ: Σ(k) ∝ k^{{-2(ν+1)}} = k^{{-{alpha_matern_hk:.2f}}}  ← 可以 >2!
    - k ≪ κ: Σ(k) → const (饱和)

  物理图像：
    - 低频 (k < κ): 不同物体的统计平均 → 功率饱和
    - 高频 (k > κ): 单个物体内部的光滑结构 → 陡峭衰减

  为什么旧核不行：
    (r+1)^{{-η}} 的 FT 在 2D 中给出 k^{{-(2-η)}}
    → 上限 k^{{-2}} (η→0)
    → 物理上：纯代数衰减缺少"关联截断"机制
    → 等效于假设图像中的空间关联无穷远仍存在
""")

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) 数据 vs 三种模型
    ax = axes[0, 0]
    k_plot = np.arange(1, 16, dtype=float)
    ax.semilogy(k_plot, D_data[1:16], 'ko', ms=8, label=r'CIFAR-10 $D(k)$')

    # Matérn
    ax.semilogy(k_plot, matern_spectrum(k_plot, sigma2, kappa, nu), 'r-', lw=2.5,
                label=rf'Matérn: $\nu={nu:.2f}$, $\kappa={kappa:.2f}$, $R^2={R2_m:.4f}$')

    # Pure power law
    ax.semilogy(k_plot, np.exp(intercept) * k_plot ** slope, 'b--', lw=2,
                label=rf'Power law: $k^{{-{alpha_pl:.2f}}}$, $R^2={R2_pl:.4f}$')

    # Old kernel limit (η=0, best case)
    old_spec = old_kernel_spectrum(k_plot, 0.0)
    old_spec *= D_data[1] / old_spec[0]  # 归一化到 k=1
    ax.semilogy(k_plot, old_spec, 'g:', lw=2,
                label=r'Old kernel limit ($\eta\to 0$): $k^{-2}$')

    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('D(k)', fontsize=12)
    ax.set_title('(a) CIFAR-10 spectrum vs models', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Matérn 的两个 regime
    ax = axes[0, 1]
    k_ext = np.linspace(0.1, 20, 200)
    spec_full = matern_spectrum(k_ext, sigma2, kappa, nu)
    ax.loglog(k_ext, spec_full, 'r-', lw=2.5, label=rf'Matérn($\nu={nu:.2f}$, $\kappa={kappa:.2f}$)')

    # 标记两个 regime
    ax.axvline(kappa, color='gray', ls='--', alpha=0.7, label=rf'$\kappa = {kappa:.2f}$ (crossover)')

    # 渐近线
    k_high = k_ext[k_ext > 2 * kappa]
    ax.loglog(k_high, sigma2 * k_high ** (-2 * (nu + 1)), 'b:', lw=1.5,
              label=rf'$k^{{-{2*(nu+1):.2f}}}$ (high-$k$ asymptote)')

    ax.loglog(k_plot, D_data[1:16], 'ko', ms=6)
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel(r'$\Sigma(k)$', fontsize=12)
    ax.set_title(r'(b) Matérn: two regimes ($k \lessgtr \kappa$)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # (c) 实空间协方差对比
    ax = axes[1, 0]
    from scipy.special import kv as besselk, gamma as gammafn
    r_arr = np.linspace(0.1, 15, 200)

    # Matérn C(r)
    x = r_arr / xi
    matern_cr = (2 ** (1 - nu) / gammafn(nu)) * (x ** nu) * besselk(nu, x)
    matern_cr /= matern_cr[0]

    # Old kernel
    for eta_val in [0.2, 0.5, 1.0]:
        old_cr = (r_arr + 1) ** (-eta_val)
        old_cr /= old_cr[0]
        ax.plot(r_arr, old_cr, '--', lw=1.5, alpha=0.7,
                label=rf'$(r+1)^{{-{eta_val}}}$')

    ax.plot(r_arr, matern_cr, 'r-', lw=2.5,
            label=rf'Matérn($\nu={nu:.2f}$, $\xi={xi:.1f}$)')
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.axvline(xi, color='gray', ls='--', alpha=0.5, label=rf'$\xi = {xi:.1f}$ px')

    ax.set_xlabel('r (pixels)', fontsize=12)
    ax.set_ylabel('C(r) / C(0)', fontsize=12)
    ax.set_title('(c) Real-space correlation: Matérn vs old kernel', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.05)

    # (d) 指数 gap 可视化
    ax = axes[1, 1]
    eta_arr = np.linspace(0, 1.5, 50)
    alpha_old = 2 - eta_arr  # 旧核能达到的指数

    ax.fill_between(eta_arr, 0, alpha_old, alpha=0.2, color='green',
                    label=r'Old kernel achievable: $\alpha = 2 - \eta$')
    ax.plot(eta_arr, alpha_old, 'g-', lw=2)
    ax.axhline(alpha_pl, color='red', lw=2.5, ls='-',
               label=rf'CIFAR-10: $\alpha = {alpha_pl:.2f}$')
    ax.axhline(2.0, color='green', lw=1, ls=':', alpha=0.7)
    ax.axhline(alpha_matern_hk, color='blue', lw=2, ls='--',
               label=rf'Matérn high-$k$: $2(\nu+1) = {alpha_matern_hk:.2f}$')

    # 标注 gap
    ax.annotate('', xy=(0.2, 2.0), xytext=(0.2, alpha_pl),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(0.35, (2.0 + alpha_pl) / 2, f'gap = {alpha_pl - 2:.2f}',
            fontsize=11, color='red', va='center')

    ax.set_xlabel(r'$\eta$', fontsize=12)
    ax.set_ylabel(r'Spectral exponent $\alpha$', fontsize=12)
    ax.set_title(r'(d) Achievable $\alpha$: old kernel vs data requirement', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 1.5)

    fig.suptitle(
        r'Why $\alpha > 2$: finite correlation length $\xi$ steepens the spectrum',
        fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {args.out}")


if __name__ == '__main__':
    main()
