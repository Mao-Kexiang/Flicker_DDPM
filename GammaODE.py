"""
GammaODE.py — Measure gamma(k,t) from the denoising network and validate
the linear-theory ODE (eq 1.1 of Theory_Report_Full.md).

Theory equation (1.1):
    dD(k,t)/dt = beta(t) * [(1-2*gamma(k,t)) * D(k,t) + Sigma_tilde(k)]

Method:
  1. Measure gamma(kh,kw,t) at every DDPM timestep via linear regression
     of eps_theta on x_t in Fourier space (symmetric/unitary FFT).
  2. Run the actual reverse process and record D(kh,kw) at every step.
  3. Numerically integrate the ODE using measured gamma — both the
     discrete-exact DDPM formula and the continuous Euler approximation.
  4. Compare predicted D(k,t) with actual and generate plots.

All Fourier transforms use norm='ortho' (symmetric/unitary convention).
"""

import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, os.path.dirname(__file__))
from Train import _build_model, _build_noise_module, _make_eval_labels
from Diffusion import extract


# ------------------------------------------------------------------ #
#  Utilities
# ------------------------------------------------------------------ #

def make_k_grid(H, W):
    """Radial frequency |k| for each (kh, kw) in standard FFT layout."""
    kh = np.arange(H, dtype=np.float64)
    kw = np.arange(W, dtype=np.float64)
    kh_freq = np.where(kh <= H // 2, kh, kh - H)
    kw_freq = np.where(kw <= W // 2, kw, kw - W)
    return np.sqrt(kh_freq[:, None] ** 2 + kw_freq[None, :] ** 2)


def radial_profile(data_2d, img_size):
    """Radially average a 2D array in FFT layout."""
    kr = make_k_grid(img_size, img_size)
    kr_int = kr.astype(int).ravel()
    tbin = np.bincount(kr_int, data_2d.ravel())
    nr = np.bincount(kr_int)
    return np.arange(len(nr)), tbin / np.maximum(nr, 1)


def compute_phi(D_2d, img_size, k_cut=None):
    """Spectral order parameter phi = P_L / P_H.

    Args:
        D_2d: (..., H, W) power spectrum array
        img_size: spatial size H = W
        k_cut: frequency cutoff (default k_max/2)
    Returns:
        phi: (...) array of P_L / P_H
    """
    kr = make_k_grid(img_size, img_size)
    if k_cut is None:
        k_cut = img_size // 4
    low = (kr > 0) & (kr <= k_cut)
    high = kr > k_cut
    P_L = D_2d[..., low].sum(axis=-1)
    P_H = D_2d[..., high].sum(axis=-1)
    return P_L / np.maximum(P_H, 1e-12)


def compute_2d_variance(x_batch):
    """D(kh,kw) = Var[x_tilde(k)] over batch and channels (unitary FFT).

    Args:
        x_batch: (B, C, H, W) tensor
    Returns:
        (H, W) numpy float64 array
    """
    x_fft = torch.fft.fft2(x_batch.float(), norm='ortho')
    flat = x_fft.reshape(-1, x_fft.shape[-2], x_fft.shape[-1]).to(torch.complex128)
    power = (flat.real ** 2 + flat.imag ** 2).mean(dim=0)
    m = flat.mean(dim=0)
    return (power - m.real ** 2 - m.imag ** 2).clamp(min=0).cpu().numpy()


def compute_cifar10_spectrum_2d(data_root, n_samples=10000):
    """D_data(kh,kw) from CIFAR-10 training set ([-1,1] normalisation)."""
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
    acc_mean = torch.zeros(H, W, dtype=torch.complex128)
    count = 0

    for images, _ in loader:
        x_fft = torch.fft.fft2(images.float(), norm='ortho')
        flat = x_fft.reshape(-1, H, W).to(torch.complex128)
        acc_power += (flat.real ** 2 + flat.imag ** 2).sum(dim=0)
        acc_mean += flat.sum(dim=0)
        count += flat.shape[0]
        if count >= n_samples * 3:
            break

    acc_power /= count
    acc_mean /= count
    D = acc_power - acc_mean.real ** 2 - acc_mean.imag ** 2
    print(f"D_data computed from {count // 3} images.")
    return D.numpy()


def _make_labels(config, batch_size, device):
    """Create CFG labels of the required batch size."""
    num_labels = config.get("num_labels", 10)
    step = batch_size // num_labels
    labels = []
    k = 0
    for i in range(1, batch_size + 1):
        labels.append(k)
        if i % step == 0 and k < num_labels - 1:
            k += 1
    return torch.tensor(labels, dtype=torch.long, device=device) + 1


def _model_forward(model, x, t_tensor, labels, w_cfg):
    """Single forward pass with optional CFG."""
    if labels is not None:
        eps = model(x, t_tensor, labels)
        if w_cfg > 0:
            eps_u = model(x, t_tensor, torch.zeros_like(labels))
            eps = (1.0 + w_cfg) * eps - w_cfg * eps_u
        return eps
    return model(x, t_tensor)


def _chunked_forward(model, x, t_tensor, labels, w_cfg, chunk):
    """Model forward in memory-friendly chunks."""
    B = x.shape[0]
    out = torch.zeros_like(x)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        lc = labels[s:e] if labels is not None else None
        out[s:e] = _model_forward(model, x[s:e], t_tensor[s:e], lc, w_cfg)
    return out


# ------------------------------------------------------------------ #
#  Gamma measurement
# ------------------------------------------------------------------ #

def measure_gamma(model, config, device, x0, chunk_size=64):
    """Measure gamma(kh,kw,t) at every timestep via linear regression.

    gamma(k,t) = Re[Cov(eps_tilde_theta, x_tilde_t)] /
                 [sqrt(1-alpha_bar_t) * Var(x_tilde_t)]

    Args:
        x0: (B, 3, H, W) CIFAR-10 images on device
    Returns:
        gamma: (T, H, W) float64 array
        R2: (T, H, W) float64 array — coefficient of determination
    """
    T = config["T"]
    B, C, H, W = x0.shape

    betas = torch.linspace(config["beta_1"], config["beta_T"], T,
                           dtype=torch.float64, device=device)
    alphas_bar = torch.cumprod(1.0 - betas, dim=0)

    mode = config.get("mode", "unconditional")
    w_cfg = config.get("w", 0.0)
    labels = _make_labels(config, B, device) if mode == "cfg" else None

    gamma = np.zeros((T, H, W), dtype=np.float64)
    R2 = np.zeros((T, H, W), dtype=np.float64)

    print(f"Measuring gamma: T={T}, batch={B}, chunk={chunk_size}")
    for t in range(T):
        ab_t = alphas_bar[t].float()
        sqrt_ab = torch.sqrt(ab_t)
        sqrt_1mab = torch.sqrt(1.0 - ab_t)

        eps = torch.randn_like(x0)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps

        t_tensor = torch.full([B], t, dtype=torch.long, device=device)
        with torch.no_grad():
            eps_theta = _chunked_forward(
                model, x_t, t_tensor, labels, w_cfg, chunk_size)

        x_fft = torch.fft.fft2(x_t.float(), norm='ortho').reshape(B * C, H, W)
        e_fft = torch.fft.fft2(eps_theta.float(), norm='ortho').reshape(B * C, H, W)
        x_fft = x_fft.to(torch.complex128)
        e_fft = e_fft.to(torch.complex128)

        ex_conj = (e_fft * x_fft.conj()).mean(dim=0)
        e_mean = e_fft.mean(dim=0)
        x_mean = x_fft.mean(dim=0)
        x_pow = (x_fft.real ** 2 + x_fft.imag ** 2).mean(dim=0)

        cov = ex_conj - e_mean * x_mean.conj()
        var_x = x_pow - x_mean.real ** 2 - x_mean.imag ** 2

        slope_re = cov.real / var_x.clamp(min=1e-12)
        gamma[t] = (slope_re / max(sqrt_1mab.item(), 1e-10)).cpu().numpy()

        e_pow = (e_fft.real ** 2 + e_fft.imag ** 2).mean(dim=0)
        var_eps = e_pow - e_mean.real ** 2 - e_mean.imag ** 2
        R2[t] = (slope_re ** 2 * var_x / var_eps.clamp(min=1e-12)).clamp(0, 1).cpu().numpy()

        if t % 50 == 0 or t == T - 1:
            _, g_rad = radial_profile(gamma[t], H)
            _, r2_rad = radial_profile(R2[t], H)
            print(f"  t={t:4d}  alpha_bar={ab_t.item():.4f}  "
                  f"gamma(k=1)={g_rad[1]:.4f}  gamma(k=8)={g_rad[min(8,len(g_rad)-1)]:.4f}  "
                  f"R2(k=1)={r2_rad[1]:.4f}  R2(k=8)={r2_rad[min(8,len(r2_rad)-1)]:.4f}")

    return gamma, R2


# ------------------------------------------------------------------ #
#  Reverse process with D tracking
# ------------------------------------------------------------------ #

def run_reverse_tracking(model, config, device, batch_size=512, chunk_size=64):
    """Run full reverse diffusion, recording D(kh,kw) at every step.

    Returns:
        D_all: (T+1, H, W) float64.  D_all[0]=noise, D_all[T]=final image.
    """
    T = config["T"]
    H = W = config["img_size"]

    betas = torch.linspace(config["beta_1"], config["beta_T"], T,
                           dtype=torch.float64, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    coeff1 = torch.sqrt(1.0 / alphas)
    coeff2 = coeff1 * (1.0 - alphas) / torch.sqrt(1.0 - alphas_bar)

    noise_module = _build_noise_module(config).to(device)

    x_t = noise_module.colorize(
        torch.randn(batch_size, 3, H, W, device=device))

    mode = config.get("mode", "unconditional")
    w_cfg = config.get("w", 0.0)
    labels = _make_labels(config, batch_size, device) if mode == "cfg" else None

    D_all = np.zeros((T + 1, H, W), dtype=np.float64)
    D_all[0] = compute_2d_variance(x_t)

    print(f"Reverse diffusion: T={T}, batch={batch_size}")
    for ts in reversed(range(T)):
        t_tensor = torch.full([batch_size], ts, dtype=torch.long, device=device)

        with torch.no_grad():
            eps_all = _chunked_forward(
                model, x_t, t_tensor, labels, w_cfg, chunk_size)

        mean = (extract(coeff1, t_tensor, x_t.shape) * x_t
                - extract(coeff2, t_tensor, x_t.shape) * eps_all)

        if ts > 0:
            noise = noise_module.colorize(torch.randn_like(x_t))
            x_t = mean + torch.sqrt(betas[ts].float()) * noise
        else:
            x_t = mean

        D_all[T - ts] = compute_2d_variance(x_t)

        if ts % 100 == 0:
            _, d_rad = radial_profile(D_all[T - ts], H)
            print(f"  step {ts:4d}  D(k=1)={d_rad[1]:.4f}")

    return D_all


# ------------------------------------------------------------------ #
#  ODE integration
# ------------------------------------------------------------------ #

def integrate_ode_discrete(gamma, D_init, betas_np, alphas_np, sigma_tilde):
    """Discrete-exact DDPM variance evolution.

    D_{t-1} = [1-(1-alpha_t)*gamma(k,t)]^2 / alpha_t * D_t + beta_t * Sigma
    No noise at t=0.
    """
    T, H, W = gamma.shape
    D = np.zeros((T + 1, H, W), dtype=np.float64)
    D[0] = D_init.copy()

    for n in range(T):
        t = T - 1 - n
        g = gamma[t]
        a_t = alphas_np[t]
        b_t = betas_np[t]

        A_sq = (1.0 - (1.0 - a_t) * g) ** 2 / a_t
        if t > 0:
            D[n + 1] = A_sq * D[n] + b_t * sigma_tilde
        else:
            D[n + 1] = A_sq * D[n]

    return D


def integrate_ode_continuous(gamma, D_init, betas_np, sigma_tilde):
    """Continuous Euler integration of ODE (1.1).

    D_{n+1} = D_n + beta_rev * [(1-2*gamma)*D_n + Sigma]
    No noise at the last step (t=0).
    """
    T, H, W = gamma.shape
    D = np.zeros((T + 1, H, W), dtype=np.float64)
    D[0] = D_init.copy()

    for n in range(T):
        t = T - 1 - n
        g = gamma[t]
        b = betas_np[t]

        if t > 0:
            D[n + 1] = D[n] + b * ((1.0 - 2.0 * g) * D[n] + sigma_tilde)
        else:
            D[n + 1] = D[n] + b * (1.0 - 2.0 * g) * D[n]
        np.clip(D[n + 1], -1e10, 1e10, out=D[n + 1])

    return D


# ------------------------------------------------------------------ #
#  Plotting
# ------------------------------------------------------------------ #

def plot_results(gamma, R2, D_actual, D_discrete, D_continuous,
                 D_data, alphas_bar_np, betas_np, sigma_tilde,
                 img_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T = gamma.shape[0]
    H = W = img_size
    n_axis = np.arange(T + 1)
    k_show = [1, 3, 7, 14]

    # Radially-averaged arrays
    num_k_max = int(make_k_grid(H, W).max()) + 1
    gamma_rad = np.zeros((T, num_k_max))
    D_act_rad = np.zeros((T + 1, num_k_max))
    D_dis_rad = np.zeros((T + 1, num_k_max))
    D_con_rad = np.zeros((T + 1, num_k_max))

    for t in range(T):
        _, gamma_rad[t] = radial_profile(gamma[t], H)
    for n in range(T + 1):
        _, D_act_rad[n] = radial_profile(D_actual[n], H)
        _, D_dis_rad[n] = radial_profile(D_discrete[n], H)
        _, D_con_rad[n] = radial_profile(D_continuous[n], H)

    _, D_data_rad = radial_profile(D_data, H)
    _, sigma_rad = radial_profile(sigma_tilde, H)

    # Gaussian theory for gamma
    # Measured gamma = Sigma_tilde * gamma_true (absorbs noise spectrum),
    # so Gaussian reference must also use the effective gamma.
    gamma_gauss = np.zeros((T, num_k_max))
    for t in range(T):
        ab = alphas_bar_np[t]
        for ki in range(num_k_max):
            denom = ab * D_data_rad[ki] + (1.0 - ab) * sigma_rad[ki]
            gamma_gauss[t, ki] = sigma_rad[ki] / max(denom, 1e-12)

    # ---- Fig 1: gamma heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    k_max_plot = min(16, num_k_max)
    extent = [0, T - 1, 0.5, k_max_plot - 0.5]
    vmax = np.percentile(gamma_rad[:, 1:k_max_plot], 95)

    im0 = axes[0].imshow(gamma_rad[:, 1:k_max_plot].T, aspect='auto',
                          origin='lower', extent=extent,
                          cmap='viridis', vmin=0, vmax=vmax)
    axes[0].set_xlabel('timestep t')
    axes[0].set_ylabel('k')
    axes[0].set_title(r'Measured $\gamma(k, t)$')
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(gamma_gauss[:, 1:k_max_plot].T, aspect='auto',
                          origin='lower', extent=extent,
                          cmap='viridis', vmin=0, vmax=vmax)
    axes[1].set_xlabel('timestep t')
    axes[1].set_ylabel('k')
    axes[1].set_title(r'Gaussian theory $\gamma(k, t)$')
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_gamma_heatmap.png'), dpi=150)
    plt.close(fig)
    print("  fig1_gamma_heatmap.png")

    # ---- Fig 2: gamma(k) at selected timesteps ----
    t_slices = [T - 1, int(0.6 * T), int(0.3 * T), int(0.05 * T)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, t_val in zip(axes.flat, t_slices):
        k_arr = np.arange(1, k_max_plot)
        ax.plot(k_arr, gamma_rad[t_val, 1:k_max_plot], 'bo-', ms=4,
                label='measured')
        ax.plot(k_arr, gamma_gauss[t_val, 1:k_max_plot], 'r^--', ms=4,
                label='Gaussian theory')
        ax.axhline(0.5, color='gray', ls=':', alpha=0.4)
        ab = alphas_bar_np[t_val]
        ax.set_title(f't = {t_val}  ($\\bar{{\\alpha}}$ = {ab:.3f})')
        ax.set_xlabel('k')
        ax.set_ylabel(r'$\gamma$')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle(r'$\gamma(k)$ at selected timesteps', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_gamma_slices.png'), dpi=150)
    plt.close(fig)
    print("  fig2_gamma_slices.png")

    # ---- Fig 3: D(k,n) evolution for selected k ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, k_val in zip(axes.flat, k_show):
        if k_val >= num_k_max:
            continue
        ax.plot(n_axis, D_act_rad[:, k_val], 'b-', lw=1.5, label='actual')
        ax.plot(n_axis, D_dis_rad[:, k_val], 'r--', lw=1.5,
                label='discrete ODE')
        ax.plot(n_axis, D_con_rad[:, k_val], 'g:', lw=1.5,
                label='continuous ODE')
        ax.axhline(D_data_rad[k_val], color='k', ls='--', alpha=0.3,
                    label=f'$D_{{data}}$={D_data_rad[k_val]:.3f}')
        ax.set_xlabel('reverse step n')
        ax.set_ylabel('D(k)')
        ax.set_title(f'k = {k_val}')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle('D(k, n): actual vs ODE predictions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_D_evolution.png'), dpi=150)
    plt.close(fig)
    print("  fig3_D_evolution.png")

    # ---- Fig 4: spectrum snapshots ----
    step_fracs = [0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis
    k_arr = np.arange(1, k_max_plot)
    for i, frac in enumerate(step_fracs):
        n_val = int(frac * T)
        n_val = min(n_val, T)
        color = cmap(i / (len(step_fracs) - 1))
        ax.plot(k_arr, D_act_rad[n_val, 1:k_max_plot], 'o-', color=color,
                ms=4, label=f'actual n={n_val}')
        ax.plot(k_arr, D_dis_rad[n_val, 1:k_max_plot], 's--', color=color,
                ms=3, alpha=0.6)
    ax.plot(k_arr, D_data_rad[1:k_max_plot], 'k--', lw=2, label='$D_{data}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('k', fontsize=20)
    ax.set_ylabel('D(k)', fontsize=20)
    ax.set_title('Spectrum snapshots (dashed = discrete ODE)', fontsize=20)
    ax.legend(fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_spectrum_snapshots.png'), dpi=150)
    plt.close(fig)
    print("  fig4_spectrum_snapshots.png")

    # ---- Fig 5: relative error ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, k_val in zip(axes.flat, k_show):
        if k_val >= num_k_max:
            continue
        d_act = D_act_rad[:, k_val]
        d_dis = D_dis_rad[:, k_val]
        d_con = D_con_rad[:, k_val]
        safe = np.maximum(np.abs(d_act), 1e-12)
        err_dis = np.abs(d_dis - d_act) / safe
        err_con = np.abs(d_con - d_act) / safe
        ax.semilogy(n_axis, err_dis, 'r-', lw=1, label='discrete ODE')
        ax.semilogy(n_axis, err_con, 'g-', lw=1, label='continuous ODE')
        ax.set_xlabel('reverse step n')
        ax.set_ylabel('|D_pred - D_actual| / |D_actual|')
        ax.set_title(f'k = {k_val}')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-4, 10)
    fig.suptitle('Relative error of ODE predictions', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig5_relative_error.png'), dpi=150)
    plt.close(fig)
    print("  fig5_relative_error.png")

    # ---- Fig 6: gamma vs t for selected k ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    t_axis = np.arange(T)
    for ax, k_val in zip(axes.flat, k_show):
        if k_val >= num_k_max:
            continue
        ax.plot(t_axis, gamma_rad[:, k_val], 'b-', lw=1, label='measured')
        ax.plot(t_axis, gamma_gauss[:, k_val], 'r--', lw=1.5,
                label='Gaussian theory')
        ax.axhline(0.5, color='gray', ls=':', alpha=0.4)
        ax.set_xlabel('timestep t')
        ax.set_ylabel(r'$\gamma$')
        ax.set_title(f'k = {k_val}')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    fig.suptitle(r'$\gamma(k, t)$: measured vs Gaussian theory', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig6_gamma_vs_t.png'), dpi=150)
    plt.close(fig)
    print("  fig6_gamma_vs_t.png")

    # ---- Fig 7: R² heatmap and R²(t) curves ----
    R2_rad = np.zeros((T, num_k_max))
    for t in range(T):
        _, R2_rad[t] = radial_profile(R2[t], H)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    im = axes[0].imshow(R2_rad[:, 1:k_max_plot].T, aspect='auto',
                         origin='lower', extent=extent,
                         cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xlabel('timestep t')
    axes[0].set_ylabel('k')
    axes[0].set_title(r'$R^2(k, t)$ — goodness of linear fit')
    fig.colorbar(im, ax=axes[0])

    for k_val in k_show:
        if k_val < num_k_max:
            axes[1].plot(t_axis, R2_rad[:, k_val], lw=1.5, label=f'k={k_val}')
    axes[1].set_xlabel('timestep t')
    axes[1].set_ylabel(r'$R^2$')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title(r'$R^2(t)$ for selected $k$')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig7_R2.png'), dpi=150)
    plt.close(fig)
    print("  fig7_R2.png")

    # ---- Fig 8: spectral order parameter phi(t) = P_L / P_H ----
    k_cut = H // 4
    phi_act = compute_phi(D_actual, H, k_cut)
    phi_dis = compute_phi(D_discrete, H, k_cut)
    phi_data = compute_phi(D_data[np.newaxis], H, k_cut)[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(n_axis, phi_act, 'b-', lw=1.5, label='actual')
    axes[0].plot(n_axis, phi_dis, 'r--', lw=1.5, label='discrete ODE')
    axes[0].axhline(phi_data, color='k', ls=':', alpha=0.5,
                     label=f'$\\varphi_{{data}}$ = {phi_data:.2f}')
    axes[0].set_xlabel('reverse step n')
    axes[0].set_ylabel(r'$\varphi = P_L / P_H$')
    axes[0].set_title(f'Spectral order parameter ($k_c = {k_cut}$)')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(n_axis, phi_act / max(phi_data, 1e-12), 'b-', lw=1.5)
    axes[1].axhline(1.0, color='k', ls=':', alpha=0.5)
    axes[1].set_xlabel('reverse step n')
    axes[1].set_ylabel(r'$\varphi(n)\,/\,\varphi_{data}$')
    axes[1].set_title('Normalized spectral order parameter')
    axes[1].set_ylim(0, max(2.0, 1.2 * (phi_act / max(phi_data, 1e-12)).max()))
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(r'Prediction 2: $\varphi(t) = P_L(t)\,/\,P_H(t)$ trajectory',
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig8_phi_trajectory.png'), dpi=150)
    plt.close(fig)
    print("  fig8_phi_trajectory.png")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Measure gamma(k,t) and validate ODE (1.1)")
    parser.add_argument("--noise_type", type=str, default="white")
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--gamma_batch", type=int, default=512)
    parser.add_argument("--reverse_batch", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--channel_mult", type=int, nargs="+",
                        default=[1, 2, 2, 2])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attn", type=int, nargs="+", default=[1])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.028)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="cfg")
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--w", type=float, default=1.8)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--colored_method", type=str, default="cholesky")
    parser.add_argument("--ckpt_dir", type=str, default="./Checkpoints_T500/")
    parser.add_argument("--ckpt", type=str, default="ckpt_199_.pt")
    parser.add_argument("--out_dir", type=str, default="./GammaODE_Results/")
    parser.add_argument("--data_root", type=str, default="./CIFAR10/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load_data", type=str, default=None,
                        help="Load saved .npz, skip measurement, re-plot only")
    args = parser.parse_args()

    config = vars(args)
    device = torch.device(config["device"])
    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    T = config["T"]
    H = W = config["img_size"]

    # Schedule arrays (float64 for precision)
    betas_np = np.linspace(config["beta_1"], config["beta_T"], T)
    alphas_np = 1.0 - betas_np
    alphas_bar_np = np.cumprod(alphas_np)

    # Noise spectrum: sigma_tilde(k) = 1 for white noise
    if config["noise_type"] == "white":
        sigma_tilde = np.ones((H, W), dtype=np.float64)
    else:
        print("Measuring colored noise spectrum...")
        plot_device = torch.device("cpu") if config["load_data"] else device
        noise_module = _build_noise_module(config).to(plot_device)
        white = torch.randn(4096, 3, H, W, device=plot_device)
        with torch.no_grad():
            colored = noise_module.colorize(white)
        sigma_tilde = compute_2d_variance(colored)
        del white, colored, noise_module
        if plot_device.type == "cuda":
            torch.cuda.empty_cache()

    print("=" * 60)
    print("  GammaODE — Measure gamma(k,t) & Validate ODE (1.1)")
    print("=" * 60)

    if config["load_data"] is not None:
        print(f"\nLoading saved data from {config['load_data']}...")
        data = np.load(config["load_data"])
        gamma = data["gamma"]
        R2 = data["R2"] if "R2" in data else np.zeros_like(gamma)
        D_actual = data["D_actual"]
        D_data = data["D_data"]
    else:
        # --- D_data from CIFAR-10 ---
        print("\nComputing CIFAR-10 data spectrum...")
        D_data = compute_cifar10_spectrum_2d(config["data_root"])

        # --- Load model ---
        ckpt_path = os.path.join(config["ckpt_dir"], config["ckpt"])
        print(f"\nLoading model from {ckpt_path}...")
        net_model = _build_model(config, device)
        net_model.dropout = 0.0
        net_model.load_state_dict(
            torch.load(ckpt_path, map_location=device))
        net_model.eval()
        print("  Model loaded.")

        # --- Measure gamma ---
        print("\nMeasuring gamma(k,t)...")
        x0, _ = load_data_batch(config["data_root"],
                                config["gamma_batch"], device)
        gamma, R2 = measure_gamma(model=net_model, config=config, device=device,
                              x0=x0, chunk_size=config["chunk_size"])
        del x0
        torch.cuda.empty_cache()

        # --- Run reverse process ---
        print("\nRunning reverse process with D tracking...")
        with torch.no_grad():
            D_actual = run_reverse_tracking(
                net_model, config, device,
                batch_size=config["reverse_batch"],
                chunk_size=config["chunk_size"])

        del net_model
        torch.cuda.empty_cache()

        # --- Save raw data ---
        save_path = os.path.join(out_dir, "gamma_ode_data.npz")
        np.savez(save_path, gamma=gamma, R2=R2, D_actual=D_actual, D_data=D_data,
                 sigma_tilde=sigma_tilde,
                 betas=betas_np, alphas=alphas_np, alphas_bar=alphas_bar_np)
        print(f"\nRaw data saved to {save_path}")

    # --- Integrate ODEs ---
    print("\nIntegrating discrete ODE...")
    D_discrete = integrate_ode_discrete(
        gamma, D_actual[0], betas_np, alphas_np, sigma_tilde)

    print("Integrating continuous ODE...")
    D_continuous = integrate_ode_continuous(
        gamma, D_actual[0], betas_np, sigma_tilde)

    # --- Print summary for selected k ---
    print(f"\n{'k':>3s}  {'D_data':>9s}  {'D_act_final':>11s}  "
          f"{'D_dis_final':>11s}  {'rel_err':>9s}  {'R2_mean':>8s}")
    print("-" * 65)
    for k_val in [1, 3, 7, 14]:
        _, d_data_r = radial_profile(D_data, H)
        _, d_act_r = radial_profile(D_actual[T], H)
        _, d_dis_r = radial_profile(D_discrete[T], H)
        _, r2_r = radial_profile(R2.mean(axis=0), H)
        if k_val < len(d_data_r):
            err = abs(d_dis_r[k_val] - d_act_r[k_val]) / max(abs(d_act_r[k_val]), 1e-12)
            print(f"  {k_val:2d}  {d_data_r[k_val]:9.4f}  {d_act_r[k_val]:11.4f}  "
                  f"{d_dis_r[k_val]:11.4f}  {err:9.2%}  {r2_r[k_val]:8.4f}")

    # --- Plot ---
    print(f"\nGenerating plots in {out_dir}...")
    plot_results(gamma, R2, D_actual, D_discrete, D_continuous,
                 D_data, alphas_bar_np, betas_np, sigma_tilde,
                 H, out_dir)

    print("\nDone.")


def load_data_batch(data_root, n, device):
    """Load n CIFAR-10 training images as (n, 3, 32, 32) in [-1,1]."""
    dataset = CIFAR10(
        root=data_root, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=n, shuffle=True, num_workers=4)
    images, labels = next(iter(loader))
    return images.to(device), labels.to(device)


if __name__ == "__main__":
    main()
