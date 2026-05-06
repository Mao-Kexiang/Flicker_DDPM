"""
plot_prediction2.py — Verify Theory Prediction 2: phi(t) = P_L / P_H trajectory.

Loads pre-computed GammaODE results for colored and white noise,
computes the spectral order parameter phi(t), and generates comparison plots.

Theory predicts:
  - White:   phi from ~1 (flat spectrum) → phi_data >> 1 (S-shaped)
  - Colored: phi ≈ phi_data throughout (constant)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def make_k_grid(H, W):
    kh = np.arange(H, dtype=np.float64)
    kw = np.arange(W, dtype=np.float64)
    kh_freq = np.where(kh <= H // 2, kh, kh - H)
    kw_freq = np.where(kw <= W // 2, kw, kw - W)
    return np.sqrt(kh_freq[:, None] ** 2 + kw_freq[None, :] ** 2)


def compute_phi(D_2d, H, k_cut):
    kr = make_k_grid(H, H)
    low = (kr > 0) & (kr <= k_cut)
    high = kr > k_cut
    P_L = D_2d[..., low].sum(axis=-1)
    P_H = D_2d[..., high].sum(axis=-1)
    return P_L / np.maximum(P_H, 1e-12)


def main():
    H = 32
    k_cut = H // 4  # k_c = 8

    colored = np.load('GammaODE_Results/gamma_ode_data.npz')
    white = np.load('GammaODE_Results_white/gamma_ode_data.npz')

    D_data = colored['D_data']
    phi_data = compute_phi(D_data[np.newaxis], H, k_cut)[0]

    phi_c = compute_phi(colored['D_actual'], H, k_cut)
    phi_w = compute_phi(white['D_actual'], H, k_cut)

    T = colored['D_actual'].shape[0] - 1
    n_axis = np.arange(T + 1)

    print(f"phi_data = {phi_data:.2f}")
    print(f"Colored: phi(0) = {phi_c[0]:.2f}, phi(T) = {phi_c[-1]:.2f}, "
          f"range = [{phi_c.min():.2f}, {phi_c.max():.2f}]")
    print(f"White:   phi(0) = {phi_w[0]:.2f}, phi(T) = {phi_w[-1]:.2f}, "
          f"range = [{phi_w.min():.2f}, {phi_w.max():.2f}]")
    print(f"Colored phi std/mean = {phi_c.std() / phi_c.mean():.4f}")
    print(f"White   phi std/mean = {phi_w.std() / phi_w.mean():.4f}")

    # ---- Main comparison plot ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: raw phi(n)
    axes[0].plot(n_axis, phi_w, 'b-', lw=1.5, label='White noise')
    axes[0].plot(n_axis, phi_c, 'r-', lw=1.5, label='Colored noise ($\\eta=0.2$)')
    axes[0].axhline(phi_data, color='k', ls=':', alpha=0.5,
                     label=f'$\\varphi_{{data}}$ = {phi_data:.1f}')
    axes[0].set_xlabel('reverse step n', fontsize=12)
    axes[0].set_ylabel(r'$\varphi = P_L\,/\,P_H$', fontsize=12)
    axes[0].set_title(r'(a) $\varphi(n)$ trajectory', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: normalized phi/phi_data
    axes[1].plot(n_axis, phi_w / phi_data, 'b-', lw=1.5, label='White')
    axes[1].plot(n_axis, phi_c / phi_data, 'r-', lw=1.5, label='Colored')
    axes[1].axhline(1.0, color='k', ls=':', alpha=0.5)
    axes[1].fill_between(n_axis, 0.9, 1.1, color='green', alpha=0.08)
    axes[1].set_xlabel('reverse step n', fontsize=12)
    axes[1].set_ylabel(r'$\varphi(n)\,/\,\varphi_{data}$', fontsize=12)
    axes[1].set_title('(b) Normalized (green = ±10% band)', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: P_L and P_H separately
    kr = make_k_grid(H, H)
    low = (kr > 0) & (kr <= k_cut)
    high = kr > k_cut

    PL_w = white['D_actual'][:, low].sum(axis=-1)
    PH_w = white['D_actual'][:, high].sum(axis=-1)
    PL_c = colored['D_actual'][:, low].sum(axis=-1)
    PH_c = colored['D_actual'][:, high].sum(axis=-1)

    PL_w /= PL_w[0]; PH_w /= PH_w[0]
    PL_c /= PL_c[0]; PH_c /= PH_c[0]

    axes[2].plot(n_axis, PL_w, 'b-', lw=1.5, label=r'White $P_L$')
    axes[2].plot(n_axis, PH_w, 'b--', lw=1.5, label=r'White $P_H$')
    axes[2].plot(n_axis, PL_c, 'r-', lw=1.5, label=r'Colored $P_L$')
    axes[2].plot(n_axis, PH_c, 'r--', lw=1.5, label=r'Colored $P_H$')
    axes[2].set_xlabel('reverse step n', fontsize=12)
    axes[2].set_ylabel('Normalized total power', fontsize=12)
    axes[2].set_title(r'(c) $P_L(n)$ and $P_H(n)$ (normalized to $n=0$)', fontsize=13)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(r'Prediction 2 Verification: $\varphi(t) = P_L / P_H$ '
                 f'($k_c = {k_cut}$)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig('fig_prediction2_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved: fig_prediction2_comparison.png")

    # ---- Also save individual fig8 into each results dir ----
    for tag, D_act, out_dir in [
        ('Colored', colored['D_actual'], 'GammaODE_Results'),
        ('White', white['D_actual'], 'GammaODE_Results_white'),
    ]:
        phi = compute_phi(D_act, H, k_cut)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(n_axis, phi, 'b-', lw=1.5, label='actual')
        ax.axhline(phi_data, color='k', ls=':', alpha=0.5,
                   label=f'$\\varphi_{{data}}$ = {phi_data:.1f}')
        ax.set_xlabel('reverse step n', fontsize=12)
        ax.set_ylabel(r'$\varphi = P_L / P_H$', fontsize=12)
        ax.set_title(f'{tag} noise: spectral order parameter ($k_c = {k_cut}$)',
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{out_dir}/fig8_phi_trajectory.png', dpi=150)
        plt.close(fig)
        print(f"Saved: {out_dir}/fig8_phi_trajectory.png")


if __name__ == '__main__':
    main()
