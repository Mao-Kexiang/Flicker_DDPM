#!/usr/bin/env python3
"""
Denoiser response J(k,t) analysis.

J(k,t) = ∂ε̃_θ(k)/∂x̃(k)  —  Jacobian diagonal in Fourier space.

Relation to score linearization coefficient γ:
    γ(k,t) = J(k,t) / √(1 - ᾱ_t)

Using J directly avoids the 1/√(1-ᾱ) divergence at t→0.

Theoretical predictions (unitary FFT, Σ̃=1 for white noise):
    Wiener:      J_W(k,t)  = √(1-ᾱ) / [ᾱ·D_inf(k) + (1-ᾱ)]
    Equilibrium: J_eq(k,t) = √(1-ᾱ) · [1/2 + 1/(2·D_inf(k))]
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, 'TheoryResults')
HW = 32 * 32

# ── Load saved data ──────────────────────────────────────────────

jd = np.load(os.path.join(OUT_DIR, 'jacobian_gamma_data.npz'))
gamma_kt = jd['gamma_kt']        # (n_t, n_k)
ab_arr   = jd['alphas_bar']       # (n_t,)
k_bins   = jd['k_bins']           # (n_k,)
t_arr    = jd['t_arr']            # (n_t,)

td = np.load(os.path.join(OUT_DIR, 'theory_validation_data.npz'),
             allow_pickle=True)
D_inf_unnorm = td['D_inf']        # (n_k_full,) unnormalized convention
D_inf = D_inf_unnorm / HW         # unitary convention

# ── Convert γ → J ────────────────────────────────────────────────

sqrt_1mab = np.sqrt(1.0 - ab_arr)                     # (n_t,)
J_kt = gamma_kt * sqrt_1mab[:, None]                  # (n_t, n_k)

# ── Theoretical J ────────────────────────────────────────────────

n_t, n_k = J_kt.shape
J_wiener = np.zeros_like(J_kt)
J_eq     = np.zeros_like(J_kt)

for ti in range(n_t):
    ab = ab_arr[ti]
    s1ma = sqrt_1mab[ti]
    for ki, k in enumerate(k_bins):
        if k < len(D_inf) and D_inf[k] > 0:
            D_kt_val = ab * D_inf[k] + (1.0 - ab)
            J_wiener[ti, ki] = s1ma / D_kt_val
            J_eq[ti, ki] = s1ma * (0.5 + 1.0 / (2.0 * D_inf[k]))

# ── Delta sweep conversion ──────────────────────────────────────

beta_1, beta_T, T = 0.0001, 0.028, 500
betas = np.linspace(beta_1, beta_T, T)
ab_full = np.cumprod(1.0 - betas)
t_probe = 250
ab_probe = ab_full[t_probe]
s1ma_probe = np.sqrt(1.0 - ab_probe)

deltas = jd['delta_sweep_deltas']
sweep_J = {}
for kp in [1, 3, 7, 14]:
    key = f'delta_sweep_k{kp}'
    if key in jd:
        sweep_J[kp] = jd[key] * s1ma_probe   # γ·√(1-ᾱ) → J

# ── Summary table ────────────────────────────────────────────────

t_mask = ab_arr < 0.9
n_valid = t_mask.sum()

print("=" * 70)
print("  Denoiser Response J(k,t) = ∂ε̃_θ(k)/∂x̃(k)")
print("=" * 70)
print(f"  {n_t} t-points, {n_k} k-bins, {n_valid} points with ᾱ < 0.9")
print()
print(f"{'k':>3s} {'J_mean':>9s} {'J_std':>8s} {'J_W_mean':>9s} "
      f"{'ratio':>7s} {'J_eq_mean':>10s}")
print("-" * 55)

for ki, k in enumerate(k_bins[:15]):
    if k < len(D_inf) and D_inf[k] > 0:
        jm = np.mean(J_kt[t_mask, ki])
        js = np.std(J_kt[t_mask, ki])
        jw = np.mean(J_wiener[t_mask, ki])
        je = np.mean(J_eq[t_mask, ki])
        r = jm / jw if jw > 1e-10 else 0
        print(f"{k:3d} {jm:9.4f} {js:8.4f} {jw:9.4f} {r:7.3f} {je:10.4f}")

# ── Detailed per-t table ─────────────────────────────────────────

print()
print("Detailed J(k,t) at selected (k, t):")
print(f"{'t':>5s} {'abar':>7s} {'sqrt1ma':>8s}", end='')
for k in [1, 3, 7, 14]:
    print(f" {'J_m('+str(k)+')':>8s} {'J_W':>7s} {'rat':>5s}", end='')
print()
print("-" * 95)
for ti in range(0, n_t, 3):
    ab = ab_arr[ti]
    s = sqrt_1mab[ti]
    print(f"{t_arr[ti]:5d} {ab:7.4f} {s:8.4f}", end='')
    for k in [1, 3, 7, 14]:
        if k in k_bins:
            ki = list(k_bins).index(k)
            jm = J_kt[ti, ki]
            jw = J_wiener[ti, ki]
            r = jm / jw if jw > 1e-10 else 0
            print(f" {jm:8.4f} {jw:7.4f} {r:5.2f}", end='')
    print()

# ── Fig J1: J(k) at fixed t slices ──────────────────────────────

t_slices = [50, 150, 300, 450]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, t_target in zip(axes.flat, t_slices):
    ti = np.argmin(np.abs(t_arr - t_target))
    t_act = t_arr[ti]
    ab = ab_arr[ti]
    kp = k_bins[:15]
    ax.plot(kp, J_kt[ti, :15], 'bo-', markersize=5, label='measured')
    ax.plot(kp, J_wiener[ti, :15], 'r^--', markersize=5, label='Wiener')
    ax.plot(kp, J_eq[ti, :15], 'g:', linewidth=1.5, label='equilibrium')
    ax.set_xlabel('k')
    ax.set_ylabel('$J(k)$')
    ax.set_title(f't = {t_act}  ($\\bar{{\\alpha}}$ = {ab:.3f})')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
fig.suptitle(r'$J(k,t) = \partial\tilde{\varepsilon}_\theta(k)'
             r'/\partial\tilde{x}(k)$ at fixed $t$', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'fig_J1_vs_k.png'), dpi=150)
plt.close(fig)
print(f"\nSaved fig_J1_vs_k.png")

# ── Fig J2: heatmaps ────────────────────────────────────────────

vmax = np.percentile(J_kt, 97)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

extent = [t_arr[0], t_arr[-1], k_bins[0] - 0.5, k_bins[-1] + 0.5]
im0 = axes[0].imshow(J_kt.T, aspect='auto', origin='lower',
                      extent=extent, cmap='viridis', vmin=0, vmax=vmax)
axes[0].set_xlabel('timestep t')
axes[0].set_ylabel('k')
axes[0].set_title(r'Measured $J(k,t)$')
fig.colorbar(im0, ax=axes[0], label='$J$')

im1 = axes[1].imshow(J_wiener.T, aspect='auto', origin='lower',
                      extent=extent, cmap='viridis', vmin=0, vmax=vmax)
axes[1].set_xlabel('timestep t')
axes[1].set_ylabel('k')
axes[1].set_title(r'Wiener $J_W(k,t)$')
fig.colorbar(im1, ax=axes[1], label='$J$')

fig.suptitle(r'$J(k,t)$: Measured vs Wiener filter', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, 'fig_J2_heatmap.png'), dpi=150)
plt.close(fig)
print("Saved fig_J2_heatmap.png")

# ── Fig J3: J(t) for selected k ─────────────────────────────────

k_show = [1, 3, 7, 14]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, k_target in zip(axes.flat, k_show):
    if k_target in k_bins:
        ki = list(k_bins).index(k_target)
        ax.plot(t_arr, J_kt[:, ki], 'b.-', markersize=3, label='measured')
        ax.plot(t_arr, J_wiener[:, ki], 'r--', linewidth=2, label='Wiener')
        ax.plot(t_arr, J_eq[:, ki], 'g:', linewidth=1.5, label='equilibrium')
        ax.set_xlabel('timestep t')
        ax.set_ylabel('$J$')
        ax.set_title(f'k = {k_target}  ($D_{{\\infty}}$ = {D_inf[k_target]:.4f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
fig.suptitle(r'$J(k,t)$: Measured vs Theory', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT_DIR, 'fig_J3_vs_t.png'), dpi=150)
plt.close(fig)
print("Saved fig_J3_vs_t.png")

# ── Fig J4: ratio heatmap ───────────────────────────────────────

ratio_kt = np.where(J_wiener > 1e-6, J_kt / J_wiener, np.nan)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(ratio_kt.T, aspect='auto', origin='lower',
               extent=extent, cmap='RdBu_r', vmin=0.5, vmax=2.0)
ax.set_xlabel('timestep t')
ax.set_ylabel('k')
ax.set_title(r'$J_{\mathrm{meas}} / J_{\mathrm{Wiener}}$')
fig.colorbar(im, ax=ax, label='ratio')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig_J4_ratio.png'), dpi=150)
plt.close(fig)
print("Saved fig_J4_ratio.png")

# ── Fig J5: delta sweep ─────────────────────────────────────────

if sweep_J:
    fig, ax = plt.subplots(figsize=(8, 5))
    for kp, vals in sorted(sweep_J.items()):
        ax.semilogx(deltas, vals, 'o-', markersize=4, label=f'k={kp}')
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel('$J$')
    ax.set_title(f'Delta sweep at t={t_probe} '
                 f'($\\bar{{\\alpha}}$={ab_probe:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_J5_delta_sweep.png'), dpi=150)
    plt.close(fig)
    print("Saved fig_J5_delta_sweep.png")

# ── Save ─────────────────────────────────────────────────────────

np.savez(os.path.join(OUT_DIR, 'denoiser_response_J.npz'),
         J_kt=J_kt, J_wiener=J_wiener, J_eq=J_eq,
         k_bins=k_bins, t_arr=t_arr, alphas_bar=ab_arr,
         D_inf_unitary=D_inf[:len(k_bins)])
print(f"\nSaved denoiser_response_J.npz")
print("\nDone!")
