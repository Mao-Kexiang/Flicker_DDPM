import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T_values = [100, 150, 200, 500]
white_paths = [f'Checkpoints_T{T}/fid_data.npz' for T in T_values]
colored_paths = [f'Checkpoints_colored_eta0.2_T{T}/fid_data.npz' for T in T_values]

white_fid = []
colored_fid = []
for wp, cp in zip(white_paths, colored_paths):
    w_data = np.load(wp)
    c_data = np.load(cp)
    white_fid.append(w_data['fid'][-1])
    colored_fid.append(c_data['fid'][-1])

T_labels = [f'T={T}' for T in T_values]
x = np.arange(len(T_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, white_fid, width, label='White Noise (baseline)',
               color='#6baed6', edgecolor='black', linewidth=1.0)
bars2 = ax.bar(x + width/2, colored_fid, width, label='Colored Noise η=0.2 (ours)',
               color="#ef6565", edgecolor='black', linewidth=1.0)

for bar, val in zip(bars1, white_fid):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
for bar, val in zip(bars2, colored_fid):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_xlabel('Diffusion Steps T', fontsize=20)
ax.set_ylabel('FID Score (lower is better)', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(T_labels, fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, max(white_fid) * 1.15)
ax.legend(fontsize=20, loc='upper right')
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig('fid_comparison_final.png', dpi=150)
print("Saved to fid_comparison_final.png")
