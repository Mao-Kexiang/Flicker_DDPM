import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cases = [
    ("Checkpoints_T500/fid_data.npz",              "White, T=500"),
    ("Checkpoints_T150/fid_data.npz",              "White, T=150"),
    ("Checkpoints_colored_eta0.2_T500/fid_data.npz","Colored (η=0.2), T=500"),
    ("Checkpoints_colored_eta0.2_T150/fid_data.npz","Colored (η=0.2), T=150"),
]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

fig, ax = plt.subplots(figsize=(10, 6))

for (path, label), color, marker in zip(cases, colors, markers):
    data = np.load(path)
    epochs = data['epoch']
    fid = data['fid']
    ax.plot(epochs, fid, marker=marker, color=color, label=label,
            markersize=5, linewidth=1.5, alpha=0.85)

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('FID Score', fontsize=13)
ax.set_title('FID Evolution: White vs Colored Noise, T=150 vs T=500', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig('fid_combined.png', dpi=150)
print("Saved to fid_combined.png")
