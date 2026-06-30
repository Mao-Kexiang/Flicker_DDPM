import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import torch


class DiffusionAnalyzer:
    def __init__(self, out_dir="./Analysis/"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.history = {}

    def _get_radial_power_spectrum(self, img_channel):
        h, w = img_channel.shape
        f_coef = np.fft.fft2(img_channel)
        f_shift = np.fft.fftshift(f_coef)
        power_spectrum = np.abs(f_shift)**2

        y, x = np.indices(power_spectrum.shape)
        center = (h // 2, w // 2)
        r = np.sqrt((x - center[0])**2 + (y - center[1])**2).astype(int)

        tbin = np.bincount(r.ravel(), power_spectrum.ravel())
        nr = np.bincount(r.ravel())
        radial_profile = tbin / (nr + 1e-10)

        k_vals = np.arange(len(radial_profile))
        mask = (k_vals > 0) & (k_vals < min(h, w) // 2)
        log_k = np.log10(k_vals[mask])
        log_s = np.log10(radial_profile[mask] + 1e-10)
        return log_k, log_s

    def plot_power_spectrum_grid(self, imgs_tensor, step):
        if torch.is_tensor(imgs_tensor):
            imgs = imgs_tensor.detach().cpu().numpy()
        else:
            imgs = imgs_tensor

        batch_size = imgs.shape[0]

        if not self.history:
            for i in range(batch_size):
                self.history[i] = {'t': [], 'alpha': [], 'r2': []}

        ncols = min(batch_size, 10)
        nrows = (batch_size + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
        if batch_size == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i in range(batch_size):
            ax = axes[i]
            batch_alphas, batch_r2s = [], []

            for c, color in enumerate(['red', 'green', 'blue']):
                log_k, log_s = self._get_radial_power_spectrum(imgs[i, c])
                slope, intercept, r_val, p_val, std_err = stats.linregress(log_k, log_s)

                batch_alphas.append(-slope)
                batch_r2s.append(r_val**2)

                ax.scatter(log_k, log_s, s=5, color=color, alpha=0.2)
                ax.plot(log_k, slope * log_k + intercept, color=color, alpha=0.8)

            avg_alpha = np.mean(batch_alphas)
            avg_r2 = np.mean(batch_r2s)
            self.history[i]['t'].append(step)
            self.history[i]['alpha'].append(avg_alpha)
            self.history[i]['r2'].append(avg_r2)

            ax.set_title(f"ID:{i} \u03b1={avg_alpha:.2f}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        for j in range(batch_size, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"Power Spectrum Fit at Step {step}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.out_dir, f"spectrum_step_{step:04d}.png"))
        plt.close(fig)

    def plot_evolution_charts(self):
        if not self.history:
            print("No data recorded. Skipping evolution charts.")
            return

        batch_size = len(self.history)
        ncols = min(batch_size, 10)
        nrows = (batch_size + ncols - 1) // ncols

        for metric in ['alpha', 'r2']:
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
            if batch_size == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for i in range(batch_size):
                ax = axes[i]
                t_axis = self.history[i]['t']
                val_axis = self.history[i][metric]

                ax.plot(t_axis, val_axis, marker='o', markersize=2, linestyle='-', color='#1f77b4')
                ax.invert_xaxis()

                if metric == 'alpha':
                    ax.set_ylim(0, 3)
                elif metric == 'r2':
                    ax.set_ylim(0, 1)

                ax.set_title(f"Img {i}", fontsize=8)
                ax.grid(True, alpha=0.3, linestyle='--')

                if i % ncols != 0:
                    ax.set_yticklabels([])
                if i < (nrows - 1) * ncols:
                    ax.set_xticklabels([])

            for j in range(batch_size, len(axes)):
                axes[j].set_visible(False)

            max_t = max(self.history[0]['t']) if self.history[0]['t'] else 0
            plt.suptitle(f"Evolution of {metric.upper()} during Denoising (T={max_t} to 0)", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            save_path = os.path.join(self.out_dir, f"evolution_{metric}.png")
            plt.savefig(save_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {save_path}")
