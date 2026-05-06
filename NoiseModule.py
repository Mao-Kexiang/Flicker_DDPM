import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NoiseModule(nn.Module):
    """
    Unified noise abstraction supporting white noise, Cholesky-based colored noise,
    and FFT-based colored noise.

    Usage:
        nm = NoiseModule("colored", img_size=32, eta=0.2, method="cholesky")
        colored = nm.colorize(torch.randn(B, C, H, W))
        whitened = nm.whiten(residue)
    """

    def __init__(self, noise_type="white", img_size=32, eta=0.2, method="cholesky"):
        super().__init__()
        assert noise_type in ("white", "colored")
        assert method in ("cholesky", "fft")

        self.noise_type = noise_type
        self.img_size = img_size
        self.eta = eta
        self.method = method

        if noise_type == "colored":
            if method == "cholesky":
                self._init_cholesky(img_size, eta)
            else:
                self._init_fft(img_size, eta)

    # ------------------------------------------------------------------ #
    #  Cholesky initialization (from DDPM_power/Diffusion/Diffusion.py)
    # ------------------------------------------------------------------ #
    def _init_cholesky(self, img_size, eta):
        n_pixels = img_size ** 2
        coords = [(r, c) for r in range(img_size) for c in range(img_size)]
        sigma = np.zeros((n_pixels, n_pixels))
        for i in range(n_pixels):
            r1, c1 = coords[i]
            for j in range(i, n_pixels):
                r2, c2 = coords[j]
                d_ij = abs(r1 - r2) + abs(c1 - c2)
                val = (d_ij + 1) ** (-eta)
                sigma[i, j] = val
                sigma[j, i] = val

        reg_sigma = sigma + np.eye(n_pixels) * 1e-7
        L = np.linalg.cholesky(reg_sigma)
        L_inv = np.linalg.inv(L)

        self.register_buffer('L', torch.from_numpy(L).float())
        self.register_buffer('L_inv', torch.from_numpy(L_inv).float())
        self.n_pixels = n_pixels

    # ------------------------------------------------------------------ #
    #  FFT initialization (from DDPM_power/noise_try.py)
    # ------------------------------------------------------------------ #
    def _init_fft(self, img_size, eta):
        N = img_size
        M = 2 * N

        c = np.arange(M)
        dist_1d = np.minimum(c, M - c)
        dx = dist_1d[:, np.newaxis]
        dy = dist_1d[np.newaxis, :]
        dist_l1 = dx + dy

        kernel_2d = (dist_l1 + 1) ** (-eta)
        lambda_spectrum = np.fft.fft2(kernel_2d).real

        lambda_safe = np.maximum(lambda_spectrum, 0)
        Q_freq = np.sqrt(lambda_safe)
        Q_freq_inv = np.where(Q_freq > 1e-8, 1.0 / Q_freq, 0.0)

        self.register_buffer('Q_freq', torch.from_numpy(Q_freq).float())
        self.register_buffer('Q_freq_inv', torch.from_numpy(Q_freq_inv).float())
        self.fft_M = M
        self.fft_N = N

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def colorize(self, white_noise):
        """Transform white noise [B,C,H,W] -> colored noise [B,C,H,W]."""
        if self.noise_type == "white":
            return white_noise
        if self.method == "cholesky":
            return self._colorize_cholesky(white_noise)
        return self._colorize_fft(white_noise)

    def whiten(self, colored_residue):
        """Transform colored residue [B,C,H,W] -> whitened residue [B,C,H,W]."""
        if self.noise_type == "white":
            return colored_residue
        if self.method == "cholesky":
            return self._whiten_cholesky(colored_residue)
        return self._whiten_fft(colored_residue)

    # ------------------------------------------------------------------ #
    #  Cholesky internals
    # ------------------------------------------------------------------ #
    def _colorize_cholesky(self, white_noise):
        B, C, H, W = white_noise.shape
        flat = white_noise.view(-1, self.n_pixels)
        colored_flat = torch.matmul(flat, self.L.t())
        return colored_flat.view(B, C, H, W)

    def _whiten_cholesky(self, colored_residue):
        B, C, H, W = colored_residue.shape
        flat = colored_residue.view(-1, self.n_pixels)
        whitened_flat = torch.matmul(flat, self.L_inv.t())
        return whitened_flat.view(B, C, H, W)

    # ------------------------------------------------------------------ #
    #  FFT internals
    # ------------------------------------------------------------------ #
    def _colorize_fft(self, white_noise):
        B, C, H, W = white_noise.shape
        M = self.fft_M
        white_2n = F.pad(white_noise, [0, M - W, 0, M - H])
        freq = torch.fft.fft2(white_2n)
        colored_freq = freq * self.Q_freq.to(freq.dtype)
        colored_2n = torch.fft.ifft2(colored_freq).real
        return colored_2n[:, :, :H, :W]

    def _whiten_fft(self, colored_residue):
        B, C, H, W = colored_residue.shape
        M = self.fft_M
        padded = F.pad(colored_residue, [0, M - W, 0, M - H])
        freq = torch.fft.fft2(padded)
        whitened_freq = freq * self.Q_freq_inv.to(freq.dtype)
        whitened_2n = torch.fft.ifft2(whitened_freq).real
        return whitened_2n[:, :, :H, :W]
