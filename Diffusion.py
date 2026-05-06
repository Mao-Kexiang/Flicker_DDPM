import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(v, t, x_shape):
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, noise_module):
        super().__init__()
        self.model = model
        self.T = T
        self.noise_module = noise_module

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels=None):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        white_noise = torch.randn_like(x_0)
        noise = self.noise_module.colorize(white_noise)

        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        )

        if labels is not None:
            pred = self.model(x_t, t, labels)
        else:
            pred = self.model(x_t, t)

        residue = pred - noise
        residue_whitened = self.noise_module.whiten(residue)
        loss = F.mse_loss(residue_whitened, torch.zeros_like(residue_whitened), reduction='none')
        return loss


class DiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, noise_module, w=0.0):
        super().__init__()
        self.model = model
        self.T = T
        self.noise_module = noise_module
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, labels=None):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)

        if labels is not None:
            eps = self.model(x_t, t, labels)
            if self.w > 0:
                nonEps = self.model(x_t, t, torch.zeros_like(labels))
                eps = (1. + self.w) * eps - self.w * nonEps
        else:
            eps = self.model(x_t, t)

        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels=None, record_interval=50):
        x_t = x_T
        intermediates = []

        for time_step in reversed(range(self.T)):
            if time_step % 100 == 0:
                print(f"Sampling step: {time_step}")

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)

            if time_step > 0:
                noise = self.noise_module.colorize(torch.randn_like(x_t))
            else:
                noise = 0

            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."

            if time_step % record_interval == 0 or time_step == self.T - 1:
                intermediates.append(x_t.detach().cpu())

        return intermediates
