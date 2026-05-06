import os
import sys
from typing import Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from Diffusion import DiffusionTrainer, DiffusionSampler
from NoiseModule import NoiseModule
from Visualizer import DiffusionAnalyzer
from Scheduler import GradualWarmupScheduler


def _setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return dist.get_rank(), local_rank


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def _build_noise_module(config):
    return NoiseModule(
        noise_type=config.get("noise_type", "white"),
        img_size=config["img_size"],
        eta=config.get("eta", 0.2),
        method=config.get("colored_method", "cholesky"),
    )


def _build_model(config, device):
    if config["mode"] == "cfg":
        from ModelCondition import UNet
        model = UNet(
            T=config["T"], num_labels=config.get("num_labels", 10),
            ch=config["channel"], ch_mult=config["channel_mult"],
            num_res_blocks=config["num_res_blocks"], dropout=config["dropout"],
        ).to(device)
    else:
        from Model import UNet
        model = UNet(
            T=config["T"], ch=config["channel"], ch_mult=config["channel_mult"],
            attn=config.get("attn", [2]),
            num_res_blocks=config["num_res_blocks"], dropout=config["dropout"],
        ).to(device)
    return model


def _plot_loss_curve(epoch_losses, save_dir, mode):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(epoch_losses)), epoch_losses, label='Train Loss')
    plt.title(f'Training Loss ({mode}, Epoch {len(epoch_losses) - 1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'))
    plt.close()


def _make_eval_labels(config, device):
    batch_size = config["batch_size"]
    num_labels = config.get("num_labels", 10)
    step = batch_size // num_labels
    label_list = []
    k = 0
    for i in range(1, batch_size + 1):
        label_list.append(torch.ones(size=[1]).long() * k)
        if i % step == 0 and k < num_labels - 1:
            k += 1
    labels = torch.cat(label_list, dim=0).long().to(device) + 1
    return labels


def _step_from_index(idx, total_intermediates, record_interval):
    if idx == 0:
        return None
    return (total_intermediates - 1 - idx) * record_interval


# ------------------------------------------------------------------ #
#  Train
# ------------------------------------------------------------------ #
def train(config: Dict):
    use_ddp = "LOCAL_RANK" in os.environ
    if use_ddp:
        rank, local_rank = _setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank = 0, 0
        device = torch.device(config["device"])

    mode = config.get("mode", "unconditional")
    save_dir = config.get("save_dir", "./Checkpoints/")
    if _is_main():
        os.makedirs(save_dir, exist_ok=True)

    if use_ddp and _is_main():
        CIFAR10(root='./CIFAR10', train=True, download=True)
    if use_ddp:
        dist.barrier()

    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=False,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    nw = config.get("num_workers", 4)
    if use_ddp:
        data_sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], sampler=data_sampler,
            num_workers=nw, drop_last=True, pin_memory=True)
    else:
        data_sampler = None
        dataloader = DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=True,
            num_workers=nw, drop_last=True, pin_memory=True)

    net_model = _build_model(config, device)

    if config.get("training_load_weight") is not None:
        net_model.load_state_dict(torch.load(
            os.path.join(save_dir, config["training_load_weight"]),
            map_location=device), strict=False)
        if _is_main():
            print("Resumed from checkpoint.")

    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=config["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=config.get("multiplier", 2.),
        warm_epoch=config["epoch"] // 10, after_scheduler=cosineScheduler)

    noise_module = _build_noise_module(config)
    trainer = DiffusionTrainer(
        net_model, config["beta_1"], config["beta_T"], config["T"], noise_module
    ).to(device)

    if use_ddp:
        trainer = DDP(trainer, device_ids=[local_rank])

    if _is_main():
        print(f"Training started: {len(dataloader)} batches/epoch, {config['epoch']} epochs")

    epoch_losses = []

    try:
        for e in range(config["epoch"]):
            net_model.train()
            total_loss, cnt = 0, 0

            if data_sampler is not None:
                data_sampler.set_epoch(e)

            loader = tqdm(dataloader, dynamic_ncols=True) if _is_main() else dataloader
            for images, labels_data in loader:
                b = images.shape[0]
                optimizer.zero_grad()
                x_0 = images.to(device)

                if mode == "cfg":
                    labels = labels_data.to(device) + 1
                    if np.random.rand() < config.get("label_drop_prob", 0.1):
                        labels = torch.zeros_like(labels)
                    loss = trainer(x_0, labels).sum() / b ** 2.
                else:
                    loss = trainer(x_0).sum() / 1000.

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), config["grad_clip"])
                optimizer.step()

                total_loss += loss.item()
                cnt += 1

                if _is_main() and hasattr(loader, 'set_postfix'):
                    loader.set_postfix(ordered_dict={
                        "epoch": e,
                        "loss": loss.item(),
                        "avg_loss": total_loss / cnt,
                        "LR": optimizer.param_groups[0]["lr"],
                    })

            avg_loss = total_loss / cnt
            epoch_losses.append(avg_loss)

            if _is_main():
                print(f"Epoch {e}/{config['epoch']-1} | avg_loss={avg_loss:.6f} | lr={optimizer.param_groups[0]['lr']:.2e}")
                _plot_loss_curve(epoch_losses, save_dir, mode)
                torch.save(net_model.state_dict(),
                           os.path.join(save_dir, f'ckpt_{e}_.pt'))

            warmUpScheduler.step()
    finally:
        if use_ddp:
            _cleanup_ddp()


# ------------------------------------------------------------------ #
#  Eval
# ------------------------------------------------------------------ #
def eval(config: Dict):
    device = torch.device(config["device"])
    mode = config.get("mode", "unconditional")
    sampled_dir = config.get("sampled_dir", "./SampledImgs/")
    os.makedirs(sampled_dir, exist_ok=True)

    with torch.no_grad():
        net_model = _build_model(config, device)
        net_model.dropout = 0.

        ckpt_path = os.path.join(config.get("save_dir", "./Checkpoints/"),
                                 config["test_load_weight"])
        net_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Model loaded from {ckpt_path}")
        net_model.eval()

        noise_module = _build_noise_module(config)
        sampler = DiffusionSampler(
            net_model, config["beta_1"], config["beta_T"], config["T"],
            noise_module, w=config.get("w", 0.0),
        ).to(device)

        analyzer = DiffusionAnalyzer(
            out_dir=os.path.join(sampled_dir, "Analysis"))

        x_T_white = torch.randn(
            size=[config["batch_size"], 3, config["img_size"], config["img_size"]],
            device=device)
        x_T = noise_module.to(device).colorize(x_T_white)

        saveNoisy = torch.clamp(x_T * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            sampled_dir, config.get("sampledNoisyImgName", "NoisyImgs.png")),
            nrow=config.get("nrow", 8))

        labels = _make_eval_labels(config, device) if mode == "cfg" else None
        if labels is not None:
            print("Conditional labels:", labels.cpu().numpy())

        record_interval = config.get("record_interval", max(1, config["T"] // 10))
        print(f"Sampling (T={config['T']}, mode={mode})...")
        intermediates = sampler(x_T, labels=labels, record_interval=record_interval)

        for i, img_tensor in enumerate(intermediates):
            img_01 = torch.clamp(img_tensor * 0.5 + 0.5, 0, 1)

            step_val = _step_from_index(i, len(intermediates), record_interval)
            if step_val is None:
                step_val = config["T"] - 1

            if i == len(intermediates) - 1:
                img_name = config.get("sampledImgName", "SampledImgs.png")
            else:
                img_name = f"step_{step_val:04d}_{config.get('sampledImgName', 'SampledImgs.png')}"

            save_image(img_01, os.path.join(sampled_dir, img_name),
                       nrow=config.get("nrow", 8))

            analyzer.plot_power_spectrum_grid(img_01, step=step_val)
            print(f"Step {step_val:04d}: saved & analyzed.")

        analyzer.plot_evolution_charts()
        print(f"All done! Results in: {sampled_dir}")
