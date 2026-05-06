import os
import torch
import torch.distributed as dist
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

from pytorch_fid.fid_score import calculate_fid_given_paths

from Diffusion import DiffusionSampler
from NoiseModule import NoiseModule
from Train import _build_model, _build_noise_module, _make_eval_labels


def _is_distributed():
    return "LOCAL_RANK" in os.environ


def _setup_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return dist.get_rank(), dist.get_world_size(), local_rank


def _cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def prepare_gt_images(root_dir, save_dir):
    if os.path.exists(save_dir) and len(os.listdir(save_dir)) >= 10000:
        return
    print(f"Extracting CIFAR-10 GT images to {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    dataset = CIFAR10(root=root_dir, train=False, download=True,
                      transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    for idx, (images, _) in enumerate(tqdm(dataloader, desc="Saving GT")):
        for i in range(images.size(0)):
            save_image(images[i], os.path.join(save_dir, f"{idx * 100 + i:05d}.png"))


def sample_for_fid(model, sampler, noise_module, config, save_dir, device,
                   rank=0, world_size=1):
    if rank == 0:
        if os.path.exists(save_dir):
            for f in os.listdir(save_dir):
                os.remove(os.path.join(save_dir, f))
        os.makedirs(save_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    mode = config.get("mode", "unconditional")
    num_samples = config.get("fid_num_samples", 10000)
    batch_size = config["batch_size"]
    img_size = config["img_size"]

    per_rank = num_samples // world_size
    remainder = num_samples % world_size
    if rank < remainder:
        per_rank += 1
    start_idx = rank * (num_samples // world_size) + min(rank, remainder)

    model.eval()
    generated = 0
    pbar = tqdm(total=per_rank, desc=f"  [Rank {rank}] Sampling for FID",
                leave=False, disable=(rank != 0))

    while generated < per_rank:
        curr_bs = min(batch_size, per_rank - generated)

        with torch.no_grad():
            x_T_white = torch.randn(curr_bs, 3, img_size, img_size, device=device)
            x_T = noise_module.colorize(x_T_white)

            if mode == "cfg":
                labels = torch.randint(0, config.get("num_labels", 10),
                                       (curr_bs,), device=device).long() + 1
                intermediates = sampler(x_T, labels)
            else:
                intermediates = sampler(x_T)

            final_imgs = intermediates[-1]
            final_imgs = torch.clamp(final_imgs * 0.5 + 0.5, 0, 1)

        for i in range(curr_bs):
            global_idx = start_idx + generated
            save_image(final_imgs[i],
                       os.path.join(save_dir, f"gen_{global_idx:05d}.png"))
            generated += 1
            pbar.update(1)
    pbar.close()

    if world_size > 1:
        dist.barrier()


def run_fid(config):
    if _is_distributed():
        rank, world_size, local_rank = _setup_dist()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device(config["device"])

    save_dir = config.get("save_dir", "./Checkpoints/")
    gt_dir = config.get("fid_gt_dir", "./FID_GT/")
    temp_dir = config.get("fid_temp_dir", "./FID_Temp/")

    if rank == 0:
        print(f"[FID] {world_size} GPU(s), checkpoints: {os.path.abspath(save_dir)}")
        prepare_gt_images('./CIFAR10', gt_dir)

    if world_size > 1:
        dist.barrier()

    all_files = os.listdir(save_dir)
    ckpt_files = [f for f in all_files if f.startswith('ckpt_') and f.endswith('_.pt')]

    epochs_available = []
    for f in ckpt_files:
        try:
            e = int(f.split('_')[-2])
            epochs_available.append((e, f))
        except (ValueError, IndexError):
            continue
    epochs_available.sort()

    interval = config.get("fid_eval_interval", 10)
    target_epochs = set(range(interval - 1, config["epoch"], interval))
    eval_list = [(e, f) for e, f in epochs_available if e in target_epochs]
    if epochs_available and epochs_available[-1] not in eval_list:
        eval_list.append(epochs_available[-1])

    if rank == 0:
        print(f"Found {len(epochs_available)} checkpoints. Evaluating {len(eval_list)}.")

    epoch_axis, fid_values = [], []

    for epoch, file_name in eval_list:
        if rank == 0:
            print(f"\nEvaluating epoch {epoch} ({file_name})")

        model = _build_model(config, device)
        model.load_state_dict(torch.load(
            os.path.join(save_dir, file_name), map_location=device))

        noise_module = _build_noise_module(config).to(device)
        sampler = DiffusionSampler(
            model, config["beta_1"], config["beta_T"], config["T"],
            noise_module, w=config.get("w", 0.0),
        ).to(device)

        sample_for_fid(model, sampler, noise_module, config, temp_dir, device,
                       rank, world_size)

        if rank == 0:
            fid_score = calculate_fid_given_paths(
                [gt_dir, temp_dir], batch_size=50, device=device, dims=2048)
            print(f"  FID: {fid_score:.4f}")
            epoch_axis.append(epoch)
            fid_values.append(fid_score)

        if world_size > 1:
            dist.barrier()

    if rank == 0 and epoch_axis:
        noise_label = config.get("noise_type", "white")
        mode_label = config.get("mode", "unconditional")

        plt.figure(figsize=(10, 6))
        plt.plot(epoch_axis, fid_values, marker='o', label=f'{noise_label} / {mode_label}')
        plt.xlabel('Epoch')
        plt.ylabel('FID Score')
        plt.title(f'FID Evolution ({noise_label} noise, {mode_label})')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'fid_evolution.png'))
        plt.close()

        np.savez(os.path.join(save_dir, 'fid_data.npz'),
                 epoch=epoch_axis, fid=fid_values)
        print(f"FID curve saved to {save_dir}/fid_evolution.png")

    if _is_distributed():
        _cleanup_dist()
