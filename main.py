import argparse
from core.train import train, eval
from core.eval_fid import run_fid


def main(model_config=None):
    parser = argparse.ArgumentParser()

    # --- Run control ---
    parser.add_argument("--eval", action="store_true", help="Run sampling & visualization")
    parser.add_argument("--fid", action="store_true", help="Run FID evaluation")
    parser.add_argument("--mode", type=str, default="cfg", choices=["unconditional", "cfg"])

    # --- Noise type ---
    parser.add_argument("--noise_type", type=str, default="colored", choices=["white", "colored"])
    parser.add_argument("--colored_method", type=str, default="cholesky", choices=["cholesky", "fft"])
    parser.add_argument("--eta", type=float, default=0.2)

    # --- Architecture ---
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    parser.add_argument("--attn", type=int, nargs="+", default=[1])
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--img_size", type=int, default=32)

    # --- Training ---
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--multiplier", type=float, default=2.5)
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.028)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_drop_prob", type=float, default=0.1)

    # --- Sampling ---
    parser.add_argument("--w", type=float, default=1.8)
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--record_interval", type=int, default=50)

    # --- Paths ---
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--training_load_weight", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="ckpt_199_.pt", help="Checkpoint file name")
    parser.add_argument("--sampled_dir", type=str, default=None)
    parser.add_argument("--sampledNoisyImgName", type=str, default="NoisyImgs.png")
    parser.add_argument("--sampledImgName", type=str, default="SampledImgs.png")
    parser.add_argument("--nrow", type=int, default=8)

    # --- FID ---
    parser.add_argument("--fid_num_samples", type=int, default=10000)
    parser.add_argument("--fid_eval_interval", type=int, default=10)
    parser.add_argument("--fid_gt_dir", type=str, default="./FID_GT/")
    parser.add_argument("--fid_temp_dir", type=str, default=None)

    args, _ = parser.parse_known_args()

    if args.eval:
        state = "eval"
    elif args.fid:
        state = "fid"
    else:
        state = "train"

    modelConfig = {
        "state": state,
        "mode": args.mode,
        "noise_type": args.noise_type,
        "colored_method": args.colored_method,
        "eta": args.eta,
        "T": args.T,
        "channel": args.channel,
        "channel_mult": args.channel_mult,
        "attn": args.attn,
        "num_res_blocks": args.num_res_blocks,
        "dropout": args.dropout,
        "img_size": args.img_size,
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "multiplier": args.multiplier,
        "beta_1": args.beta_1,
        "beta_T": args.beta_T,
        "grad_clip": args.grad_clip,
        "label_drop_prob": args.label_drop_prob,
        "w": args.w,
        "num_labels": args.num_labels,
        "record_interval": args.record_interval,
        "device": args.device,
        "training_load_weight": args.training_load_weight,
        "test_load_weight": args.ckpt,
        "sampledNoisyImgName": args.sampledNoisyImgName,
        "sampledImgName": args.sampledImgName,
        "nrow": args.nrow,
        "fid_num_samples": args.fid_num_samples,
        "fid_eval_interval": args.fid_eval_interval,
        "fid_gt_dir": args.fid_gt_dir,
    }

    if model_config is not None:
        modelConfig.update(model_config)

    t_tag = f"_T{modelConfig['T']}"
    if modelConfig["noise_type"] == "colored":
        tag = f"_colored_eta{modelConfig['eta']}{t_tag}"
    else:
        tag = t_tag
    default_save = f"./Checkpoints{tag}/"
    default_sampled = f"./SampledImgs{tag}/"
    default_fid_temp = f"./FID_Temp{tag}/"

    modelConfig["save_dir"] = args.save_dir if args.save_dir is not None else default_save
    modelConfig["sampled_dir"] = args.sampled_dir if args.sampled_dir is not None else default_sampled
    modelConfig["fid_temp_dir"] = args.fid_temp_dir if args.fid_temp_dir is not None else default_fid_temp

    print("=" * 60)
    print("  Model Configuration")
    print("=" * 60)
    for k, v in modelConfig.items():
        print(f"  {k:.<30s} {v}")
    print("=" * 60)

    if modelConfig["state"] == "train":
        train(modelConfig)
    elif modelConfig["state"] == "eval":
        eval(modelConfig)
    elif modelConfig["state"] == "fid":
        run_fid(modelConfig)
    else:
        raise ValueError(f"Unknown state: {modelConfig['state']}")


if __name__ == '__main__':
    main()
