import argparse
import datetime

import torch
import torch.nn.functional as F

from ddpm.unet import UNet
from ddpm.diffusion import (
    GaussianDiffusion,
    generate_linear_schedule,
    generate_cosine_schedule,
)


def cycle(dl):
    """
    https://github.com/lucidrains/denoising-diffusion-pytorch/
    """
    while True:
        for data in dl:
            yield data


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    """
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def diffusion_defaults():
    defaults = dict(
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        use_labels=False,

        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        time_emb_dim=128 * 4,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),

        ema_decay=0.9999,
        ema_update_rate=1,
    )

    return defaults


def get_diffusion_from_args(args):
    activations = {
        "relu": F.relu,
        "mish": F.mish,
        "silu": F.silu,
    }

    model = UNet(
        img_channels=args.img_channels,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
        time_emb_dim=args.time_emb_dim,
        norm=args.norm,
        dropout=args.dropout,
        activation=activations[args.activation],
        attention_resolutions=args.attention_resolutions,
        num_classes=args.num_classes,
        initial_pad=0,
        num_groups=args.num_groups
    )

    if args.schedule == "cosine":
        betas = generate_cosine_schedule(args.num_timesteps)
    else:
        betas = generate_linear_schedule(
            args.num_timesteps,
            args.schedule_low * 1000 / args.num_timesteps,
            args.schedule_high * 1000 / args.num_timesteps,
        )

    diffusion = GaussianDiffusion(
        model,
        args.img_size,
        args.img_channels,
        args.num_classes,
        betas,
        ema_decay=args.ema_decay,
        ema_update_rate=args.ema_update_rate,
        ema_start=2000,
        loss_type=args.loss_type,
    )

    return diffusion


def get_args() -> argparse.Namespace:
    """
    Get args for all (u-net and diffusion training super-params.)
    Default for training tiny model with mnist dataset. You can also pass args to override.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    time_frame = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")
    run_name = f"tiny_mnist_{time_frame}"
    defaults = dict(
        img_channels=1,
        img_size=(28, 28),
        num_classes=2,
        num_groups=2,
        learning_rate=2e-4,
        batch_size=128,
        iterations=800000,
        log_to_wandb=False,
        log_rate=500,
        checkpoint_rate=1000,
        log_dir="./checkpoints/nano2",
        project_name="aigc-ddpm",
        run_name=run_name,
        model_checkpoint=None,
        optim_checkpoint=None,
        schedule_low=1e-4,
        schedule_high=0.02,
        device=device,
        num_timesteps=1000,
        schedule="linear",
        loss_type="l2",
        base_channels=4,
        channel_mults=(1, 2),
        num_res_blocks=1,
        time_emb_dim=8,
        norm="gn",
        dropout=0.1,
        activation="silu",
        attention_resolutions=(1,),
        ema_decay=0.999,
        ema_update_rate=1,
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # args could be override by user
    return parser.parse_args()


if __name__ == '__main__':
    betas = generate_cosine_schedule(10)
    betas = generate_linear_schedule(
        1000,
        0 * 1000 / 1000,
        1 * 1000 / 1000,
    )
    print(betas)