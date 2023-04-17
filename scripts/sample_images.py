import argparse
import os
import tqdm
import torch
import torchvision
from torchvision.utils import make_grid

from ddpm.script_utils import get_args
from ddpm.script_utils import get_diffusion_from_args
from tools import file_utils
from tools import cv2_utils


def main(args: argparse.Namespace):
    # key args
    model_path = args.model_path
    save_dir = args.save_dir
    num_samples = args.num_samples
    vis_process = args.vis_process
    device = args.device
    num_classes = args.num_classes

    assert model_path is not None
    assert os.path.exists(model_path), f"model file not exist: {model_path}"

    diffusion = get_diffusion_from_args(args).to(device)
    diffusion.load_state_dict(torch.load(model_path))

    file_utils.mkdir(save_dir)
    num_each_label = num_samples // num_classes

    if vis_process:
        for label in range(num_classes):
            y = torch.ones(num_each_label, dtype=torch.long, device=device) * label

            def generate_images() -> "yield image numpy array":
                gen = diffusion.sample_diffusion_sequence(num_each_label, device, y)
                for idx, image_tensor in tqdm.tqdm(enumerate(gen), desc=f"Generating for label {label}..", total=args.num_timesteps):
                    if idx % 5 != 0:  # 1000 / 5 = 200 frames
                        continue
                    grid = make_grid(image_tensor, nrow=num_each_label)
                    arr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    image_bgr = arr[..., ::-1]
                    yield image_bgr

            to_video = os.path.join(save_dir, f"{label}.mp4")
            cv2_utils.images_to_video(generate_images(), to_video)

            to_gif = os.path.join(save_dir, f"{label}.gif")
            cv2_utils.images_to_gif(list(generate_images()), to_gif)
    else:
        for label in range(num_classes):
            y = torch.ones(num_each_label, dtype=torch.long, device=device) * label
            samples = diffusion.sample(num_each_label, device, y=y)
            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                torchvision.utils.save_image(image, f"{save_dir}/{label}-{image_id}.png")


def get_sample_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    model_path = "./checkpoints/nano2/" \
                 "aigc-ddpm-tiny_mnist_ddpm-2023-04-15-19-16-iteration-2400-model.pth"
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--save_dir", type=str, default="./model_eval")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--vis_process", "-v", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    parser = get_sample_arg_parser()
    args = get_args(parser)
    for k, v in args.__dict__.items():
        print(f"===> {k}: {v}")
    main(args)
