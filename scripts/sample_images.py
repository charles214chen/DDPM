import argparse
import os
import torch
import torchvision

from ddpm.script_utils import get_args
from ddpm.script_utils import get_diffusion_from_args
from tools import file_utils


def main(model_path: str, save_dir: str, num_images: int, vis_process=True):
    assert model_path is not None
    assert os.path.exists(model_path), f"model file not exist: {model_path}"
    train_args = get_args()
    device = train_args.device
    num_classes = train_args.num_classes

    diffusion = get_diffusion_from_args(train_args).to(device)
    diffusion.load_state_dict(torch.load(model_path))

    if vis_process:
        # TODO visualize the 'diffusion' processing, to output video.
        pass
    else:
        for label in range(num_classes):
            y = torch.ones(num_images // num_classes, dtype=torch.long, device=device) * label
            samples = diffusion.sample(num_images // num_classes, device, y=y)
            for image_id in range(len(samples)):
                image = ((samples[image_id] + 1) / 2).clip(0, 1)
                file_utils.mkdir(save_dir)
                torchvision.utils.save_image(image, f"{save_dir}/{label}-{image_id}.png")


def get_sample_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    model_path = "./checkpoints/nano2/" \
                 "aigc-ddpm-tiny_mnist_ddpm-2023-04-15-19-16-iteration-800-model.pth"
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--save_dir", type=str, default="./model_eval")
    parser.add_argument("--num_images", type=int, default=10)

    parser.add_argument("--vis_process", "-v", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_sample_args()
    print(args)
    main(**args.__dict__)
