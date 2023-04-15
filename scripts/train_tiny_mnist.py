# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
Train a tiny model with mnist dataset, only using two labels: 0 1
Even with cpu couple of hours should be enough.

Authors: ChenChao (chenchao214@outlook.com)
"""
import torch
import wandb

from torch.utils.data import DataLoader
from ddpm import script_utils
from datasets.tiny_mnist import MnistDataset
from ddpm.script_utils import get_args
from tools import file_utils


def main(args):
    device = args.device
    file_utils.mkdir(args.log_dir)
    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)
        # torch.compile(diffusion)  # may help. windows not support.

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))
        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        if args.log_to_wandb:
            if args.project_name is None:
                raise ValueError("args.log_to_wandb set to True but args.project_name is None")

            wandb_runner = wandb.init(
                project=args.project_name,
                entity='chenchao214',
                config=vars(args),
                name=args.run_name,
            )
            wandb.watch(diffusion)

        batch_size = args.batch_size

        target_labels = list(range(args.num_classes))
        train_dataset = MnistDataset(is_train=True, target_labels=target_labels)

        test_dataset = MnistDataset(is_train=False, target_labels=target_labels)

        train_loader = script_utils.cycle(DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        ))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=0)

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()

            x, y = next(train_loader)
            x = x.to(device)
            y = y.to(device)

            loss = diffusion(x, y)

            print(f"=====> iter: {iteration}, loss: {round(loss.item(), 6)}")

            acc_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            diffusion.update_ema()

            if iteration % args.log_rate == 0:
                test_loss = 0
                with torch.no_grad():
                    diffusion.eval()
                    for x, y in test_loader:
                        x = x.to(device)
                        y = y.to(device)
                        loss = diffusion(x, y)
                        test_loss += loss.item()

                samples = diffusion.sample(args.num_classes, device, y=torch.arange(args.num_classes, device=device))
                samples = ((samples + 1) / 2).clip(0, 1).permute(0, 2, 3, 1).numpy()
                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate
                if args.log_to_wandb:
                    wandb.log({
                        "test_loss": test_loss,
                        "train_loss": acc_train_loss,
                        "samples": [wandb.Image(sample) for sample in samples],
                    })

                acc_train_loss = 0
                print(f"---------> test loss: {round(test_loss, 6)}")

            if iteration % args.checkpoint_rate == 0:
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)

        if args.log_to_wandb:
            wandb_runner.finish()
    except KeyboardInterrupt:
        print("Keyboard interrupt, run finished early")
    finally:
        if args.log_to_wandb:
            wandb_runner.finish()


if __name__ == "__main__":
    args = get_args()
    for k, v in args.__dict__.items():
        print(f"===> {k} : {v}")
    main(args)
