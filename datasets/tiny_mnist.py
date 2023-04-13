# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
DESCRIPTION.

Authors: ChenChao (chenchao214@outlook.com)
"""
import os

import torchvision.datasets as datasets
from torch.utils.data import Dataset

from datasets.utils import get_transform
from tools import file_utils


class MnistDataset(Dataset):
    def __init__(self, is_train=True, target_labels: list = None):
        self.is_train = is_train
        self.target_labels = target_labels if target_labels is not None else list(range(10))
        self.imgs, self.labels = self._get_data()

    def _get_data(self):
        mnist_root = _get_mnist_data_root()
        dataset = datasets.MNIST(root=mnist_root, train=self.is_train, download=True, transform=get_transform())
        imgs, labels = [], []
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            if label in self.target_labels:
                imgs.append(img)
                labels.append(label)
        return imgs, labels

    def __getitem__(self, item):
        return self.imgs[item], self.labels[item]

    def __len__(self):
        return len(self.imgs)


def _get_mnist_data_root():
    base = os.path.dirname(os.path.abspath(__file__))  # datasets
    mnist_root = os.path.join(base, "mnist")
    file_utils.mkdir(mnist_root)
    return mnist_root


if __name__ == '__main__':
    dataset = MnistDataset(is_train=False, target_labels=[0, 1])
    print(f"0, 1 samples: {len(dataset)}")
    sample, target = dataset[0]
    img = sample.numpy()
    print(img.shape)
    print(target)
