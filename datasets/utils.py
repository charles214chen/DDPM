# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
DESCRIPTION.

Authors: ChenChao (chenchao214@outlook.com)
"""
import torchvision


def get_transform():
    class RescaleChannels(object):
        def __call__(self, sample):
            # [0, 1] --> [-1, 1]
            return 2 * sample - 1

    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        RescaleChannels(),
    ])