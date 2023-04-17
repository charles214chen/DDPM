# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
DESCRIPTION.

Authors: ChenChao (chenchao214@outlook.com)
"""

import tqdm
import cv2
import numpy as np
from collections import namedtuple

import imageio

VideoMeta = namedtuple("VideoMeta", ["fps", "frame_num", "w", "h"])


def video_to_images(video_path):
    """
    yield image from video
    Args:
        video_path:

    Returns:

    """
    # capture init
    capture = cv2.VideoCapture(video_path)
    success, frame_gbr = capture.read()
    while success:
        yield frame_gbr
        success, frame_gbr = capture.read()

    capture.release()


def get_video_meta(video_path):
    """
    get video meta info.
    Args:
        video_path:

    Returns:

    """
    capture = cv2.VideoCapture(video_path)
    fps = int(capture.get(cv2.CAP_PROP_FPS))  # int??
    frame_num = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_meta = VideoMeta(fps=fps, frame_num=frame_num, w=w, h=h)
    print('fps: {}, frame num: {}'.format(fps, frame_num))
    return video_meta


def images_to_video(img_bgr_iterable, to_video_file, fps: int = 25):
    """
    Args:
        img_bgr_iterable: bgr
        to_video_file:
        fps:
    Returns:
    """
    capture = None
    try:
        for idx, img_bgr in tqdm.tqdm(enumerate(img_bgr_iterable), desc="Video generating..."):
            h, w, c = img_bgr.shape

            if idx == 0:
                print(f"video res: {(h, w)}")

                h_first, w_first, c_first = h, w, c
                capture = cv2.VideoWriter(to_video_file,
                                          cv2.VideoWriter_fourcc(*'XVID'),
                                          fps,
                                          (w, h))
            else:
                assert (h, w, c) == (h_first, w_first, c_first),  \
                    f"Frame {idx} shape not equal to first one. {(h, w, c)} != {(h_first, w_first, c_first)}"

            capture.write(img_bgr)
    except Exception as e:
        print(f"====> Fail, message: {e}")
    finally:
        if capture is not None:
            capture.release()
            print(f"==> video saved in {to_video_file}")


def images_to_gif(img_bgr_iterable, to_gif_file: str, fps: int = 25):
    duration = 1. / fps
    imageio.mimsave(to_gif_file, img_bgr_iterable, format="GIF", duration=duration)


def img_show(img: np.ndarray, win_name="show it!"):
    """
    在本地有屏幕开发环境上show一张图片
    Args:
        img:
        win_name:
    Returns:

    """
    cv2.imshow(win_name, img)
    cv2.waitKeyEx(0)
    cv2.destroyAllWindows()
