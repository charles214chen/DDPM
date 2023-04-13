# !/usr/bin/env python3
# coding=utf-8
#
# All Rights Reserved
#
"""
Utils related to directory / file-io.

Authors: ChenChao (chenchao214@outlook.com)
"""

import os
import json
import uuid


def mkdir(dir_):
    """
    Create dir if needed.
    """
    if not os.path.exists(dir_):
        os.makedirs(dir_)


def read_json(json_file):
    """
    Args:
        json_file:

    Returns:

    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def write_json(data, json_file):
    """
    Args:
        data:
        json_file:

    Returns:

    """
    if os.path.exists(json_file):
        print("Warning: override {}".format(json_file))
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)


def get_random_id():
    """
    get random id (str)
    """
    return str(uuid.uuid1())


def write_bytes_to_disk(bytes_content, file_path):
    """
    bytes数据写进磁盘
    """
    with open(file_path, "wb") as fout:
        fout.write(bytes_content)
    print("file saved in {}".format(file_path))


def read_file_from_disk(file_path):
    """
    从磁盘中读文件，返回bytes
    """
    with open(file_path, "rb") as f:
        bytes_content = f.read()
    return bytes_content
