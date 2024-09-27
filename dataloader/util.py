import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from typing import Optional, Tuple, List
from torch import Tensor



def get_ct_image_tensor(dcm_png_path, resize=None):

    if resize is not None:
        resize_height = resize[0]
        resize_width = resize[1]

    # open a dcm png file
    img = Image.open(dcm_png_path)
    original_size = (img.height, img.width)

    # resize image. in case of PIL image, width first !!!
    if resize is not None:
        img = img.resize((resize_width, resize_height), resample=Image.BICUBIC)

    # change into numpy
    # img_buffer shape = (resized_h, resized_w), dtype = uint8
    img_buffer = np.asarray(img)

    # change numpy into tensor (resized_h, resized_w) --> (1, resized_h, resized_w)
    img_tensor = transforms.ToTensor()(img_buffer.copy())

    return img_tensor, original_size


def get_mask_tensor(mask_png_path, resize, original_size, mask_exist=True):

    if resize is not None:
        resize_height = resize[0]
        resize_width = resize[1]

    # open a mask png file
    if mask_exist:
        img = Image.open(mask_png_path)
        if resize is not None:
            # resize image
            img = img.resize((resize_width, resize_height), resample=Image.NEAREST)

        # change into numpy
        # img_buffer shape = (resized_h, resized_w), dtype = uint8
        img_buffer = np.asarray(img)
    else:
        if resize is not None:
            img_buffer = np.zeros((resize_width, resize_height), dtype=np.uint8)
        else:
            height, width = original_size
            img_buffer = np.zeros((width, height), dtype=np.uint8)

    # change numpy into tensor (resized_h, resized_w) -> (1, resized_h, resized_w)
    img_tensor = transforms.ToTensor()(img_buffer.copy())

    return img_tensor

def horizontal_flip(img_tensor: Tensor) -> Tensor:
    """ horizontal flip

    Args:
        img_tensor: shape = (n, c, h, w) or (n, c, d, h, w)
    """
    return F.hflip(img_tensor)


def save_image_from_tensor(img_tensor, output_path):

    assert img_tensor.ndim == 3
    assert img_tensor.shape[0] == 1

    img_buffer = img_tensor.numpy()
    # (C,H,W) --> (H,W,C)
    img_buffer = img_buffer.transpose(1, 2, 0)
    img_buffer = img_buffer * 255
    img_buffer = img_buffer.astype(np.uint8)
    # (H,W,C) --> (H,W)
    img_buffer = img_buffer.squeeze()

    Image.fromarray(img_buffer).save(output_path)
