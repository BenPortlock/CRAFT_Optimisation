"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np
from skimage import io
import cv2
from typing import Tuple


def loadImage(img_file):
    img = io.imread(img_file)           # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0,
                    mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0,
                    variance[2] * 255.0], dtype=np.float32)
    return img


def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.0):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w/2), int(target_h/2))

    return resized, ratio, size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


# === Additional Functionality ===
def preprocess_image(
    image: np.ndarray, canvas_size: int, mag_ratio: float
) -> Tuple[np.ndarray, float, float]:
    """
    Given an image, resizes it according to the target canvas size, normalises the pixel values channel-wise, adds padding to match the target canvas size, and transposes it from (H, W, C) to (C, H, W)

    Args:
        image: A numpy array representing the image being inferenced
        canvas_size: The target image size accepted by the model
        mag_ratio: The magnification applied to the image as required

    Returns:
        The preprocessed image along with the resizing ratios required for bounding box annotations
    """
    img_resized, target_ratio, _ = resize_aspect_ratio(
        image,
        canvas_size,
        interpolation=cv2.INTER_LINEAR,
        mag_ratio=mag_ratio
    )

    ratio_h = ratio_w = 1 / target_ratio

    image_normalised = normalizeMeanVariance(img_resized)

    h, w, _ = image_normalised.shape
    pad_w = canvas_size - w
    pad_h = canvas_size - h
    pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    image_padded = np.pad(image_normalised, pad_width, mode='constant')

    image_transposed = np.transpose(image_padded, (2, 0, 1))
    return image_transposed, ratio_w, ratio_h
