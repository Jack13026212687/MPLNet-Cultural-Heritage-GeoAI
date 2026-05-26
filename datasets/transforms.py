"""
Data transforms and augmentations for GeoAI segmentation.
"""

import numpy as np
from typing import Tuple, List, Optional

import torch
import torch.nn.functional as F


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, **data):
        for t in self.transforms:
            data = t(**data)
        return data


class Resize:
    """Resize image and mask."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, image, mask=None, **kwargs):
        import cv2
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return {"image": image, "mask": mask}


class RandomFlip:
    """Random horizontal and vertical flip."""

    def __init__(self, h_prob: float = 0.5, v_prob: float = 0.5):
        self.h_prob = h_prob
        self.v_prob = v_prob

    def __call__(self, image, mask=None, **kwargs):
        import cv2

        if np.random.random() < self.h_prob:
            image = cv2.flip(image, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        if np.random.random() < self.v_prob:
            image = cv2.flip(image, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)

        return {"image": image, "mask": mask}


class RandomRotate:
    """Random rotation by 90, 180, or 270 degrees."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, mask=None, **kwargs):
        import cv2

        if np.random.random() < self.prob:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k).copy()
            if mask is not None:
                mask = np.rot90(mask, k).copy()

        return {"image": image, "mask": mask}


class ColorJitter:
    """Random color jittering."""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        prob: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def __call__(self, image, mask=None, **kwargs):
        import cv2
        import random

        if np.random.random() < self.prob:
            if self.brightness > 0:
                factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                image = np.clip(image * factor, 0, 255).astype(np.uint8)

            if self.contrast > 0:
                factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                mean = image.mean(axis=(0, 1), keepdims=True)
                image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

            if self.saturation > 0 and len(image.shape) == 3 and image.shape[2] == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + random.uniform(-self.saturation, self.saturation)), 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return {"image": image, "mask": mask}


class Normalize:
    """Normalize image with mean and std."""

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, mask=None, **kwargs):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return {"image": image, "mask": mask}


def get_train_transforms(img_size: Tuple[int, int] = (512, 512)):
    """Get training transforms."""
    return Compose([
        Resize(img_size),
        RandomFlip(h_prob=0.5, v_prob=0.5),
        RandomRotate(prob=0.5),
        ColorJitter(prob=0.5),
        Normalize(),
    ])


def get_val_transforms(img_size: Tuple[int, int] = (512, 512)):
    """Get validation transforms."""
    return Compose([
        Resize(img_size),
        Normalize(),
    ])
