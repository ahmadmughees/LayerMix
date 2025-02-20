# the below code has been adapted from
# https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#AugMix
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/auto_augment.py
import math
import random

import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch import Tensor
from torchvision.transforms.v2 import functional as F, InterpolationMode


def random_magnitude(max_val: float, magnitude: float, max_levels=10., negative=True) -> float:
    assert magnitude <= max_levels, f"Level {magnitude} is out of range"
    updated_max = max_val * magnitude / max_levels
    level = random.uniform(0.1, updated_max)
    if negative and random.random() < 0.5:
        return -level
    return level


tensor_to_pil = T.ToPILImage()
pil_to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def hflip(img: Tensor, _) -> Tensor: return F.hflip(img)  # pil mirror
def invert(img: Tensor, _) -> Tensor: return F.invert(img)
def equalize(img: Tensor, _) -> Tensor: return F.equalize(img)
def autocontrast(img: Tensor, _) -> Tensor: return F.autocontrast(img)
def grayscale(img: Tensor, _) -> Tensor: return F.rgb_to_grayscale(img, num_output_channels=3)


def posterize_increasing(img: Tensor, magnitude) -> Tensor:
    """Reduce the number of bits for each color channel."""
    return F.posterize(img, 4 - int(random_magnitude(4, magnitude, negative=False)))


def posterize(img: Tensor, magnitude) -> Tensor:
    """Reduce the number of bits for each color channel."""
    return F.posterize(img, int(random_magnitude(4, magnitude, negative=False)))


def posterize_original(img: Tensor, magnitude) -> Tensor:
    """Reduce the number of bits for each color channel."""
    return F.posterize(img, 4 + int(random_magnitude(4, magnitude, negative=False)))


def solarize(img: Tensor, magnitude):
    """Invert all pixel values above a threshold."""
    magnitude = random_magnitude(1, magnitude, negative=False)
    return F.solarize(img, magnitude)


def solarize_increasing(img: Tensor, magnitude):
    """Invert all pixel values above a threshold."""
    magnitude = random_magnitude(1, 1 - magnitude, negative=False)
    return F.solarize(img, magnitude)


def rotate(img: Tensor, magnitude: int) -> Tensor:
    return F.rotate(img, random_magnitude(30, magnitude, negative=True),
        interpolation=InterpolationMode.BILINEAR)


def shear_x(img: Tensor, magnitude: int) -> Tensor:
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html
    magnitude = math.degrees(math.atan(random_magnitude(0.3, magnitude, negative=True)))
    return F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[magnitude, 0],
                    interpolation=InterpolationMode.BILINEAR)


def shear_y(img: Tensor, magnitude: int) -> Tensor:
    magnitude = math.degrees(math.atan(random_magnitude(0.3, magnitude, negative=True)))
    return F.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, magnitude],
                    interpolation=InterpolationMode.BILINEAR)


def translate_x(img: Tensor, magnitude: int) -> Tensor:
    magnitude = random_magnitude(int(img.shape[2] * 0.33), magnitude, negative=True)
    return F.affine(img, angle=0, translate=[magnitude, 0], scale=1, shear=[0, 0],
                    interpolation=InterpolationMode.BILINEAR)


def translate_y(img: Tensor, magnitude: int) -> Tensor:
    magnitude = random_magnitude(int(img.shape[1] * 0.33), magnitude, negative=True)
    return F.affine(img, angle=0, translate=[0, magnitude], scale=1, shear=[0, 0],
                    interpolation=InterpolationMode.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(img: Tensor, magnitude: int) -> Tensor:
    magnitude = random_magnitude(0.9, magnitude, negative=True)
    return F.adjust_saturation(img, magnitude + 1.0)


def contrast(img: Tensor, magnitude: int) -> Tensor:
    magnitude = random_magnitude(0.9, magnitude, negative=True)
    return F.adjust_contrast(img, magnitude + 1.0)


def brightness(img: Tensor, magnitude: int) -> Tensor:
    magnitude = random_magnitude(0.9, magnitude, negative=True)
    return F.adjust_brightness(img, magnitude + 1.0)


def sharpness(img: Tensor, magnitude: int) -> Tensor:
    magnitude = random_magnitude(0.9, magnitude, negative=True)
    return F.adjust_sharpness(img, magnitude + 1.0)


augs_image = [
    equalize,
    posterize,
    posterize_increasing,
    solarize,
    solarize_increasing,
    autocontrast,
    grayscale
]
augs_spatial = [
    rotate,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]
augs_c_test = [
    color,
    contrast,
    brightness,
    sharpness
]

# as per pix the color, contrast, brightness, sharpness are the only part of Imagenet-C test set
# but as per IPmix autocontrast is included.

# -------- MIXINGS ------------------------


def get_ab(blending_ratio, simplify)-> tuple[float, float]:
    if simplify or np.random.random() < 0.5:
        return (np.float32(np.random.beta(blending_ratio, 1)),
                np.float32(np.random.beta(1, blending_ratio)))
    else:
        a = 1 + np.float32(np.random.beta(1, blending_ratio))
        b = -np.float32(np.random.beta(1, blending_ratio))
    return a, b


def add(img1, img2, beta, simplify):
    a, b = get_ab(beta, simplify=simplify)
    if simplify:
        return a * img1 + b * img2
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2


def multiply(img1, img2, blending_raio, simplify):
    a, b = get_ab(blending_raio, simplify=simplify)
    if simplify:
        # this is my hacked implementation as I am still not sure of the benefits of the below
        # approach and I have calculated it be equivalent.
        # the below approach does give slight improvement in the accuracy.
        return (img1 ** a) * (img2.clip(1e-37) ** b)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1 ** a) * (img2.clip(1e-37) ** b)
    return out / 2


def random_pixels(img1: Tensor, img2: Tensor, blending_ratio, simplify) -> Tensor:
    _, H, W = img1.shape
    mask = (torch.rand([1, H, W]) < torch.rand([])).type(torch.float32)  # bool to float32
    return mask * img1 + (1 - mask) * img2


def random_elems(img1: Tensor, img2: Tensor, blending_ratio, simplify) -> Tensor:
    (C, H, W) = img1.shape
    mask = (torch.rand([C, H, W]) < torch.rand([])).type(torch.float32)
    return mask * img1 + (1 - mask) * img2
