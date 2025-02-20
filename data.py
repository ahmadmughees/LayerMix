import inspect
import random
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import RandomErasing, Normalize

from transforms import pil_to_tensor


class MixerDataset(Dataset):
    def __init__(self,
            dataset: Dataset,
            mixing_set: Dataset,
            depth: int,
            width: int,
            image_aug_fns: list[Callable],
            spatial_aug_fns: list[Callable],
            blending_fns: list[Callable],
            magnitude: int,
            blending_ratio: float,
            mixer_type: str,
            jsd: bool = False,
            normalize: Callable = Normalize([0.5] * 3, [0.5] * 3)
        ):
        self.mixer_function = getattr(self, mixer_type)
        self.dataset = dataset
        self.mixing_set = mixing_set
        self.len_mixing_set = len(mixing_set)
        self.depth = depth
        self.width = width
        self.image_aug_fns = image_aug_fns
        self.spatial_aug_fns = spatial_aug_fns
        self.all_aug_fns = image_aug_fns + spatial_aug_fns
        self.blending_fns = blending_fns
        self.magnitude = magnitude
        self.blending_ratio = blending_ratio
        self.jsd = jsd
        self.normalize = normalize
        self.patch_sizes = [4, 8, 16, 32, 64, 224]  # TODO this is for ImageNet
        print(f" ----------------- \n "
              f"mixing function used is \n {inspect.getsource(self.mixer_function)} "
              f"\n ------------")

    def __getitem__(self, i):
        x, y = self.dataset[i]  # returns the PIL image and the label
        x = pil_to_tensor(x)
        mixing_pic = pil_to_tensor(self.mixing_set[torch.randint(self.len_mixing_set, (1,))][0])

        # random erase is a special and weired case, we have to normalize the image first.
        if self.mixer_function == self.random_erase:
            return self.mixer_function(self.normalize(x), None), y

        if self.jsd:
            return (self.normalize(x),
                    self.normalize(self.mixer_function(x, mixing_pic)),
                    self.normalize(self.mixer_function(x, mixing_pic))), y
        return self.normalize(self.mixer_function(x, mixing_pic)), y

    def random_erase(self, img: Tensor, _) -> Tensor:
        return RandomErasing()(img)

    def baseline(self, img: Tensor, _) -> Tensor:
        return img

    def ipmix(self, img: Tensor, mixing_img: Tensor) -> Tensor:
        mix = torch.zeros_like(img)
        # pytorch supports dirichlet but that is slightly slow on cpu my quick exp shows it takes half-time.
        # I may be wrong but some other time pal!
        for mw in torch.from_numpy(np.float32(np.random.dirichlet([1] * self.depth))):  # todo k=2, t=3 from paper
            ops = random.choices(self.image_aug_fns + self.spatial_aug_fns, k=self.width)  # todo move all functions to torch.rand
            mixed = img.clone().detach()  # equivalent to img.copy()
            if torch.randint(2, (1,)):
                for op in ops:  # image augs
                    mixed = op(mixed, self.magnitude)
            else:  # patch augs
                mixed = self.__patch_mixer(mixed, mixing_img)
            mix += mw * mixed
        m = torch.rand(1)  # equivalent to np.uniform
        return (1 - m) * img + m * mix

    def __patch_mixer(self, img: Tensor, mixing_img: Tensor) -> Tensor:
        """ops will be per patch for performance"""

        # width_rand = torch.randint(1, self.width + 1, (1,)).item()
        width_rand = 1
        rand_patch_sizes = random.choices(self.patch_sizes, k=width_rand)  # list of ints for random patches
        rand_mixing_ops = random.choices(self.blending_fns, k=width_rand)  # list of fns

        _, H, W = img.shape
        patches = list[tuple[int, int, int, int]]()
        for patch_size in rand_patch_sizes:
            patch_left = random.randint(0, W - patch_size)
            patch_top = random.randint(0, H - patch_size)

            if torch.randint(2, (1,)):
                patch_right, patch_bottom = patch_left + patch_size, patch_top + patch_size
            else:
                patch_right = patch_left + torch.randint(2, patch_size, (1,)).item()
                patch_bottom = patch_top + torch.randint(2, patch_size, (1,)).item()
            patches.append((patch_top, patch_bottom, patch_left, patch_right))

        for p, mixing_fn in zip(patches, rand_mixing_ops):
            img[p[0]:p[1], p[2]:p[3]] = mixing_fn(img[p[0]:p[1], p[2]:p[3]],
                                                  mixing_img[p[0]:p[1], p[2]:p[3]], self.blending_ratio, simplify=False)
        return torch.clip(img, 0, 1)

    def pixmix(self, img: Tensor, mixing_img: Tensor):
        mixed = img if torch.randint(2, (1,)) else self.__random_augment(img)
        for mixing_op in random.choices(self.blending_fns, k=torch.randint(self.depth+1, (1,)).item()):
            aug_image_copy = mixing_img if torch.randint(2, (1,)) else self.__random_augment(img)
            mixed = torch.clip(mixing_op(mixed, aug_image_copy, self.blending_ratio, simplify=False), 0, 1)
        return mixed

    def __random_augment(self, img: Tensor) -> Tensor:
        return random.choice(self.all_aug_fns)(img, self.magnitude)

    def laddermix(self, img: Tensor, mixing_pic: Tensor) -> Tensor:
        steps = torch.randint(3, (1,)).item()
        img_copy = img.clone().detach()
        aug_fns = random.choices(self.all_aug_fns, k=steps+1)
        img = aug_fns[0](img, self.magnitude)
        if steps == 0: return img

        img_2 = aug_fns[0](img_copy, self.magnitude)
        img = random.choice(self.blending_fns)(img, img_2, self.blending_ratio, simplify=False)
        img = torch.clip(img, 0, 1)
        if steps == 1: return img

        img = random.choice(self.blending_fns)(img, mixing_pic, self.blending_ratio, simplify=False)
        img = torch.clip(img, 0, 1)
        img = aug_fns[0](img, self.magnitude)
        return img


    def augmix(self, img: Tensor, _) -> Tensor:
        mixing_weights = torch.from_numpy(np.float32(np.random.dirichlet([1.] * self.width)))
        m = torch.from_numpy(np.float32(np.random.beta(1.,1.)))
        mixed = torch.zeros_like(img)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = random.choices(self.all_aug_fns, k=depth)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug, self.magnitude)
            mixed += mw * img_aug
        torch.clip(mixed, 0, 1., out=mixed)
        return m * img + (1 - m) * mixed

    def __len__(self):
        return len(self.dataset)
