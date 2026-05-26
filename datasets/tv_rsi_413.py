"""
TV-RSI-413 Dataset Loader

Dataset: Traditional Village Remote Sensing Image Dataset
- 413 traditional villages from Jiangxi Province
- 2,478 expert-annotated scenes
- 22 semantic classes
"""

import os
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class TVRSI413Dataset(Dataset):
    """
    TV-RSI-413 Dataset for traditional village semantic segmentation.

    The dataset contains high-resolution remote sensing images with
    pixel-level annotations for 22 semantic classes related to
    traditional village morphology and cultural heritage.
    """

    CLASSES = [
        "background",
        "building",
        "road",
        "water",
        "vegetation",
        "farmland",
        "bare_soil",
        "temple",
        "ancestral_hall",
        "well",
        "pond",
        "fence",
        "bridge",
        "traditional_structure",
        "modern_building",
        "mountain",
        "forest",
        "grass",
        "cultivated_land",
        "construction_land",
        "cultural_heritage_site",
        "other",
    ]

    NUM_CLASSES = 22

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: Tuple[int, int] = (512, 512),
        transforms: Optional[Callable] = None,
        cache_data: bool = False,
    ):
        """
        Args:
            root: Dataset root directory
            split: 'train', 'val', or 'test'
            img_size: Target image size (H, W)
            transforms: Albumentations transforms
            cache_data: Whether to cache images in memory
        """
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.transforms = transforms
        self.cache_data = cache_data

        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "masks"

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.samples = self._load_samples()

        self.cache = {} if cache_data else None

    def _load_samples(self) -> List[Dict[str, str]]:
        """Load all sample paths."""
        samples = []

        image_files = sorted(list(self.image_dir.glob("*.tif")) +
                           list(self.image_dir.glob("*.png")) +
                           list(self.image_dir.glob("*.jpg")))

        for img_path in image_files:
            mask_path = self.mask_dir / img_path.with_suffix("_mask.png").name

            if not mask_path.exists():
                mask_path = self.mask_dir / img_path.with_suffix(".png").name

            if mask_path.exists():
                samples.append({
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "name": img_path.stem,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file."""
        if path.endswith(".tif") and HAS_RASTERIO:
            with rasterio.open(path) as src:
                image = src.read()
                image = np.transpose(image, (1, 2, 0))
        else:
            image = np.array(Image.open(path))

        return image

    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask from file."""
        mask = np.array(Image.open(path))
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        sample = self.samples[idx]

        image = self._load_image(sample["image"])
        mask = self._load_mask(sample["mask"])

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = torch.from_numpy(mask).long()

        output = {
            "image": image,
            "mask": mask,
            "name": sample["name"],
        }

        if self.cache is not None:
            self.cache[idx] = output

        return output


def get_dataloader(
    root: str,
    split: str,
    batch_size: int,
    img_size: Tuple[int, int] = (512, 512),
    transforms=None,
    num_workers: int = 4,
    shuffle: bool = True,
    cache_data: bool = False,
) -> torch.utils.data.DataLoader:
    """Create dataloader for TV-RSI-413 dataset."""
    dataset = TVRSI413Dataset(
        root=root,
        split=split,
        img_size=img_size,
        transforms=transforms,
        cache_data=cache_data,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )

    return loader
