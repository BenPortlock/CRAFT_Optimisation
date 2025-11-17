"""
Custom extension of the torch 'Dataset' class. Each 'row' in the dataset includes an image, a character heatmap, an affinity heatmap, and the resizing ratios required for bounding box annotation
"""

import glob
from pathlib import Path
from typing import Dict, Union

import numpy as np
from torch.utils.data import Dataset

from utils.imgproc import loadImage, preprocess_image


class CraftDataset(Dataset):

    def __init__(self, data_dir: str, canvas_size: int, mag_ratio: float):
        """
        Initialises a CraftDataset object, for which each "element" is a dictionary containing an image, its ground truth heatmaps, and its resizing ratios

        Args:
            data_dir: Path to directory containing images and heatmaps
            canvas_size: Image size used by the model (padded if necessary)
            mag_ratio: Magnification factor applied to the image
        """
        self.data_dir = data_dir
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.image_paths = glob.glob(f"{data_dir}/images/*")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, float]]:
        """
        Loads an image and its heatmaps, preprocessing the former

        Args:
            idx: The index of the data being retrieved

        Returns:
            A dictionary containing the preprocessed image, the associated char and affinity maps, and the resizing ratios required for bounding box annotations
        """
        image_path = self.image_paths[idx]
        image_name = Path(image_path).stem

        image = loadImage(image_path)
        image_resized, ratio_w, ratio_h = preprocess_image(
            image=image,
            canvas_size=self.canvas_size,
            mag_ratio=self.mag_ratio,
        )
        char_map = np.load(f"{self.data_dir}/char_maps/{image_name}.npy")
        affinity_map = np.load(
            f"{self.data_dir}/affinity_maps/{image_name}.npy"
        )
        return {
            "image": image_resized,
            "char_map": char_map,
            "affinity_map": affinity_map,
            "ratio_w": ratio_w,
            "ratio_h": ratio_h
        }
