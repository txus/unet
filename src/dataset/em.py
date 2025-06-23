import glob
import os

import numpy as np
from PIL import Image
from scipy import ndimage
from torchvision.datasets import VisionDataset


def compute_weight_map(mask, w0=10, sigma=5):
    """
    Compute the weight map as described in the U-Net paper.

    Parameters:
        mask: 2D numpy array. Binary mask where cells are 1 and background is 0.
        w0: Weight for border emphasis.
        sigma: Gaussian sigma for border weighting.

    Returns:
        weight_map: 2D numpy array of same shape as mask.
    """
    # Label individual cells
    labeled_mask, num_objects = ndimage.label(mask)

    # Compute class weights: simple inverse frequency
    wc = np.zeros_like(mask, dtype=np.float32)
    if mask.sum() > 0:
        freq_fg = mask.sum()
        freq_bg = mask.size - freq_fg
        wc[mask == 1] = 1.0 / freq_fg
        wc[mask == 0] = 1.0 / freq_bg

    # Distance maps to nearest and second-nearest objects
    distances = np.zeros((num_objects, *mask.shape), dtype=np.float32)
    for i in range(1, num_objects + 1):
        cell_mask = labeled_mask == i
        distances[i - 1] = ndimage.distance_transform_edt(~cell_mask)

    if num_objects < 2:
        return wc  # No borders to emphasize

    # Stack distances and find smallest two distances per pixel
    stacked = np.sort(distances, axis=0)
    d1 = stacked[0]
    d2 = stacked[1]

    border_term = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))

    weight_map = wc + border_term
    return weight_map


class EMSegmentationDataset(VisionDataset):
    def __init__(
        self, root: str, prelude, input_epilogue, target_epilogue, weight_map_epilogue
    ):
        self.root = root

        self.prelude = prelude
        self.input_epilogue = input_epilogue
        self.target_epilogue = target_epilogue
        self.weight_map_epilogue = weight_map_epilogue

        if self.weight_map_epilogue is not None:
            self.precompute_weight_maps()

    def precompute_weight_maps(self):
        print("Precomputing weight maps...")
        os.makedirs(f"{self.root}/weight_maps", exist_ok=True)

        for image in glob.glob(f"{self.root}/labels/train-labels*.jpg"):
            file_name = image.split("/")[-1].replace(".jpg", ".npy")
            target = Image.open(image)

            out_path = f"{self.root}/weight_maps/{file_name}"
            if not os.path.exists(out_path):
                with open(f"{self.root}/weight_maps/{file_name}", "wb") as f:
                    np.save(f, compute_weight_map(np.round(np.array(target) / 255.0)))

    def __getitem__(self, index: int):
        padded_idx = str(index).rjust(2, "0")
        path = f"{self.root}/images/train-volume{padded_idx}.jpg"
        target = f"{self.root}/labels/train-labels{padded_idx}.jpg"
        img = Image.open(path)
        weight_map_path = f"{self.root}/weight_maps/train-labels{padded_idx}.npy"
        target = Image.open(target)

        if self.weight_map_epilogue:
            with open(weight_map_path, "rb") as f:
                weight_map = np.load(f)
            input_tile, target_tile, weight_map_tile = self.prelude(
                img, target, weight_map
            )

            return self.input_epilogue(input_tile), (
                self.target_epilogue(target_tile),
                self.weight_map_epilogue(weight_map_tile),
            )
        else:
            inputs, targets = self.prelude(img, target)
            return self.input_epilogue(inputs), self.target_epilogue(targets)

    def __len__(self):
        return len(glob.glob(f"{self.root}/images/*.jpg"))
