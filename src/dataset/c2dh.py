import glob

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset


class C2DHSegmentationDataset(VisionDataset):
    def __init__(self, root: str, prelude, input_epilogue, target_epilogue):
        self.root = root

        self.prelude = prelude
        self.input_epilogue = input_epilogue
        self.target_epilogue = target_epilogue

        self.samples_per_collection = len(glob.glob(f"{self.root}/01/t*.tif"))

    def __getitem__(self, index: int):
        padded_idx = str(index % self.samples_per_collection).rjust(3, "0")

        collection = "01" if index < self.samples_per_collection else "02"

        path = f"{self.root}/{collection}/t{padded_idx}.tif"
        target = f"{self.root}/{collection}_ST/SEG/man_seg{padded_idx}.tif"
        img = Image.open(path)
        target = (np.array(Image.open(target)) > 0).astype(np.uint8) * 255

        inputs, targets = self.prelude(img, target)

        return self.input_epilogue(inputs), self.target_epilogue(targets)

    def __len__(self):
        return len(glob.glob(f"{self.root}/01/t*.tif")) + len(
            glob.glob(f"{self.root}/02/t*.tif")
        )
