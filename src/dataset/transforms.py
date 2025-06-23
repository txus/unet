import random

import elasticdeform
import torch
import torch.nn.functional as F


class ElasticDeformation:
    def __init__(self, sigma: int, points: int) -> None:
        self.sigma = sigma
        self.points = points

    def __call__(self, *samples):
        out = elasticdeform.deform_random_grid(
            [s.squeeze(0).numpy() for s in samples],
            sigma=self.sigma,
            points=self.points,
            zoom=1.5,
            rotate=30,
            prefilter=False,
        )
        return [torch.from_numpy(x).unsqueeze(0) for x in out]


class NormalizeAndQuantize:
    def __call__(self, sample):
        return torch.round(sample / 255.0).to(torch.long)


class UNetRandomCropWithPadding:
    def __init__(
        self, input_tile_size: int, target_tile_size: int, padding_mode="reflect"
    ):
        self.input_tile_size = input_tile_size
        self.target_tile_size = target_tile_size
        self.pad_size = (input_tile_size - target_tile_size) // 2
        self.padding_mode = padding_mode

    def __call__(self, image, target, weight_map=None):
        """
        Args:
            image: Tensor of shape (C, H, W)
            target: Tensor of shape (H, W) or (1, H, W)
        Returns:
            input_patch: padded input image patch (C, input_size, input_size)
            target_patch: centered target patch (H, W)
        """
        _, H, W = image.shape

        # Ensure we can crop the label patch
        if H < self.target_tile_size or W < self.target_tile_size:
            raise ValueError("Input image smaller than desired label size.")

        # Choose random top-left for the label patch
        y = random.randint(0, H - self.target_tile_size)
        x = random.randint(0, W - self.target_tile_size)

        # Extract label patch (center region)
        target_patch = target[
            ..., y : y + self.target_tile_size, x : x + self.target_tile_size
        ]
        if weight_map is not None:
            weight_map_patch = weight_map[
                ..., y : y + self.target_tile_size, x : x + self.target_tile_size
            ]

        # Coordinates for the input patch with padding around the label patch
        input_y = max(0, y - self.pad_size)
        input_x = max(0, x - self.pad_size)
        input_y_end = min(H, y + self.target_tile_size + self.pad_size)
        input_x_end = min(W, x + self.target_tile_size + self.pad_size)

        input_patch = image[:, input_y:input_y_end, input_x:input_x_end]

        # Compute actual padding needed (if the crop was near the border)
        pad_top = max(0, self.pad_size - y)
        pad_left = max(0, self.pad_size - x)
        pad_bottom = max(0, (y + self.target_tile_size + self.pad_size) - H)
        pad_right = max(0, (x + self.target_tile_size + self.pad_size) - W)

        # Pad to get the exact input size
        padding = [pad_left, pad_right, pad_top, pad_bottom]  # left, right, top, bottom
        input_patch = F.pad(input_patch, padding, mode=self.padding_mode)

        if weight_map is not None:
            return input_patch, target_patch, weight_map_patch
        else:
            return input_patch, target_patch
