from contextlib import contextmanager

import matplotlib.pyplot as plt
import torch.nn.functional as F


@contextmanager
def display(sample, ground_truth_mask=None):
    if len(sample) > 2:
        x, y, weight_map = sample
    else:
        x, y = sample
        weight_map = None

    side_pad = (x.shape[-1] - y.shape[-1]) // 2
    padding = (side_pad, side_pad, side_pad, side_pad)

    y = F.pad(y, padding, value=0)

    extent = (0, 25, 0, 25)

    things = ["image", "masked", "mask"]

    if ground_truth_mask is not None:
        ground_truth_mask = F.pad(ground_truth_mask, padding, value=0)
        things.append("true_mask")

    if weight_map is not None:
        weight_map = weight_map.squeeze(0)
        weight_map = F.pad(weight_map, padding, value=0)
        things.append("weight_map")

    fig, ax = plt.subplot_mosaic([things], sharex=True, sharey=True)

    ax["image"].imshow(x.permute(1, 2, 0), cmap=plt.cm.gray, extent=extent)
    ax["image"].axis("off")
    ax["image"].set_title("Image")

    ax["mask"].imshow(y.permute(1, 2, 0), cmap=plt.cm.gray, extent=extent)
    ax["mask"].axis("off")
    ax["mask"].set_title("Mask")

    if ground_truth_mask is not None:
        ax["true_mask"].imshow(
            ground_truth_mask.permute(1, 2, 0), cmap=plt.cm.gray, extent=extent
        )
        ax["true_mask"].axis("off")
        ax["true_mask"].set_title("True Mask")

    if weight_map is not None:
        ax["weight_map"].imshow(weight_map, cmap=plt.cm.hot, extent=extent)
        ax["weight_map"].axis("off")
        ax["weight_map"].set_title("Weight Map")

    ax["masked"].imshow(x.permute(1, 2, 0), cmap=plt.cm.gray, extent=extent)
    ax["masked"].imshow(y.permute(1, 2, 0), cmap=plt.cm.hot, extent=extent, alpha=0.4)
    ax["masked"].axis("off")
    ax["masked"].set_title("Image w/mask")

    yield fig

    plt.close()


def show(sample, ground_truth_mask=None):
    with display(sample, ground_truth_mask=ground_truth_mask):
        plt.show()
