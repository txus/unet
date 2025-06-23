import matplotlib
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2

from dataset.em import EMSegmentationDataset
from dataset.c2dh import C2DHSegmentationDataset
from dataset.transforms import (
    ElasticDeformation,
    NormalizeAndQuantize,
    UNetRandomCropWithPadding,
)

from dataset.display import show, display

torch.manual_seed(0)


def _get_output_size(tile_size, model):
    contraction_steps = len(model.contraction)
    expansion_steps = len(model.up_convs)

    size = tile_size
    for contraction_step in range(contraction_steps):
        size -= 4  # 2 pixels per 2 convolutions
        if contraction_step != contraction_steps - 1:
            size /= 2  # max pool

    for expansion_step in range(expansion_steps):
        size *= 2  # residual concat
        size -= 4  # 2 pixels per 2 convolutions

    size -= 4  # final 2 convs

    return int(size)


def get_dataset(
    model, dataset: str, root="./data", tile_size: int = 512, use_weight_map=False
) -> tuple[datasets.VisionDataset, datasets.VisionDataset]:
    assert (
        tile_size <= 512
    ), "input images are originally 512x512. Choose less than that to use smaller random tiles"

    output_size = _get_output_size(tile_size, model)

    print(f"Inferred {output_size} output size for tiles of {tile_size}")

    prelude = v2.Compose(
        [
            v2.ToImage(),
            ElasticDeformation(
                sigma=10, points=3  # as per the paper, 10 pixels std, 3x3 grid
            ),
            UNetRandomCropWithPadding(
                input_tile_size=tile_size,
                target_tile_size=output_size,
                padding_mode="reflect",
            ),
        ]
    )

    input_epilogue = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    target_epilogue = v2.Compose(
        [
            NormalizeAndQuantize(),
            v2.ToDtype(torch.long, scale=False),
        ]
    )

    weight_map_epilogue = None
    if use_weight_map:
        weight_map_epilogue = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=False),
            ]
        )

    if dataset == "em":
        return EMSegmentationDataset(
            "./data/em",
            prelude=prelude,
            input_epilogue=input_epilogue,
            target_epilogue=target_epilogue,
            weight_map_epilogue=weight_map_epilogue,
        )
    elif dataset == "phc":
        return C2DHSegmentationDataset(
            "./data/phc-c2dh-u373",
            prelude=prelude,
            input_epilogue=input_epilogue,
            target_epilogue=target_epilogue,
        )
    elif dataset == "hela":
        return C2DHSegmentationDataset(
            "./data/DIC-C2DH-HeLa",
            prelude=prelude,
            input_epilogue=input_epilogue,
            target_epilogue=target_epilogue,
        )


def train_val_dataset(dataset, val_split: int):
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), test_size=val_split
    )
    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    return datasets


def get_dataloaders(
    dataset: str,
    batch_size: int,
    tile_size: int,
    model,
    val_split=0.2,
    num_workers=4,
    root="./data",
    use_weight_map=False,
) -> DataLoader:
    if use_weight_map and dataset != "em":
        raise NotImplementedError("Weight map only makes sense with the EM dataset")

    ds = get_dataset(
        model=model,
        dataset=dataset,
        root=root,
        tile_size=tile_size,
        use_weight_map=use_weight_map,
    )
    datasets = train_val_dataset(ds, val_split=val_split)

    return DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=True,
    ), DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        num_workers=num_workers,
    )


__all__ = ["show", "display", "get_dataloaders"]


if __name__ == "__main__":
    import sys
    from model import Model

    dataset = sys.argv[1]

    assert dataset in ["em", "phc", "hela"]

    model = Model(n_classes=2)

    #matplotlib.use("module://matplotlib-backend-kitty")
    train_dl, _ = get_dataloaders(
        sys.argv[1],
        batch_size=2,
        tile_size=256,
        model=model,
        num_workers=0,
        use_weight_map=dataset == "em",
    )

    if dataset == "em":
        x, (y, weight_maps) = next(iter(train_dl))
        with display((x[0], y[0], weight_maps[0]), ground_truth_mask=y[0]) as fig:
            fig.savefig('em.png')
    else:
        x, y = next(iter(train_dl))
        with display((x[0], y[0]), ground_truth_mask=y[0]) as fig:
            fig.savefig(dataset)

