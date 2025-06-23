import random
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torchsummary import summary

import wandb
from dataset import display, get_dataloaders, show
from model import Model

run = None

config = {
    "bs": 4,
    "lr": 3e-04,
    "max_steps": 4_000,
    "tile_size": 512,
    "weight_map": True,
    "dropout": 0.1,
    "eval_every": 200,
    "val_split": 0.2,
    "optimizer": "adam",
    "dataset": "em",
    "compile": True,
    "log": True,
}

assert config["dataset"] in ["em", "phc", "hela"], "invalid dataset"

if config["dataset"] != "em":
    config["weight_map"] = False

device = torch.device("cuda")

run = None
if config["log"]:
    run_name = "_".join(
        [x for x in [
            f"{config['dataset']}",
            f"tileSize{config['tile_size']}",
            (("weightMap" if config["weight_map"] else "noWeightMap") if config["dataset"] == 'em' else None),
            f"b{config['bs']}",
            f"dropout{config['dropout']}",
            f"lr{config['lr']}",
            config["optimizer"],
        ] if x is not None]
    )

    run = wandb.init(
        entity="deeplearning-artificialintelligence-2023",
        project="unet",
        name=run_name,
        config=config,
    )


@torch.compile
def criterion(pred, targets, weight_maps):
    loss_per_pixel = F.nll_loss(pred, targets, reduction="none")
    weighted_loss = loss_per_pixel * weight_maps
    return weighted_loss.mean()


step = 0


def log(payload, commit=False):
    if run:
        run.log(payload, step=step, commit=commit)


model = Model(n_classes=2, dropout=config["dropout"])

if compile:
    model = torch.compile(model)

_train_dl, val_dl = get_dataloaders(
    config["dataset"],
    config["bs"],
    tile_size=config["tile_size"],
    model=model,
    val_split=config["val_split"],
    num_workers=8,
    use_weight_map=config["weight_map"],
)


def infinitely(dl):
    while True:
        for batch in iter(dl):
            yield batch


train_dl = infinitely(_train_dl)

if not compile:
    the_batch = next(train_dl)
    inputs, _ = the_batch
    summary(model=model, input_size=inputs[0].shape, device="cpu")

model.to(device)

if config["optimizer"] == "adam":
    optim = Adam(model.parameters(), lr=config["lr"])
elif config["optimizer"] == "sgd":
    optim = SGD(model.parameters(), lr=config["lr"], momentum=0.99)  # as per the paper
else:
    raise NotImplementedError("optimizer must be adam or sgd")


def compute_iou(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-6
    intersection = (
        (outputs & labels).float().sum((1, 2))
    )  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0
    iou = (intersection + SMOOTH) / (
        union + SMOOTH
    )  # We smooth our devision to avoid 0/0
    thresholded = (
        torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10
    )  # This is equal to comparing with thresolds
    return thresholded.mean()


def estimate_loss(model, split):
    def forward(model, batch):
        if config["weight_map"]:
            inputs, (targets, weight_maps) = batch
            weight_maps = weight_maps.to(device)
        else:
            inputs, targets = batch
            weight_maps = 1.0

        inputs = inputs.to(device)
        targets = targets.to(device).squeeze(dim=1)  # channel

        pred = F.log_softmax(model(inputs), dim=1)  # channel

        loss = criterion(pred, targets, weight_maps)
        return loss, inputs, targets, weight_maps, pred

    if split == "train":
        now = time.time()
        batch = next(train_dl)
        loss, _, _, _, _ = forward(model, batch)

        elapsed_s = time.time() - now

        kpixels_per_sec = int((
            elapsed_s
            / config["bs"]
            * config["tile_size"]
            * config["tile_size"]
        ) / 1024)

        return loss, kpixels_per_sec
    else:
        with torch.no_grad():
            val_losses = []
            elements = []
            ious = []

            now = time.time()

            batches = 0

            for batch in val_dl:
                batches += 1

                loss, inputs, targets, weight_maps, preds = forward(
                    model, batch
                )

                if not isinstance(weight_maps, torch.Tensor):
                    weight_map = None
                else:
                    weight_map = weight_maps.cpu()[0]

                val_losses.append(loss)
                outs = torch.argmax(preds, dim=1)
                ious.append(compute_iou(outs, targets.squeeze(dim=1)))
                elements.append(
                    (
                        inputs.cpu()[0],
                        targets.cpu()[0],
                        weight_map,
                        preds.cpu()[0],
                    )
                )

            elapsed_s = time.time() - now

            kpixels_per_sec = int(elapsed_s / (
                batches * config["bs"] * config["tile_size"] * config["tile_size"]
            ) / 1024)

            random.shuffle(elements)  # select a random thing to visualize

            return (
                torch.mean(torch.tensor(val_losses)).item(),
                torch.mean(torch.tensor(ious)).item(),
                elements[0],
                kpixels_per_sec,
            )


done = False


while step < config["max_steps"]:
    step += 1

    # perform update!

    optim.zero_grad()

    loss, kpixels_per_sec = estimate_loss(model, "train")

    log({"train/loss": loss.cpu().item(), "train/kpixels_per_sec": kpixels_per_sec})
    print("step", step, "loss", loss.item(), "kpixels/sec", kpixels_per_sec)

    loss.backward()
    optim.step()

    # eval!

    if step == 1 or step % config["eval_every"] == 0:
        with torch.no_grad():
            loss, iou, (input, target, weight_map, pred), kpixels_per_sec = (
                estimate_loss(model, "val")
            )

            print("step", step, "eval IOU", iou, "kpixels/sec", kpixels_per_sec)

            out = torch.argmax(pred, dim=0).cpu()

            if config["weight_map"]:
                todisplay = (input, out.unsqueeze(0), weight_map)
            else:
                todisplay = (input, out.unsqueeze(0))

            if config["log"]:
                with display(todisplay, ground_truth_mask=target.unsqueeze(0)) as fig:
                    log(
                        {
                            "visual_eval": fig,
                            "val/loss": loss,
                            "val/iou": iou,
                            "val/kpixels_per_sec": kpixels_per_sec,
                        },
                        commit=True,
                    )
            else:
                show(todisplay, ground_truth_mask=target.unsqueeze(0))
