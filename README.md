## Reproducing U-Net

This is the code for [this blogpost](https://doubledissent.bearblog.dev/reproducing-u-net/) detailing my journey attempting to reproduce the [U-Net paper](https://arxiv.org/pdf/1505.04597) from 2015.

## Setup

```bash
uv venv
source .venv/bin/activate
python -m ensurepip
pip3 install -r requirements.txt
pip3 install -r requirements.dev.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Training

Check out / change hyperparams in `src/train.py`, as well as the wandb project / organization.

```bash
uv run python src/train.py
```

## Datasets

All datasets are vendored here, but due credit and lineage is below.

To preview a dataloader (with transforms and so on):


```bash
PYTHONPATH=src uv run python src/dataset/__init__.py phc # phc, em, or hela
```

### ISBI 2012 EM Segmentation Dataset


The dataset contains 30 ssTEM (serial section Transmission Electron Microscopy) images taken from the Drosophila larva ventral nerve cord (VNC). The images represent a set of consecutive slices within one 3D volume. Corresponding segmentation ground truths are also provided in this dataset.

[Original URL](http://brainiac2.mit.edu/isbi_challenge) is down (archived version is [available here](https://web.archive.org/web/20200605014329/http://brainiac2.mit.edu/isbi_challenge/)), but Hoang Pham [uploaded it to Github](https://github.com/hoangp/isbi-datasets).

As far as I can dig, the dataset had originally been published [in this paper](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1000502).

### PHC-C2DH-U373 [Trainset URL](https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip)

Glioblastoma-astrocytoma U373 cells on a polyacrylamide substrate

Dr. S. Kumar. Department of Bioengineering, University of California at Berkeley, Berkeley CA (USA)

### DIC-C2DH-HeLa [Trainset URL](https://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip)

HeLa cells on a flat glass

Dr. G. van Cappellen. Erasmus Medical Center, Rotterdam, The Netherlands

