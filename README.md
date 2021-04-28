# IRTM

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

With horovod multi-gpu support:
```bash
# sets environment variables and invokes
# conda env create -f environment.yml
fish environment.fish
```

Without multi-gpu support:

```bash
conda create --name irtm python=3.8
conda install pytorch
pip install -r requirements.txt
pip install -e .
```


## Command Line Client

```
 > irtm --help
Usage: irtm [OPTIONS] COMMAND [ARGS]...

  IRTM - working with texts and graphs

Options:
  --help  Show this message and exit.

Commands:
  graphs     Working with graphs
  kgc        Knowledge graph completion models
  streamlit  Run a streamlit app instance
  tests      Run unit tests
  text       Process text data
```

To get more specific information, each subcommand also offers help:

```
 > irtm graphs --help
Usage: irtm graphs [OPTIONS] COMMAND [ARGS]...

  Working with graphs

Options:
  --help  Show this message and exit.

Commands:
  graph  Networkx graph abstractions
  split  Create open world triple splits
```


## Knowledge Graph Completion

The `irtm.kgc` module offers kgc functionality on top of
[pykeen](https://github.com/pykeen/pykeen).


### Training

You need a split dataset (see `irtm.graphs.split.Dataset`) and
configuration file (see `conf/kgc/*.json`). Models are trained by
simply providing these two arguments:

```
irtm kgc train \
  --config conf/kgc/complex-sweep.json \
  --split-dataset data/split/oke.fb15k237_30061990_50
```

This writes the results to `data/kgc/<DATASET>/<MODEL>-<TIMESTAMP>`
(where `data/kgc` is set by `irtm.ENV.KGC_DIR` in `irtm/__init__.py`)


### Evaluation

The evaluation is run seperately from the training (to avoid the
temptation to tune your hyperparameters on the test data ;) ). It is
possible to provide multiple directories which are evaluated
successively:

```
dataset=data/split/oke.fb15k237_30061990_50
sweep=data/kgc/oke.fb15k237_30061990_50/DistMult-2020-10-22_10:58:50.325146
irtm kgc evaluate --out $sweep --split_dataset $dataset $sweep
```
