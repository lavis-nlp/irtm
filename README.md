# RŶN

> But no wizardry nor spell, neither fang nor venom, nor devil's art
> nor beast-strength, could overthrow Huan
> utterly.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

```bash
conda create --name ryn python=3.8
pip install -r requirements.txt
pip install -e .
```

You also need to install pytorch. For multi-gpu support, you need horovod.


## Command Line Client

```
 > ryn --help
Usage: ryn [OPTIONS] COMMAND [ARGS]...

  RYN - working with texts and graphs

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
 > ryn graphs --help
Usage: ryn graphs [OPTIONS] COMMAND [ARGS]...

  Working with graphs

Options:
  --help  Show this message and exit.

Commands:
  graph  Networkx graph abstractions
  split  Create open world triple splits
```


## Knowledge Graph Completion

The `ryn.kgc` module offers kgc functionality on top of
[pykeen](https://github.com/pykeen/pykeen).


### Training

You need a split dataset (see `ryn.graphs.split.Dataset`) and
configuration file (see `conf/kgc/*.json`). Models are trained by
simply providing these two arguments:

```
ryn kgc train \
  --config conf/kgc/complex-sweep.json \
  --split-dataset data/split/oke.fb15k237_30061990_50
```

This writes the results to `data/kgc/<DATASET>/<MODEL>-<TIMESTAMP>`
(where `data/kgc` is set by `ryn.ENV.KGC_DIR` in `ryn/__init__.py`)


### Evaluation

The evaluation is run seperately from the training (to avoid the
temptation to tune your hyperparameters on the test data ;) ). It is
possible to provide multiple directories which are evaluated
successively:

```
dataset=data/split/oke.fb15k237_30061990_50
sweep=data/kgc/oke.fb15k237_30061990_50/DistMult-2020-10-22_10:58:50.325146
ryn kgc evaluate --out $sweep --split_dataset $dataset $sweep
```
