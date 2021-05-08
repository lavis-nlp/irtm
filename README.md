# IRTM

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Installation

We highly recommend using
[miniconda](https://docs.conda.io/en/latest/miniconda.html) for python
version control. All requirements and self-installation is defined in
the `requirements.txt`.


```bash
conda create --name irtm python=3.8
conda install pytorch
pip install -r requirements.txt
```

There is now a command-line client installed: `irtm`. It handles the
entry points for both [pykeen](https://github.com/pykeen/pykeen) based
closed-world knowledge graph completion (`kgc`) and open-world kgc
using a [huggingface BERT
transformer](https://github.com/huggingface/transformers) trained
using
[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning). Each
entry point is defined in the modules' `__init__.py`. If you do not
like to use the CLI, you can look there to see the associated API
entry point (e.g. for `irtm kgc train`, the `irtm.kgc.__init__.py`
invokes `irt.kgc.trainer.train_from_kwargs`, which in turn calls
`irt.trainer.train`). The whole project follows this convention.


```bash
> irtm --help
Usage: irtm [OPTIONS] COMMAND [ARGS]...

  IRTM - working with texts and graphs

Options:
  --help  Show this message and exit.

Commands:
  kgc   Closed-world knowledge graph completion
  text  Open-world knowledge graph completion using free text
```

A log file is written to `data/irtm.log` when using the CLI. You can
configure the logger using `conf/logging.conf`.


## Closed-World Knowledge Graph Completion

The `irtm.kgc` module offers kgc functionality on top of
[pykeen](https://github.com/pykeen/pykeen).


``` bash
 > irtm kgc train --help
Usage: irtm kgc train [OPTIONS]

  Train a knowledge graph completion model

Options:
  --config TEXT             yaml (see conf/kgc/*yml)  [required]
  --dataset TEXT            path to irt.Dataset folder  [required]
  --participate / --create  for multi-process optimization
  --help                    Show this message and exit.
```

### Training

You need an IRT dataset (see `irt.Dataset`) and configuration file
(see `conf/kgc/*.yml`). Models are trained by simply providing these
two arguments:

```bash
irtm kgc train \
  --config conf/kgc/irt.cde.distmult-sweep.yml \
  --split-dataset data/irt/irt.cde \
  --out data/kgc/irt-cde/distmult.sweep
```

This particular configuration starts a hyperparameter sweep (defining
ranges/sets for the parameter space). If you want to have multiple
instances (i.e. multiple gpus) train in parallel for the same sweep,
simply invoke the same command adding `--participate`:

``` bash
irtm kgc train \
  --config conf/kgc/irt.cde.distmult-sweep.yml \
  --split-dataset data/irt/irt.cde \
  --out data/kgc/irt-cde/distmult.sweep \
  --participate
```

To employ the hyperparameter configuration used for the model
described in the paper, use the associated `*-best.yml` files.


### Evaluation

To evaluate a trained model, use the `irtm kgc evaluate` command. This
expects one or many directories containing trained models (e.g. all
models of a sweep), runs an evaluation on one of the dataset's splits
(e.g. "validation") and saves the results to a file:

``` bash
irtm kgc evaluate \
  --dataset ../irt/data/irt/irt.cde \
  --out data/kgc/irt-cde/distmult.sweep \
  data/kgc/irt-cde/distmult.sweep/trial-00*
```

