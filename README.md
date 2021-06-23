# IRTM

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Inductive Reasoning with Text - Models

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [IRTM](#irtm)
    - [Installation](#installation)
    - [Current Performance](#current-performance)
    - [Closed-World Knowledge Graph Completion](#closed-world-knowledge-graph-completion)
        - [Training](#training)
        - [Evaluation](#evaluation)
    - [Open-World Knowledge Graph Completion](#open-world-knowledge-graph-completion)

<!-- markdown-toc end -->


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


## Current Performance

TODO


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
  data/kgc/irt-cde/distmult.sweep/trial-*
```


## Open-World Knowledge Graph Completion

The `irtm.text` module offers training for the text projector. You
need to have a closed world KGC model trained with the `irtm.kgc`
module as described [here](#closed-world-knowledge-graph-completion).

``` bash
irtm text --help
Usage: irtm text [OPTIONS] COMMAND [ARGS]...

  Open-world knowledge graph completion using free text

Options:
  --help  Show this message and exit.

Commands:
  cli                Open an interactive python shell dataset: path to...
  evaluate           Evaluate a mapper on the test split
  evaluate-all       Run evaluations for all saved checkpoints
  evaluate-csv       Run evaluations based on a csv file
  resume             Resume training of a mapper
  train              Train a mapper to align embeddings
```

### Command Line Interface

If you just want to play around a little bit and understand the
datamodule, you can spawn an interactive ipython shell with the `cli`
command:

``` bash
irtm text cli --dataset ../irt/data/irt/irt.cde --model bert-base-cased

IRT dataset:
IRT graph: [irt-cde] (17050 entities)
IRT split: closed_world=137388 | open_world-valid=41240 | open_world-test=27577
irt text: ~24.71 text contexts per entity

keen dataset: [irt-cde]: closed world=137388 | open world validation=41240 | open world testing=27577

--------------------
 IRTM KEEN CLIENT
--------------------

variables in scope:
    ids: irt.Dataset
    kow: irt.KeenOpenworld
    tdm: irt.TorchModule

you can now play around, e.g.:
  dl = tdm.train_dataloader()
  gen = iter(dl)
  next(gen)
```


### Training

Training a mapper requires some configuration. You can find the
configuration options extensively documented in
`conf/text/defaults.yml`. The configuration used for the experiments
documented in the paper is composed of the files in
`conf/text/irt*`. You can pass an arbitrary amount of yml files via
the `-c` parameter and the final configuration is created based on
this sequence. Later configurations overwrite former ones. Single options
can also be set directly via command line flag:

``` bash
irtm text train --help
Usage: irtm text train [OPTIONS]

  Train a mapper to align embeddings

Options:
  --debug                         only test a model and do not log
  -c, --config TEXT               one or more configuration files
  --valid-split FLOAT
  --wandb-args--project TEXT
  --wandb-args--log-model BOOLEAN
  --trainer-args--gpus INTEGER
  --trainer-args--max-epochs INTEGER
  --trainer-args--fast-dev-run BOOLEAN
  (...) etc
```


We leave the configurations we used for the experiments as is for
documentation. You certainly don't need to have it flexible like this
and you can provide a single configuration file of course. But, for
example, to train a 30-sentence multi-context mapper that uses an
early stopper on a 24GB RAM GPU on IRT-CDE while overwriting the
learning rate, you can combine the configuration like this:

``` bash
irtm text train \
    -c conf/text/irt/defaults.yml \
    -c conf/text/irt/early_stopping.30.yml \
    -c conf/text/irt/cde.gpu.24g.train.30.yml \
    -c conf/text/irt/cde.yml \
    -c conf/text/irt/exp.m02.yml \
    --optimizer-args--lr 0.00005
```

