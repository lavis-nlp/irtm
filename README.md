# IRTM

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Inductive Reasoning with Text - Models

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [IRTM](#irtm)
    - [Installation](#installation)
    - [Downloads and more](#downloads-and-more)
    - [Closed-World Knowledge Graph Completion](#closed-world-knowledge-graph-completion)
        - [Training](#training)
        - [Evaluation](#evaluation)
    - [Open-World Knowledge Graph Completion](#open-world-knowledge-graph-completion)
        - [Command Line Interface](#command-line-interface)
        - [Training](#training-1)
    - [Legacy Download](#legacy-download)

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


## Downloads and more

You can see the different validation/test results [in the
spreadsheet](https://bit.ly/3rsR20x-IRT-spreadsheet). For more
training insights see the Weights&Biases result trackers for
[closed-world KGC](https://bit.ly/3ruaiuQ-IRT-KGC) and [Mapper
training](https://bit.ly/3roPmFq-IRT-Text). You can find a selection
of these models in the [legacy download section](#legacy-download)
below (they use the pre-refactoring code). These models have been
trained with the new code (see, for example, `irtm.text.`):

| Version | Text   | Mapper | Contexts | Download                                                                        |
|---------|--------|--------|----------|---------------------------------------------------------------------------------|
| IRT-CDE | masked | multi  | 30       | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.cde.30.multi-cls.masked.tgz) |
| IRT-FB  | masked | multi  | 30       | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.fb.30.multi-cls.masked.tgz)  |


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
  --dataset ../data/irt/irt.cde \
  --out data/kgc/irt-cde/distmult.sweep
```

This particular configuration starts a hyperparameter sweep (defining
ranges/sets for the parameter space). If you want to have multiple
instances (i.e. multiple gpus) train in parallel for the same sweep,
simply invoke the same command adding `--participate`:

``` bash
irtm kgc train \
  --config conf/kgc/irt.cde.distmult-sweep.yml \
  --dataset ../irt/data/irt/irt.cde \
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

```
irtm text cli --dataset ../irt/data/irt/irt.cde --model bert-base-cased [--mode masked]

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

[1] dl = tdm.train_dataloader()
[2] gen = iter(dl)
[3] next(gen)
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

```
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

```
irtm text train \
    -c conf/text/irt/defaults.yml \
    -c conf/text/irt/early_stopping.30.yml \
    -c conf/text/irt/cde.gpu.24g.train.30.yml \
    -c conf/text/irt/cde.yml \
    -c conf/text/irt/exp.m02.yml \
    --optimizer-args--lr 0.00005 \
    --mode masked
```


### Evaluation

To evaluate the trained model, run any of the `irtm text evaluate*`
commands. For example, to evaluate a single checkpoint, `irtm
evaluate` requires the following parameters:

```
irtm text evaluate --help
Usage: irtm text evaluate [OPTIONS]

  Evaluate a mapper on the test split

Options:
  --path TEXT        path to model directory  [required]
  --checkpoint TEXT  path to model checkpoint  [required]
  -c, --config TEXT  one or more configuration files
  --debug            run everything fast, do not write anything
  --help             Show this message and exit.
```

So, for a trained model that is inside folder `$dir`:

```
irtm text evaluate \
  --path $dir \
  --checkpoint $dir/weights/.../epoch=...ckpt \
  -c $dir/config.yml \
  --debug
```

This writes the evaluation results to a yaml file with a name
according to the provided checkpoint. For example:

``` bash
cat $dir/report/evaluation.epoch=53-step=61559.ckpt.yml | grep -E 'transductive|inductive|test|both.realistic.hits_at_10'
  inductive:
    both.realistic.hits_at_10: 0.4268671193016489
  test:
    both.realistic.hits_at_10: 0.42410341951626357
  transductive:
    both.realistic.hits_at_10: 0.37879945846798846
```


## Legacy Download

Selection of original models. You need the legacy datasets that can be
found in [the IRT repository](https://github.com/lavis-nlp/irt). The
code version required to load this data and these models is
[here](https://github.com/lavis-nlp/irtm/tree/157df680f9ee604b43a13581ab7de45d40ac81d6). Contact
me, if you need other models (see the
[spreadsheet](https://bit.ly/3rsR20x-IRT-spreadsheet)) just drop me a
message and I will extend this table:


| Version | Text   | Mapper | Contexts | Download                                                                                |
|---------|--------|--------|----------|-----------------------------------------------------------------------------------------|
| IRT-CDE | masked | single | 1        | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.legacy.cde.1.single-cls.masked.tgz)  |
| IRT-CDE | masked | multi  | 30       | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.legacy.cde.30.multi-cls.masked.tgz)  |
| IRT-CDE | masked | single | 30       | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.legacy.cde.30.single-cls.masked.tgz) |
| IRT-FB  | masked | single | 1        | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.legacy.fb.1.single-cls.masked.tgz)   |
| IRT-FB  | masked | multi  | 30       | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.legacy.fb.30.multi-cls.masked.tgz)   |
| IRT-FB  | masked | single | 30       | [Link](http://lavis.cs.hs-rm.de/storage/irt/mapper.legacy.fb.30.single-cls.masked.tgz)  |
