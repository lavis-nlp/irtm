# RŶN

> But no wizardry nor spell, neither fang nor venom, nor devil's art
> nor beast-strength, could overthrow Huan without forsaking his body
> utterly.


## Installation

```bash
conda create --name ryn python=3.8
pip install -r requirements.txt
pip install -e .
```

## Run Command Line Client

```
 > ryn --help
RYN - working with knowledge graphs

usage: ryn CMD [ARGS]

  possible values for CMD:
       help: print this message
        app: handle streamlit instances
      tests: run some tests
     graphs: analyse paths through the networks
     embers: work with graph embeddings

to get CMD specific help type ryn CMD --help
e.g. ryn embers --help
```

Detailed informations

```
 > ryn graphs --help
attempting to run "graphs"
usage: ryn [-h] [--seeds SEEDS [SEEDS ...]] [--ratios RATIOS [RATIOS ...]] [-c CONFIG] [-s SPEC]
           [-g GRAPHS [GRAPHS ...]] [--uri URI [URI ...]]
           _ cmd subcmd

positional arguments:
  _
  cmd                   one of graph, split
  subcmd                graph: (cli),

optional arguments:
  -h, --help            show this help message and exit
  --seeds SEEDS [SEEDS ...]
                        random seeds
  --ratios RATIOS [RATIOS ...]
                        ratio thresholds (cut at n-th relation for concepts)
  -c CONFIG, --config CONFIG
                        config file (conf/*.conf)
  -s SPEC, --spec SPEC  config specification file (conf/*.spec.conf)
  -g GRAPHS [GRAPHS ...], --graphs GRAPHS [GRAPHS ...]
                        selection of graphs (names defined in --config)
  --uri URI [URI ...]   instead of -c -s -g combination
```


For example, to run streamlit, simply type: `ryn app`
