# -*- coding: utf-8 -*-

import ryn
from ryn.common import helper

import yaml

from functools import partial
from collections import defaultdict

from typing import Any
from typing import Dict
from typing import Sequence


def load(*, configs: Sequence[str], **kwargs):
    """

    Load and join configurations from yaml and kwargs

    """

    if not configs and not kwargs:
        raise ryn.RynError('no configuration provided')

    as_path = partial(
        helper.path, exists=True,
        message='loading {path_abbrv}')

    # first join all yaml configs into one dictionary;
    # later dictionaries overwrite earlier ones

    def _update(d1, d2):
        for k, v in d2.items():
            if v is None:
                continue

            if k not in d1 or type(v) is not dict:
                d1[k] = v

            else:
                _update(d1[k], d2[k])

    result = {}
    for path in map(as_path, configs):
        with path.open(mode='r') as fd:
            _update(result, yaml.load(fd))

    # then join all kwargs;
    # this is practically the reverse of what
    # print_click_arguments does
    sep = '__'
    dic = defaultdict(dict)
    for k, v in kwargs.items():
        if sep in k:
            # only two levels deep
            k_head, k_tail = k.split(sep)
            dic[k_head][k_tail] = v
        else:
            dic[k] = v

    _update(result, dic)
    return result


@helper.notnone
def print_click_arguments(*, dic: Dict[str, Any] = None) -> str:

    def _resolve(k, v):
        name = k.replace('_', '-')
        if type(v) is dict:
            for it in v.items():
                for subname, argtype in _resolve(*it):
                    yield f'{name}--{subname}', argtype
        else:
            yield name, type(v).__name__

    opts = []
    for it in dic.items():
        for name, argtype in _resolve(*it):
            opts.append((name, argtype))

    for name, argtype in opts:
        print(f"@click.option('--{name}', type={argtype})")
