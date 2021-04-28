# -*- coding: utf-8 -*-
# fmt: off

from irtm.cli import main
from irtm.common import ryaml

import click

from typing import List


@main.group()
def common():
    """
    Commonly used things
    """
    pass


@common.group(name='ryaml')
def click_ryaml():
    """
    Process yaml files
    """
    pass


@click_ryaml.command('click-arguments')
@click.option(
    '-c', '--config', type=str, multiple=True,
    help='one or more configuration files')
def ryaml_create_click_arguments(config: List[str] = None):
    """
    Generate click arguments from yaml files
    """
    dic = ryaml.load(configs=config)
    ryaml.print_click_arguments(dic=dic)
