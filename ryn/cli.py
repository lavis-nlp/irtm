#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ryn.common import logging

import click


log = logging.get('cli')


@click.group()
def main():
    """
    RYN - working with texts and graphs
    """
    log.info(' · RYN CLI ·')
    log.info(f'initialized path to ryn: {ryn.ENV.ROOT_DIR}')


# registered modules (see their respective __init__.py)
# not a super nice solution, but it works well

import ryn.app     # noqa: F401, E402
import ryn.kgc     # noqa: F401, E402
import ryn.text    # noqa: F401, E402
import ryn.tests   # noqa: F401, E402
import ryn.graphs  # noqa: F401, E402
