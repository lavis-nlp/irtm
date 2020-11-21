#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import click
import pretty_errors  # noqa: F401


@click.group()
def main():
    """
    RYN - working with texts and graphs
    """
    from ryn.common import logging
    log = logging.get('cli')

    log.info(' · RYN CLI ·')
    log.info(f'initialized path to ryn: {ryn.ENV.ROOT_DIR}')


# registered modules (see their respective __init__.py)
# not a super nice solution, but it works well

import ryn.common  # noqa: F401, E402
import ryn.app     # noqa: F401, E402
import ryn.kgc     # noqa: F401, E402
import ryn.text    # noqa: F401, E402
import ryn.tests   # noqa: F401, E402
import ryn.graphs  # noqa: F401, E402
