#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import pretty_errors

import logging
import unittest

from irtm.common import logger

logger.init()
log = logging.getLogger(__name__)


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


@click.group()
def main():
    """
    IRTM - working with texts and graphs
    """
    log.info(" · IRTM CLI ·")
    log.info(f"initialized path to irtm: {irtm.ENV.ROOT_DIR}")


# tests are distributed over submodules
# entry point is registered here
@main.group(name="tests")
def click_tests():
    """
    Run unit tests
    """
    pass


@click_tests.command()
@click.option(
    "--name",
    type=str,
    help="test case(s) to run (e.g. graphs.tests.test_graph.FindTests)",
)
def run(name: str = None):
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    tests = loader.discover(irtm.ENV.SRC_DIR)

    if name:
        tests = loader.loadTestsFromName(name)

    runner.run(tests)


# registered modules (see their respective __init__.py)
# not a super nice solution, but it works well

import irtm.common  # noqa: E402
import irtm.kgc  # noqa: E402
import irtm.text  # noqa: E402
