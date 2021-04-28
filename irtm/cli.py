#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click
import unittest
import pretty_errors


pretty_errors.configure(
    filename_display=pretty_errors.FILENAME_EXTENDED,
    lines_after=2,
    line_number_first=True,
)


@click.group()
def main():
    """
    RYN - working with texts and graphs
    """
    from ryn.common import logging

    log = logging.get("cli")

    log.info(" · RYN CLI ·")
    log.info(f"initialized path to ryn: {ryn.ENV.ROOT_DIR}")


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
    tests = loader.discover(ryn.ENV.SRC_DIR)

    if name:
        tests = loader.loadTestsFromName(name)

    runner.run(tests)


# registered modules (see their respective __init__.py)
# not a super nice solution, but it works well

import ryn.common  # noqa: E402
import ryn.kgc  # noqa: E402
import ryn.text  # noqa: E402
import ryn.graphs  # noqa: E402
