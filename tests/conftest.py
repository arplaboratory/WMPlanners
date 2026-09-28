"""Pytest options shared by the project test suite."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--pipeline-out",
        default=None,
        metavar="DIR",
        help="Also test a local dataset generated under DIR by the pipeline",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--pipeline-out"):
        return
    skip_local = pytest.mark.skip(reason="requires --pipeline-out")
    for item in items:
        if "local_data" in item.keywords:
            item.add_marker(skip_local)
