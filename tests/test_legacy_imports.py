"""Compatibility checks for the former ``quadsv`` namespace."""

import importlib

import pytest


def test_quadsv_namespace_forwards_to_sonic():
    sonic = importlib.import_module("sonic")

    with pytest.warns(DeprecationWarning, match="renamed to sonic"):
        quadsv = importlib.import_module("quadsv")

    assert quadsv.__version__ == sonic.__version__
    assert quadsv.Detector is sonic.Detector
    assert importlib.import_module("quadsv.statistics") is importlib.import_module(
        "sonic.statistics"
    )
    assert importlib.import_module("quadsv.kernels.fft") is importlib.import_module(
        "sonic.kernels.fft"
    )
