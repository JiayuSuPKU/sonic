"""
sonic: kernel-based spatial pattern detection and comparison for spatial omics.

The public top-level API is organised in four layers:

1. **Kernels** — :class:`MatrixKernel` (dense / sparse), :class:`FFTKernel`
   (regular grid), :class:`NUFFTKernel` (irregular 2D coordinates). The
   :class:`~sonic.kernels.Kernel` and
   :class:`~sonic.kernels.MatrixKernelBase` ABCs live in
   :mod:`sonic.kernels` and are intended for backend authors.
2. **Statistical tests** — :func:`spatial_q_test` and :func:`spatial_r_test`.
   A single entry point per test dispatches on the kernel type (matrix, FFT,
   or NUFFT). Signature: ``(x, kernel, null_params=None, return_pval=True,
   is_standardized=False)``.
3. **Detectors** — :class:`DetectorIrregular` consumes :class:`anndata.AnnData`
   (irregular grids, matrix/NUFFT backends); :class:`DetectorGrid` consumes
   :class:`spatialdata.SpatialData` (regular grids, FFT backend).
4. **Comparators** — cross-sample pattern comparison:
   :class:`ComparatorIrregular` on a list of AnnData (NUFFT backend);
   :class:`ComparatorGrid` on a list of SpatialData (FFT backend).
"""

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Version resolution order: prefer the file written by ``setuptools-scm`` at
# build time (``src/sonic/_version.py`` — see ``[tool.setuptools_scm]`` in
# ``pyproject.toml``), fall back to installed-package metadata, then to a
# last-known release string for unbuilt / shallow-clone checkouts.
try:
    from sonic._version import version as __version__  # type: ignore[assignment]
except ImportError:  # _version.py absent — source checkout without a build step
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("sonic")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0+unknown"

from sonic.api import Comparator, Detector
from sonic.comparators import ComparatorGrid, ComparatorIrregular
from sonic.detectors.grid import DetectorGrid
from sonic.detectors.irregular import DetectorIrregular
from sonic.kernels import MatrixKernel
from sonic.kernels.fft import FFTKernel
from sonic.kernels.nufft import NUFFTKernel
from sonic.statistics import spatial_q_test, spatial_r_test

# The :class:`~sonic.kernels.Kernel` and
# :class:`~sonic.kernels.MatrixKernelBase` ABCs are intentionally not
# re-exported here. They are extension points for backend authors and
# live at ``sonic.kernels`` (the canonical path). Importing them through
# ``sonic`` directly is unsupported.

__all__ = [
    # Kernels
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
    # Statistical tests
    "spatial_q_test",
    "spatial_r_test",
    # Detectors
    "DetectorIrregular",
    "DetectorGrid",
    # Cross-sample
    "ComparatorIrregular",
    "ComparatorGrid",
    # Factories — type-dispatched discovery face on the four classes above
    "Detector",
    "Comparator",
]
