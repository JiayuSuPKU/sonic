"""
``sonic.kernels`` — spatial-kernel layer.

Subpackage grouping the public kernel classes plus the ABCs that
backend authors subclass:

- :class:`Kernel` (ABC) — universal interface.
- :class:`MatrixKernelBase` (ABC) — matrix-form base with dense /
  sparse / sparse-precision auto-switching.
- :class:`MatrixKernel` — standard concrete matrix kernel.
- :class:`FFTKernel` — regular-grid FFT-accelerated kernel.
- :class:`NUFFTKernel` — irregular-coordinate NUFFT-accelerated
  kernel.

All five are importable from this subpackage:

    from sonic.kernels import FFTKernel, NUFFTKernel, MatrixKernel
    from sonic.kernels import Kernel, MatrixKernelBase  # for backend authors

The three concrete classes are also re-exported at the top of the
:mod:`sonic` namespace.
"""

from sonic.kernels.base import Kernel, MatrixKernelBase
from sonic.kernels.fft import FFTKernel
from sonic.kernels.matrix import MatrixKernel
from sonic.kernels.nufft import NUFFTKernel

__all__ = [
    "Kernel",
    "MatrixKernelBase",
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
]
