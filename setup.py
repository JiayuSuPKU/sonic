"""Build the native Davies evaluator; project metadata lives in pyproject.toml."""

import sys

from setuptools import Extension, setup

if __name__ == "__main__":
    setup(
        ext_modules=[
            Extension(
                "sonic._davies",
                sources=["src/sonic/_davies.c", "src/sonic/_vendor/chi2comb.c"],
                depends=["src/sonic/_vendor/chi2comb.h"],
                libraries=[] if sys.platform == "win32" else ["m"],
            )
        ]
    )
