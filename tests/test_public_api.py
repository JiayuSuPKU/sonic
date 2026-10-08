"""Public-API freeze test — long-term guardrail for the four-layer
``sonic`` surface.

What this test enforces:

1. **Snapshot of ``sonic.__all__``.** Any addition or removal to the
   top-level exports forces a deliberate edit to ``EXPECTED_ALL``
   below, which surfaces in code review.
2. **Every public name imports + has a docstring.** Catches typos in
   ``__all__`` and missing documentation on new public symbols.
3. **Canonical-path identity.** Top-level re-exports resolve to the
   *same* object as their canonical path (e.g.
   ``sonic.spatial_q_test is sonic.statistics.spatial_q_test``).
   Guards against accidental re-export breakage during refactors.
"""

from __future__ import annotations

import importlib

import sonic

# ---------------------------------------------------------------------------
# Snapshot of the four-layer public surface. Any drift from this list — adds
# or removes — must be a deliberate edit reviewed alongside the change.
# Group order mirrors the package docstring (Kernels → Statistics →
# Detectors → Comparators → Factories).
# ---------------------------------------------------------------------------
EXPECTED_ALL: list[str] = [
    # Kernels
    "MatrixKernel",
    "FFTKernel",
    "NUFFTKernel",
    # Statistical tests
    "spatial_q_test",
    "spatial_q_test_fft_many",
    "spatial_r_test",
    # Detectors
    "DetectorIrregular",
    "DetectorGrid",
    # Cross-sample
    "ComparatorIrregular",
    "ComparatorGrid",
    # Factories
    "Detector",
    "Comparator",
]

# ABCs used only by backend authors. They live in ``sonic.kernels`` and
# are not part of the top-level public surface, so they must not show up
# in ``sonic.__all__`` or as attributes on the package.
_INTERNAL_BACKEND_ABCS: list[str] = ["Kernel", "MatrixKernelBase"]


def test_top_level_all_matches_snapshot():
    """``sonic.__all__`` matches ``EXPECTED_ALL`` (set comparison).

    Order doesn't matter; the *set* is the contract. Edit
    ``EXPECTED_ALL`` deliberately when the public surface changes.
    """
    assert set(sonic.__all__) == set(EXPECTED_ALL), (
        "sonic.__all__ drifted from the expected snapshot.\n"
        f"  added:   {sorted(set(sonic.__all__) - set(EXPECTED_ALL))}\n"
        f"  removed: {sorted(set(EXPECTED_ALL) - set(sonic.__all__))}"
    )


def test_every_public_name_resolves_and_documented():
    """Every name in ``sonic.__all__`` must resolve and carry a
    non-empty docstring."""
    for name in sonic.__all__:
        obj = getattr(sonic, name, None)
        assert obj is not None, f"{name} listed in __all__ but unresolved"
        doc = getattr(obj, "__doc__", None)
        assert doc and doc.strip(), f"{name} has no docstring"


# ---------------------------------------------------------------------------
# Canonical-path identity contract.
#
# Each top-level re-export must point at the same object as the
# canonical submodule path. If the re-export drifts (e.g. somebody
# accidentally rebinds the name in ``sonic.__init__``), tests still
# import the canonical class but the user-facing shortcut becomes
# stale; this test fails loudly.
# ---------------------------------------------------------------------------
_CANONICAL_PATHS: dict[str, tuple[str, str]] = {
    "spatial_q_test_fft_many": ("sonic.statistics", "spatial_q_test_fft_many"),
    # name on sonic: (submodule, attribute on submodule)
    "MatrixKernel": ("sonic.kernels", "MatrixKernel"),
    "FFTKernel": ("sonic.kernels.fft", "FFTKernel"),
    "NUFFTKernel": ("sonic.kernels.nufft", "NUFFTKernel"),
    "spatial_q_test": ("sonic.statistics", "spatial_q_test"),
    "spatial_r_test": ("sonic.statistics", "spatial_r_test"),
    "DetectorIrregular": ("sonic.detectors.irregular", "DetectorIrregular"),
    "DetectorGrid": ("sonic.detectors.grid", "DetectorGrid"),
    "ComparatorIrregular": ("sonic.comparators", "ComparatorIrregular"),
    "ComparatorGrid": ("sonic.comparators", "ComparatorGrid"),
    "Detector": ("sonic.api", "Detector"),
    "Comparator": ("sonic.api", "Comparator"),
}


def test_top_level_objects_identity_match_canonical_paths():
    """Every top-level re-export points at the same object as the
    canonical submodule path.
    """
    for name, (modpath, attr) in _CANONICAL_PATHS.items():
        top = getattr(sonic, name)
        canonical = getattr(importlib.import_module(modpath), attr)
        assert top is canonical, f"sonic.{name} drifted from {modpath}.{attr}"


def test_backend_abcs_are_not_top_level_public():
    """``Kernel`` and ``MatrixKernelBase`` are extension points for
    backend authors. They live at ``sonic.kernels`` and must not be
    accessible as top-level attributes on the ``sonic`` package.
    """
    for name in _INTERNAL_BACKEND_ABCS:
        assert name not in sonic.__all__, f"{name} should not be in sonic.__all__"
        assert not hasattr(sonic, name), (
            f"sonic.{name} should not be reachable on the top-level package; "
            f"import from sonic.kernels instead."
        )
    # ...but the canonical path is still importable.
    from sonic.kernels import Kernel, MatrixKernelBase  # noqa: F401
