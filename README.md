# SONIC: Spatial Organization through Nonrandom-pattern Inference and Comparison

SONIC detects spatial patterns within one sample and compares patterns across
multiple samples. It works with coordinates, graphs, and regular grids stored
in `AnnData` or `SpatialData`.

The two main workflows are:

- **Pattern detection:** find spatially variable features with the Q-test and
  spatially co-expressed feature pairs with the R-test.
- **Pattern comparison:** find features whose spatial pattern changes between
  samples or conditions, without registering the samples to one another.

## Installation

Install the release candidate from PyPI:

```bash
pip install sonic-spatial==1.0.0rc2
```

The distribution name is `sonic-spatial`; the Python import is `sonic`.

For the latest source:

```bash
pip install "sonic-spatial @ git+https://github.com/JiayuSuPKU/sonic.git"
```

Code written for the former `quadsv` package remains supported through the
compatibility distribution:

```bash
pip install quadsv==1.0.0rc2
```

## Detect patterns in one sample

`Detector` reads an `AnnData` object with coordinates or a graph and tests all
features in the expression matrix. Start with NUFFT and a Matérn kernel for
two-dimensional spatial coordinates. Set `bandwidth` to the spatial scale of
interest, in the same units as the coordinates.

```python
import anndata as ad
from sonic import Detector

adata = ad.read_h5ad("spatial_data.h5ad")

detector = Detector(
    adata,
    kernel_method="matern",
    backend="nufft",
    bandwidth=2.0,
    nu=1.5,
).setup_data(adata, obsm_key="spatial", min_cells_frac=0.05)

q_results = detector.compute_qstat(return_pval=True)
svgs = q_results[q_results["P_adj"] < 0.05]

top_genes = q_results.nlargest(100, "Q").index.tolist()
r_results = detector.compute_rstat(
    features_x=top_genes,
    features_y=None,
    return_pval=True,
)
```

Use `spatial_q_test` and `spatial_r_test` directly when you already have a
kernel and want to test one feature or one feature pair. Regular rasterized
grids, including Visium HD bins, use the same `Detector` interface with a
`SpatialData` object.

## Compare patterns across samples

`Comparator` converts each sample into per-gene spatial spectra and tests for
differences between groups. Samples may use different orientations and
coordinate systems.

```python
import anndata as ad
import numpy as np
from sonic import Comparator

paths = [
    "control_1.h5ad",
    "control_2.h5ad",
    "control_3.h5ad",
    "case_1.h5ad",
    "case_2.h5ad",
    "case_3.h5ad",
]
samples = [ad.read_h5ad(path) for path in paths]
groups = np.array([0, 0, 0, 1, 1, 1])

comparison = Comparator(samples).compute_spectra(n_jobs=4)

# Optional: remove the sample-wide spectral background before testing.
# comparison.normalize_background()

pattern_results = comparison.test_diff_freq(
    groups,
    statistic="log_l2",
    normalize_shape=True,
)
expression_results = comparison.test_diff_expr(groups)
```

`pattern_results` tests the spatial layout of each gene.
`expression_results` tests its sample-level mean expression. Run both when
you want to distinguish a pattern change from an expression-level change.
Background normalization is optional: use it when sample-wide spectral
differences are nuisance variation. It can also remove shared biological signal.

## Choose an interface

Q-test defaults depend on the backend: Gaussian, Matérn, CAR and graph-Laplacian
kernels use upper-tail Welch on MatrixKernel and upper-tail moment matching on
FFTKernel/NUFFTKernel. Moran uses two-sided CLT on every backend. All defaults
account for sample standardization. See the
[kernel calibration table](https://sonic-spatial.readthedocs.io/en/latest/guides/kernels.html#default-q-test-calibration)
for signed/custom kernels and overrides.

| Input | Task | Interface |
|---|---|---|
| `AnnData` with coordinates or a graph | Detect patterns | `Detector` |
| `SpatialData` with regular bins | Detect patterns on a large grid | `Detector` |
| A list of `AnnData` samples | Compare patterns | `Comparator` |
| A list of `SpatialData` samples | Compare patterns on regular grids | `Comparator` |
| Arrays plus a kernel | Run one Q-test or R-test | `spatial_q_test`, `spatial_r_test` |

## Documentation

- [Quick Start](https://sonic-spatial.readthedocs.io/en/latest/guides/quickstart.html)
- [Choosing a kernel](https://sonic-spatial.readthedocs.io/en/latest/guides/kernels.html)
- [Cross-sample comparison](https://sonic-spatial.readthedocs.io/en/latest/guides/multisample.html)
- [Scaling to large datasets](https://sonic-spatial.readthedocs.io/en/latest/guides/scaling.html)
- [Theory](https://sonic-spatial.readthedocs.io/en/latest/guides/theory.html)
- [API reference](https://sonic-spatial.readthedocs.io/en/latest/autoapi/sonic/index.html)

## Development

```bash
git clone https://github.com/JiayuSuPKU/sonic.git
cd sonic
pip install -e ".[dev,docs]"
pytest -q
```

See the [installation guide](https://sonic-spatial.readthedocs.io/en/latest/guides/installation.html)
for environment troubleshooting.

## Citation

Su, Jiayu, et al. “On the consistent and scalable detection of spatial
patterns.” arXiv:2602.02825 (2026). [Preprint](https://arxiv.org/abs/2602.02825)

SONIC is released under the [BSD 3-Clause License](LICENSE). Please report bugs
and feature requests through [GitHub Issues](https://github.com/JiayuSuPKU/sonic/issues).
