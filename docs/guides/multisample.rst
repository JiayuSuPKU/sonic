Cross-sample Pattern Comparison
===============================

SONIC compares the spatial pattern of each gene across samples. Samples may
differ in their spots, coordinate systems, and orientations.

The default radial representation records how much variation occurs at each
spatial scale. This makes the comparison insensitive to translation and
rotation. Use the directional representation only when orientation is part of
the biological question.


Compare two groups
------------------

Start with a list of :class:`anndata.AnnData` samples. They must share
``var_names`` and contain coordinates in ``obsm["spatial"]``. Pass ``layer=``
to :func:`~sonic.Comparator` when the values are stored outside ``X``.

.. code-block:: python

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

Both result tables contain ``Feature``, ``Statistic``, ``P_value``, and
``P_adj``. ``pattern_results`` tests spatial layout;
``expression_results`` tests sample-level mean expression. Looking at both
separates a rearranged pattern from a simple change in abundance.

``normalize_background()`` is optional and is not applied by the default
workflow. It divides spectra by each sample's geometric-mean spectrum. Use it
when sample-wide spectral differences are nuisance variation; it can also
remove shared biological signal. To enable it, uncomment the call above so it
runs before testing.

``normalize_shape=True`` compares the distribution of power across spatial
scales after removing total spectrum amplitude. Leave it off when amplitude is
part of the pattern you want to compare. Every nonzero spectrum is normalized
to unit total, including very small powers; all-zero spectra remain zero.
Comparisons and normalization use float64 calculations even when cached
spectra were stored as float32.

Comparison spectra use squared Fourier averages: raw power is divided by
the square of the number of observations (occupied bins for FFT grids,
points for NUFFT). This removes the amplitude inflation from denser sampling
of the same physical field while preserving expression-amplitude differences.
It does not correct differences in spatial coverage or nonuniform sampling
within a sample. DC expression means are unchanged.

For 2D NUFFT features and rotation landmarks, SONIC evaluates an extra border
of Fourier modes for interpolation at the sampling Nyquist boundary. The
original frequency bins and physical units stay unchanged. FFT spectra use
their periodic frequency boundary.

NUFFT spectra retain approximation error controlled by ``eps``. With exactly
zero within-group variation, even a tiny difference between computed spectra
can receive a very small analytic p-value. Count normalization does not remove
this numerical limitation. ``compute_spectra()`` warns when a fully observed
gene has nonzero between-sample spectral differences below the requested
transform accuracy. This diagnostic leaves spectra and p-values unchanged;
recompute with smaller ``eps`` and check effect-size stability before
interpreting significance in such degenerate comparisons.


Regular grids
-------------

Pass a list of :class:`spatialdata.SpatialData` objects when each sample can be
rasterized onto a rectangular grid:

.. code-block:: python

   comparison = Comparator(
       spatial_samples,
       bins="square_008um",
       table_name="square_008um",
       col_key="array_col",
       row_key="array_row",
       spacing=(8.0, 8.0),
   ).compute_spectra()

   pattern_results = comparison.test_diff_freq(groups)

``spacing`` is the physical height and width of a bin. FFT comparisons use
periodic rectangular grids. For standard Visium hexagonal spots, keep the data
as coordinates in ``AnnData`` and use the irregular workflow above.


Account for known spatial covariates
------------------------------------

Use :meth:`~sonic.ComparatorIrregular.normalize_covariates` after computing
spectra to remove variation associated with known spatial maps:

.. code-block:: python

   comparison.normalize_covariates(
       ["celltype_astro", "celltype_neuron", "MALAT1"]
   )

For ``AnnData``, SONIC looks for each name in ``obs`` first and then in
``var_names``. Values must be numeric. The same call works with
``SpatialData`` table fields. You can also pass one pre-rasterized covariate
array per sample; see the API reference for the expected shapes.

The covariate design, including its intercept, must leave at least one residual
frequency dimension. SONIC raises ``ValueError`` if its rank reaches the number
of retained feature bins. Use fewer independent covariates or more supported
bins; no samples are modified when this check fails.


Multi-factor and continuous designs
-----------------------------------

Use a pandas data frame for batch terms, paired designs, or continuous
covariates. Name the effect to test with ``contrast``:

.. code-block:: python

   import pandas as pd

   design = pd.DataFrame(
       {
           "condition": ["control", "control", "control", "case", "case", "case"],
           "batch": ["A", "B", "C", "A", "B", "C"],
       }
   )

   pattern_results = comparison.test_diff_freq(
       design,
       contrast="condition",
       normalize_shape=True,
   )
   expression_results = comparison.test_diff_expr(
       design,
       contrast="condition",
   )

SONIC adds an intercept and encodes categorical columns for a pandas design.
For a NumPy design matrix, include any intercept yourself and pass a contrast
vector.


Permutation tests
-----------------

The analytic test is the default. For a binary comparison, request a label
permutation test explicitly:

.. code-block:: python

   pattern_results = comparison.test_diff_freq(
       groups,
       null="permutation",
       n_perm=1000,
       random_state=0,
   )

Permutation tests are not available for multi-column designs. With very few
samples, the number of distinct label assignments also limits attainable
p-values.


Keep physical units consistent
------------------------------

Spatial frequencies are comparable only when coordinates use the same unit.
Use ``unit_scales`` to convert each sample before spectrum calculation:

.. code-block:: python

   comparison = Comparator(
       samples,
       unit_scales=[1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
   ).compute_spectra()

For example, the last three samples above may use pixels that are half the size
of those in the first three samples. You can also supply a common or per-sample
``spacing`` together with ``grid_shape`` when you need to control the internal
NUFFT grid. Automatic inference is sufficient for most datasets.


Preserve direction when needed
------------------------------

The default ``feature_mode="radial"`` removes direction. To compare
directional structure, use ``feature_mode="2d"``. SONIC estimates one rotation
per sample before comparison:

.. code-block:: python

   comparison = Comparator(
       samples,
       feature_mode="2d",
   ).compute_spectra(
       landmark_genes=["EPCAM", "KRT8", "KRT18"],
   )

Use landmark genes with a stable directional pattern across samples. By
default, SONIC estimates alignment from the geometric-mean spectrum.


See also
--------

- :doc:`/guides/quickstart` for the two main workflows.
- :doc:`/guides/scaling` for chunking and parallelism.
- :doc:`/autoapi/sonic/comparators/index` for all comparator options.
