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

For 2D NUFFT features and rotation landmarks, SONIC evaluates an extra border
of Fourier modes for interpolation at the sampling Nyquist boundary. The
original frequency bins and physical units stay unchanged. FFT spectra use
their periodic frequency boundary.

With the analytic null, DF tests return NaN p-values for nonzero contrasts
when the selected residual covariance is numerically zero; zero contrasts
return p=1.
Untestable features are excluded from BH correction.


Compare coupling between genes
------------------------------

Cross-spectrum tests compare the spatial coupling of gene pairs across
conditions. Compute the requested pairs once, then select a representation:

.. code-block:: python

   comparison.compute_cross_spectra([("CD8", "FoxP3"), ("CD8", "CD20")])

   # DF-coherence: changes in coupling strength across frequency bins.
   coupling = comparison.test_cross_diff_freq(groups)

   # DF-coherency: changes in signed, in-phase coupling.
   signed_coupling = comparison.test_cross_diff_freq(
       groups, representation="coherency",
   )

- ``representation="coherence"`` (default) compares coupling strength using
  log squared-magnitude coherence.
- ``representation="coherency"`` compares the real part of normalized
  cross-spectra, retaining the sign of in-phase coupling.

Both use a weighted L2 statistic. For the default ``null="analytic"``, choose
how to estimate the cross-bin residual covariance used by the Liu approximation:

- ``covariance=None`` or ``"pooled"`` (default) shares covariance across eligible
  pairs to stabilize estimation when their covariance is similar.
- ``covariance="per_pair"`` estimates covariance separately for each pair,
  retaining pair-specific variance and cross-bin correlation.

.. code-block:: python

   coupling = comparison.test_cross_diff_freq(groups, covariance="per_pair")

Design matrices and continuous ``contrast=`` terms work as in ``test_diff_freq``.
Both support ``null="analytic"`` and ``null="rotation"``.
``null="permutation"`` is available only for binary comparisons, as described
below.

Each pair uses samples where both genes are present and bins valid in all those
samples. ``freq_weights`` sets bin weights in the original flattened bin order
(radius then angle for polar features); zero weights exclude bins. For missing
data, ``min_samples_per_group`` sets the minimum per arm in binary tests, and
``min_resid_df`` sets the minimum residual degrees of freedom in GLMs.

Results include pair names, the statistic, raw and BH-adjusted p-values, and
sample/bin counts. ``Status`` identifies skipped pairs; ``N_pairs_pooled``
reports the covariance pool size. Retained bins and samples are recorded in
``attrs["bin_indices"]`` and ``attrs["sample_indices"]``.


Explain selected frequency bins
-------------------------------

Use ``explain_spectrum`` to reconstruct selected frequency bins from an
individual sample's source data:

.. code-block:: python

   single = comparison.explain_spectrum(
       "CD8", sample_index=0, bands=[1, 2], representation="power_norm",
   )
   pair = comparison.explain_spectrum(
       ["CD8", "FoxP3"], sample_index=0, bands=[1, 2],
       representation="coherency", memory_budget_bytes="512 MiB",
   )

- ``sample_index`` selects the sample; ``bands`` selects retained spectrum bins.
- For one gene, ``representation="power"`` shows raw power contributions;
  ``"power_norm"`` scales them by total power across retained bins.
- For a pair, ``"cross"`` shows the real co-spectrum, ``"coherency"`` shows
  signed normalized coupling, and ``"coherence"`` shows squared-magnitude coupling.
- ``memory_budget_bytes`` sets the reconstruction memory budget.

The returned ``SpatialExplanation`` contains source ``values``, per-bin
``filtered`` fields, and ``contributions`` whose sums reproduce the selected
raw-source feature. Use its plotting and region-selection methods to inspect
local patterns:

.. code-block:: python

   regions = pair.select_regions(map_index=0, size=100.0, direction="absolute")
   figure = pair.plot(map_index=0, regions=regions)

``map_index`` selects a returned band, ``size`` sets region width in source
coordinate units, and ``direction="absolute"`` selects large contributions of
either sign. Omit ``size`` for an automatic frequency-based width when the
band's lower frequency is positive. See :doc:`quickstart` for plotting options.


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

The covariate design must leave at least one residual frequency dimension.


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


Choose the analytic DF covariance
---------------------------------

For analytic DF and DF-norm tests, ``covariance`` controls how cross-bin
residual covariance is estimated:

- ``None`` or ``"pooled"`` (default) shares covariance across eligible genes to
  stabilize estimation when their covariance is similar.
- ``"per_gene"`` estimates covariance separately for each gene, retaining
  gene-specific variance and cross-bin correlation.

Both use the same weighted L2 statistic and Liu null approximation. The options
support binary groups, continuous contrasts, and gene-specific missing samples.

.. code-block:: python

   pattern_results = comparison.test_diff_freq(
       groups, covariance="per_gene", normalize_shape=True,
   )
   adjusted_results = comparison.test_diff_freq(
       design, contrast="condition", covariance="per_gene", normalize_shape=True,
   )

Use ``estimate_null_covariance`` to inspect the covariance, eigenvalues, and
fitted ``beta`` coefficients used by the test:

.. code-block:: python

   diagnostics = comparison.estimate_null_covariance(
       groups, covariance="per_gene", normalize_shape=True,
   )

The standalone binary and GLM comparison functions accept the same option.
Cross-spectrum tests use ``covariance="per_pair"`` for separate pair estimates.


Conditional rotation tests
--------------------------

``null="rotation"`` calibrates the same weighted L2 statistic by rotating
profiles in the sample space after removing nuisance directions. This preserves
each feature's cross-bin scatter and accounts for uncertainty without a
plug-in covariance estimate.
Use it for binary groups or an estimable scalar GLM contrast, including
continuous predictors and missing samples:

.. code-block:: python

   pattern_results = comparison.test_diff_freq(
       design, contrast="condition", normalize_shape=True,
       null="rotation", n_rotations=19999, random_state=0,
   )

Leave ``covariance=None`` for rotation. The same controls work with
``test_cross_diff_freq`` and the standalone comparison functions. Samples
should be independent biological units. The conditional null assumes Gaussian
errors on the tested profile scale with common covariance across samples within
each feature; covariance may differ between features. DF-norm's nonlinear
normalization does not generally preserve Gaussian errors.

``n_rotations`` sets the draw count and ``random_state`` makes draws reproducible.
The p-value is ``(1 + exceedances) / (n_rotations + 1)``, counting ties
conservatively. With 19999 draws the minimum is 0.00005, which can limit BH
resolution. Result ``attrs`` records ``n_rotations``, the effective
``random_state``, and ``min_p_value``. Few residual degrees of freedom can limit
power. Zero-information profiles receive p=1; nonzero perfect fits can reach
the Monte Carlo floor.

For contrast weights ``a = c @ pinv(X)``, let B contain the unit contrast
axis and an orthonormal full-model residual basis. With transformed profiles Y
and diagonal bin weights W, rotation compares ``L2**2 / (a @ a)`` with
``u.T @ (B.T @ Y @ W @ Y.T @ B) @ u`` for uniform sphere directions u.
The nuisance-adjusted cross-bin scatter stays fixed.


Permutation tests
-----------------

``null="permutation"`` is available only for binary comparisons in
``test_diff_freq`` and ``test_cross_diff_freq``, including masked comparisons.
Pass one-dimensional labels for two exchangeable groups and omit ``contrast``.
GLM designs, including continuous predictors, nuisance covariates, or an explicit
contrast, support ``null="analytic"`` (default) and ``null="rotation"``.

For a binary comparison, permutation calibration relabels samples:

.. code-block:: python

   pattern_results = comparison.test_diff_freq(
       groups,
       null="permutation",
       n_perm=1000,
       random_state=0,
   )

``n_perm`` sets the number of sampled relabelings, and ``random_state`` makes
them reproducible. When the number of possible assignments is at most
``max_exact_permutations``, SONIC enumerates them all.


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
