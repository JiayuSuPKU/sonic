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

DF and DF-norm return signed ``Estimate_per_bin`` for every null: mean A minus
mean B (sorted binary labels), or the specified GLM contrast. Estimates are
in natural-log power units (log power share for DF-norm); divide by ``ln(2)``
for log₂ units. ``log_l2`` reports their weighted RMS as ``Statistic``, which
serves as an aggregated effect size metric. Skipped genes have NaN estimates.

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

Both default to ``statistic="l2"``. For ``null="analytic"``, choose
how to estimate the cross-bin residual covariance used by the Liu approximation:

- ``covariance=None`` or ``"pooled"`` (default) shares covariance across eligible
  pairs to stabilize estimation when their covariance is similar.
- ``covariance="per_pair"`` estimates covariance separately for each pair,
  retaining pair-specific variance and cross-bin correlation.

.. code-block:: python

   coupling = comparison.test_cross_diff_freq(groups, covariance="per_pair")

Design matrices and continuous ``contrast=`` terms work as in ``test_diff_freq``.
L2 supports ``null="analytic"`` and ``null="rotation"`` for either design;
``null="permutation"`` is available only for binary comparisons. To also
localize the best bin, use ``statistic="t_cauchy"`` with ``null="analytic"``:

.. code-block:: python

   pair_bins = comparison.test_cross_diff_freq(groups, statistic="t_cauchy")
   signed_pair_bins = comparison.test_cross_diff_freq(
       design, contrast="condition", representation="coherency", statistic="t_cauchy",
   )

This applies binwise Welch or OLS tests to log coherence or real coherency,
respectively, and combines the p-values with Cauchy's rule as described below.

Each pair uses samples where both genes are present and bins valid in all those
samples. ``freq_weights`` sets bin weights in the original flattened bin order
(radius then angle for polar features); zero weights exclude bins. For missing
data, ``min_samples_per_group`` sets the minimum per arm in binary tests, and
``min_resid_df`` sets the minimum residual degrees of freedom in GLMs.

Results include pair names, the statistic, raw and BH-adjusted p-values, and
sample/bin counts. ``Status`` identifies skipped pairs; ``N_pairs_pooled``
reports the covariance pool size. Retained bins and samples are recorded in
``attrs["bin_indices"]`` and ``attrs["sample_indices"]``.

Both statistics return ``Estimate_per_bin``: the mean for group A minus the mean
for group B (sorted labels), or the specified GLM contrast. Coherence estimates
use natural-log coherence; coherency estimates are differences on the signed real
scale. L2 fits all supported bins together; only positive-weight bins contribute
to the weighted RMS, pooled covariance, or null calibration. Unsupported bins
and skipped pairs have NaN estimates.


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


Binwise t tests and Cauchy combination
--------------------------------------

Use ``statistic="log_t_cauchy"`` to test each frequency bin separately and
combine its p-values into a feature-level result, while retaining per-bin
p-values to localize the best bin:

.. code-block:: python

   bin_results = comparison.test_diff_freq(groups, statistic="log_t_cauchy")
   adjusted_bin_results = comparison.test_diff_freq(
       design, contrast="condition", statistic="log_t_cauchy",
       normalize_shape=True,
   )

``log_t_cauchy`` uses the same ``log(max(profile, 1e-12))`` transformation as
``log_l2``. Binary labels select per-bin Welch tests; a design matrix or
explicit contrast selects per-bin OLS t tests, including nuisance covariates.
``normalize_shape=True`` divides each spectrum by its total before logging.

Welch permits different variances between groups and requires two observations
per arm. OLS uses each bin's residual variance and ``n_samples - rank(design)``
degrees of freedom, assuming constant residual variance across samples. Thus
an explicit binary GLM differs from the Welch path. Masked tests fit only
observed samples and apply the same eligibility thresholds as the L2 tests.
Log-scale effects concern mean log power, or mean log power share for DF-norm.

Both DF ``log_t_cauchy`` and cross-spectrum ``t_cauchy`` require
``null="analytic"`` and ``covariance=None``.
``freq_weights`` controls the Cauchy combination: None gives equal weights;
nonnegative weights are renormalized over each feature's informative bins.
Zero weights exclude bins from combination, the statistic summary and
``Best_bin``. Choose
weights independently of the observed signal, for example to emphasize
prespecified spatial scales. Per-bin p-values remain unchanged.

The combined statistic is ``sum(w[k] * cot(pi * p[k]))``, with retained weights
summing to one. Its p-value uses the standard Cauchy tail, without estimating
a cross-bin covariance. The Cauchy
reference is an approximation for dependent bin p-values; log transformation
and shape normalization do not guarantee calibration at small sample sizes.
Low residual degrees of freedom or unequal sample variances can inflate the
combined rejection rate even when marginal bin tests look well calibrated.

Both Cauchy tests return the usual feature or pair identifiers, ``Statistic``,
``P_value`` and ``P_adj``, plus these binwise outputs:

- ``Estimate_per_bin``: mean A minus mean B (sorted binary labels), or the fitted
  GLM contrast, on the tested profile scale.
- ``T_per_bin``: signed t statistics.
- ``P_value_per_bin``: unadjusted two-sided bin p-values.
- ``Best_bin``: nullable, zero-based index of the smallest p-value among
  informative, positive-weight bins. Ties use the first original index;
  no eligible bin gives a missing value.

Arrays preserve the original bin order, including valid zero-weight bins.
Cross-spectrum arrays use the original flattened radius/angle order, with
NaNs for unsupported bins. ``Statistic`` is the maximum absolute t over
positive-weight bins; ``P_value`` is the Cauchy-combined tail and ``P_adj`` is
BH-adjusted across features or pairs. ``Best_bin`` is descriptive: feature-level
significance does not establish significance of the selected bin.
Cross-spectrum ``DF_resid`` reports design residual degrees of freedom, not
the Welch degrees of freedom, which vary by bin.

Constant bins with a zero contrast are excluded from combination; a feature
with no informative bins receives p=1. Exact zero standard error gives p=1
for a zero effect and a limiting zero tail for a nonzero effect. Binary Welch
tails are clipped to the smallest positive float. Skipped features keep NaN
feature results, NaN per-bin arrays and missing ``Best_bin``. Cauchy result
``attrs`` records ``statistic``, ``null`` and ``per_bin_test`` (``welch`` or ``ols``).

The standalone ``compare_two_groups``, ``compare_glm``, and their masked
variants accept ``statistic="log_t_cauchy"`` with the same binwise outputs.


Choose the analytic DF covariance
---------------------------------

For analytic L2 DF and DF-norm tests, ``covariance`` controls how cross-bin
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

For DF and DF-norm, ``null="rotation"`` evaluates the conditional rotation
null by deterministic integration. It tests the same weighted L2 statistic,
conditioning on each feature's nuisance-adjusted scatter.
The conditional null is exact under independent Gaussian errors on the tested profile
scale, with common covariance across samples within each feature or pair.
Cross-bin correlations are unrestricted, and covariance may differ between
features or pairs. Rotation can remain well calibrated under non-Gaussian
errors; robustness depends on the statistic, design and error distribution.

.. code-block:: python

   pattern_results = comparison.test_diff_freq(
       design, contrast="condition", normalize_shape=True,
       null="rotation", integration_tol=1e-6,
   )

Use ``statistic="log_l2"`` and leave ``covariance=None``. The four standalone
binary/GLM comparison functions accept the same option, including masked data.
``integration_tol`` is the initial Davies absolute p-value error target
(default 1e-6). Each call uses at most 10,000 terms. Accepted integration also
requires a relative error budget at most ``0.001*p``; a valid result failing
this check gets one retry at ``min(integration_tol, 1e-4*p)``. Exact beta cases
and spectral boundaries bypass integration.

Unresolved tails use Kuonen's saddlepoint approximation.
``P_value_method`` identifies ``davies``, ``saddlepoint``, ``beta`` or ``exact``
(missing for skipped rows). ``P_value_error`` reports the accepted Davies accuracy
budget, a roundoff estimate for beta, or zero at exact boundaries. It is NaN for
saddlepoint and skipped rows. The integration tolerance does not bound
saddlepoint approximation error; neither quantity is statistical uncertainty.
A tail that cannot be evaluated by either route raises an error.

Result ``attrs`` records ``method="davies"``, ``integration_tol``,
``integration_limit=10000``, the number of approximate rows ``n_saddlepoint``,
``null="rotation"`` and ``covariance=None``. BH correction includes all finite
p-values, including the marked approximations.

For contrast weights ``a = c @ pinv(X)``, let B contain the unit contrast
axis and an orthonormal full-model residual basis. With transformed profiles Y
and diagonal bin weights W, set ``A = B.T @ Y @ W @ Y.T @ B`` and
``q = L2**2 / (a @ a)``. The conditional tail is
``P(u.T @ A @ u >= q) = P(sum((eigenvalues(A) - q) * Z**2) >= 0)``,
where u is uniform on the sphere and Z has independent standard normal entries.
All ``df_resid + 1`` directions remain, including zero eigenvalues of A.

Cross-spectrum ``test_cross_diff_freq(null="rotation")`` uses the same
deterministic evaluator and ``integration_tol``, with ``statistic="l2"`` and
``covariance=None``.

.. code-block:: python

   coupling_results = comparison.test_cross_diff_freq(
       design, contrast="condition", representation="coherency",
       null="rotation", integration_tol=1e-6,
   )


Permutation tests
-----------------

``null="permutation"`` is available only for binary comparisons in
``test_diff_freq`` and ``test_cross_diff_freq``, including masked comparisons.
Pass one-dimensional labels for two exchangeable groups and omit ``contrast``.
GLM designs, including continuous predictors, nuisance covariates, or an explicit
contrast, support ``null="analytic"`` (default) and ``null="rotation"`` for L2.
Log t-Cauchy supports its analytic calibration only.

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
``max_exact_permutations``, SONIC enumerates them all. Sampled and exhaustive
tests count distances tied with the observed L2 within floating-point roundoff.


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
