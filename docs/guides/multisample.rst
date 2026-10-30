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

Both DF tests return NaN p-values for nonzero contrasts when the pooled
residual covariance is numerically zero; zero contrasts return p=1.
Untestable features are excluded from BH correction.

NUFFT spectra retain approximation error controlled by ``eps``. With very small
nonzero within-group variation, a tiny difference between computed spectra
can still receive a very small analytic p-value. Count normalization does not remove
this numerical limitation. ``compute_spectra()`` warns when a fully observed
gene has nonzero between-sample spectral differences below the requested
transform accuracy. This diagnostic leaves spectra and p-values unchanged;
recompute with smaller ``eps`` and check effect-size stability before
interpreting significance in such degenerate comparisons.


Compare coupling between genes
-----------------------------

Cross-spectrum tests ask whether two genes change their spatial coupling
across conditions, even when their individual auto-spectra stay unchanged.
Compute the requested pairs once, then test either representation:

.. code-block:: python

   comparison.compute_cross_spectra([("CD8", "FoxP3"), ("CD8", "CD20")])

   # DF-coherence: changes in coupling strength across frequency bins.
   coupling = comparison.test_cross_diff_freq(groups)

   # DF-coherency: changes in signed, in-phase coupling.
   signed_coupling = comparison.test_cross_diff_freq(
       groups, representation="coherency",
   )

``representation="coherence"`` uses ``log(max(C, 1e-12))``, where
``C = abs(S_xy / sqrt(S_xx * S_yy))**2``. ``representation="coherency"``
uses the real part of ``S_xy / sqrt(S_xx * S_yy)`` without a transformation.
A radial sign reversal can change coherency while leaving coherence unchanged.
For polar bins, coherence also includes the magnitude of the imaginary part;
both tests are unaffected by the half-turn ambiguity of axial alignment.
There is no signed L1 normalization or asinh transformation.

Each test returns one aggregate p-value per pair. The statistic is the weighted
L2 distance between mean profiles (or the specified GLM contrast). The analytic
null follows DF: fit each pair's retained-sample design, pool residual
cross-products into one shared covariance, then use each pair's supported
covariance submatrix in the Liu mixture-chi-square approximation. There is no covariance inversion;
the number of bins can exceed the number of samples. A larger pair panel does
not increase the number of independent biological samples.

As in ``test_diff_freq``, expression presence determines which samples are
used, requiring both genes in a pair to be present. Only bins valid in every
retained sample for that pair are tested. Invalid auto-power is never converted
into valid zero coherence by the log floor. ``freq_weights`` refers to the
original flattened bins (radius then angle for polar features); zero weights
exclude bins, and weights are
renormalized over retained bins. Support is fixed before inference and remains
fixed across label permutations. ``attrs["bin_indices"]`` records the retained
bin indices for each pair; ``attrs["sample_indices"]`` records retained samples.
Different bin support sets share covariance estimates. For each covariance
entry, residual cross-products are summed over eligible pairs supporting both
bins and divided by those pairs' total residual degrees of freedom. Entries
with no joint support start at zero. Because these entrywise estimates can
form an indefinite matrix, negative eigenvalues of its correlation matrix are
clipped, then its diagonal is rescaled to restore the pooled marginal variances.
This PSD correction is applied once before taking each pair's principal
submatrix and may change supported off-diagonal entries. Each test retains its
own sample-dependent contrast scale and normalized bin weights.
``N_pairs_pooled`` counts all eligible pairs in the shared pool; the number of
contributors to individual entries varies with bin overlap. A singleton pool
has no cross-pair covariance borrowing. The existing small-residual-df warnings
are unchanged for both auto-spectrum and cross-spectrum DF; PSD correction
does not establish finite-sample calibration.

The output includes ``Feature_1``, ``Feature_2``, ``Statistic``, ``P_value``,
``P_adj``, ``N_samples``, ``N_bins``, ``N_pairs_pooled``, ``DF_resid``,
``Effective_rank``, and ``Status``. Self-pairs, pairs with no common supported
bins, pairs with insufficient observations or an ineligible retained-sample
design, and nonzero effects with a numerically zero pooled residual covariance
have NaN p-values and an explanatory status. Constant null profiles have p=1.
BH correction follows DF and applies separately to each call using only finite
p-values. Untestable pairs and self-pairs are excluded.
Running both representations does not automatically correct across both calls.

Design matrices and ``contrast=`` follow ``test_diff_freq``. Binary comparisons
require ``min_samples_per_group=2`` observed samples per arm on the masked
path; masked GLMs require ``min_resid_df=1`` and an estimable contrast.
Unmasked binary tests allow singleton arms: analytic inference requires positive
residual degrees of freedom, while permutations need both labels to be present.
These controls follow ``test_diff_freq``. Binary output includes ``n_obs_A`` and
``n_obs_B``. Binary tests optionally support ``null="permutation"`` with
``n_perm``, ``random_state``, and ``max_exact_permutations``. These are unrestricted
sample-label permutations within each pair's retained samples. As in DF,
masked pairs draw separately in original feature order, even when sample masks
match. Unmasked pairs share relabellings. With masked presence, ``n_permutations``
and ``exact_permutations`` in ``attrs`` are dictionaries keyed by pair.
There are no residual or blocked permutations.
Use independent biological samples, not repeated specimens as independent rows.

The Liu approximation can accumulate large p-values at 1 when covariance has
a dominant eigenvalue; it is an upper-tail approximation, not an exact
finite-sample p-value calculation.

The analytic null assumes approximately Gaussian contrasts and comparable
residual covariance among pooled pairs. Log coherence is not a guaranteed
variance-stabilizing transformation; very small coherence values, heterogeneous
pair noise, small samples and small covariance pools require calibration checks.
Pairs sharing a gene are dependent, and BH does not guarantee control under
arbitrary dependence. Normalization removes multiplicative amplitude scaling,
but does not eliminate all abundance, composition or sampling effects.

Testing leaves cached auto- and cross-spectra unchanged. It materializes one
support group's binned profiles at a time and uses bounded DF residual scratch
blocks. The bin-by-bin covariance still uses quadratic memory in the number of
retained bins; no Fourier coefficient cache is created.


Explain selected frequency bins
-------------------------------

After selecting a feature and frequency bins, reconstruct their spatial
contributions in an individual sample:

.. code-block:: python

   single = comparison.explain_spectrum(
       "CD8", sample_index=0, bands=[1, 2], representation="power_norm",
   )
   pair = comparison.explain_spectrum(
       ["CD8", "FoxP3"], sample_index=0, bands=[1, 2],
       representation="coherency", memory_budget_bytes="512 MiB",
   )
   pair.scores  # pair.contributions.sum(axis=1)

This uses the same ``SpatialExplanation`` container as detector Q/R maps.
``values`` contains the selected original observations, ``filtered`` contains
one field per bin and feature, and ``contributions`` contains one signed map
per bin. ``observation_indices`` and ``coordinates`` locate the rows in the
source sample. FFT indices address C-order raster cells; coordinates are local
(y, x) positions with origin zero and the fitted spacing. NUFFT coordinates
retain their source column order and units; ``metadata["unit_scale"]`` gives
the conversion used for the frequency grid. No spatial coordinates are rotated.

``bands`` indexes retained spectrum columns, including any removal of
unsupported radial intervals. Polar columns flatten radius then angle.
``metadata`` records these indices, corresponding radial intervals and angular
centers, the sample alignment angle, and the representation. The fitted radial
or interpolated polar reducer is reused without changing alignment or caches.

One feature supports ``power`` and ``power_norm``; the latter divides by total
raw power over **all retained bins**, not just the displayed bins. Pairs support
``cross`` (real co-spectrum), ``coherency`` (real signed coherency), and
``coherence`` (squared magnitude, including polar phase). For coherency, the
observed denominator sqrt(P_x P_y) is fixed during allocation. For coherence,
a complex cross-spectrum allocation a_i is mapped to
Re(conj(S_xy) a_i) / (P_x P_y). This sums to coherence but may be negative locally.
``filtered`` includes these fixed normalization and phase factors so that the
shared elementwise contribution formula still applies.

Scores reconstruct the **untransformed raw-source feature**. They do not
attribute log profiles, aggregate DF statistics, p-values, or background- or
covariate-adjusted caches. Raw features are recomputed even if such adjustments
were previously applied to ``spectra_``; metadata labels this
``spectrum_source="recomputed_raw"``. Use the same unchanged source data as the
fit. These are descriptive allocations, not derivatives or local significance.

Only one/two genes are transformed, and requested bands are processed one at a
time. The memory estimate includes returned maps and transform/reducer scratch,
but excludes resident inputs, rasterization internals and library overhead.
Invalid normalized pair bins raise an error instead of producing a misleading
map; finite raw power and co-spectrum maps remain available. Plotting and
region selection are separate from this computation API.


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
