Cross-sample pattern comparison
===============================

SONIC measures spatial variation in frequency bins.
Low frequencies describe variation over longer distances.
High frequencies describe variation over shorter distances.
Use the same physical coordinate units for all samples.

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Test
     - Measurement in each bin
     - Use it to compare
   * - DF
     - Natural logarithm of a gene's power
     - The amount of spatial variation at each scale
   * - DF-norm
     - Natural logarithm of the bin's share of a gene's total power
     - How a gene's spatial variation is distributed across scales
   * - DF-coherence
     - Natural logarithm of squared-magnitude coherency
     - The strength of spatial coupling between two genes
   * - DF-coherency
     - Real part of coherency
     - The signed, in-phase component of spatial coupling

Let :math:`x_g` be the mean-centered expression of gene :math:`g` across
locations in one sample. Let :math:`K_b` be the real spatial kernel that
averages Fourier power in frequency bin :math:`b`. The bin power has the
quadratic form of the Q statistic. The real part of the cross-spectrum
between genes :math:`g` and :math:`h` has the bilinear form of the R statistic.

.. math::

   P_g(b) = x_g^\top K_b x_g, \qquad
   \operatorname{Re} S_{gh}(b) = x_g^\top K_b x_h.

The :doc:`single-sample Q and R tests <theory>` use standardized expression.
The power profiles retain each gene's expression scale. Dividing
:math:`S_{gh}(b)` by :math:`\sqrt{P_g(b) P_h(b)}` gives coherency.
Its real part is positive for in-phase variation and negative for opposing
variation. Its squared magnitude is coherence.

Compare single genes with DF and DF-norm
----------------------------------------

DF compares mean log power in each bin. Doubling a gene's expression at every
location quadruples its power in every bin, even when the spatial pattern
stays the same. To test whether power shifts between spatial scales, use DF-norm,
which divides each bin's power by the sum across bins for that gene and sample.

Supply the same gene names in the same order for all samples. For
:class:`anndata.AnnData`, the default coordinate key is ``obsm["spatial"]``.
Use ``layer`` if expression is stored outside ``X``.

.. code-block:: python

   import anndata as ad
   import numpy as np
   from sonic import Comparator

   paths = [
       "control_1.h5ad", "control_2.h5ad", "control_3.h5ad",
       "case_1.h5ad", "case_2.h5ad", "case_3.h5ad",
   ]
   samples = [ad.read_h5ad(path) for path in paths]
   groups = np.array([0, 0, 0, 1, 1, 1])
   comparison = Comparator(samples).compute_spectra()

   df = comparison.test_diff_freq(groups)
   df_norm = comparison.test_diff_freq(groups, normalize_shape=True)
   expression = comparison.test_diff_expr(groups)

Compare gene pairs with DF-coherence and DF-coherency
-----------------------------------------------------

Gene-pair tests use the cross-spectrum to compare spatial coupling between
groups. DF-coherence tests for changes in coupling strength. DF-coherency
tests for changes in signed alignment. Within a frequency bin, positive real
coherency means the two expression patterns tend to peak together. Negative
values mean peaks of one pattern tend to align with troughs of the other.

Like single-gene DF tests, cross-spectral tests separate spatial variation
into frequency bins. When interpreting a bin, do not confuse the wavelength of
an expression pattern from the distance over which two genes may interact.
For example, higher coherence in a bin with wavelengths near 100 µm indicates
stronger coupling in expression patterns that repeat about every 100 µm.
It does not mean that the two genes interact over a distance of 100 µm.

.. code-block:: python

   comparison.compute_cross_spectra([("CD8", "FoxP3"), ("CD8", "CD20")])
   strength = comparison.test_cross_diff_freq(groups)
   signed = comparison.test_cross_diff_freq(groups, representation="coherency")

Each pair uses samples where both genes are present. The comparison uses
frequency bins that support a coherency estimate in every retained sample.
Pair tests use spectra from the source expression values, including when
you have normalized the single-gene spectra.

Identify a frequency bin and map its spatial contributions
----------------------------------------------------------

Each DF test combines information across frequency bins into one p-value per
gene or pair. In test results, each row reports the overall p-value (``P_value``)
for one gene or pair. ``P_adj`` gives that p-value after Benjamini-Hochberg
adjustment across the genes or pairs tested.

We can also examine the per-bin results to see which spatial scale has the
strongest evidence for a difference. ``Estimate_per_bin`` gives the signed
effect on the tested scale in each bin.
For two groups, this is the mean for the first sorted label minus the mean
for the second. In the example above, a positive estimate means the bin mean is
higher in controls (0) than in cases (1). With a design matrix, the sign
follows the specified contrast. Divide a natural-log estimate by ``ln(2)`` to
express it in log₂ units.

A frequency bin describes variation across the whole sample. To see where
that variation occurs, map each spot's contribution to the selected bin in
one sample. The same idea applies to single-sample Q and R statistics.
For gene power, the bin kernel applied to centered expression :math:`x`
gives the filtered response :math:`K_b x`. Spot :math:`i` contributes
:math:`x_i (K_b x)_i`. These contributions sum to the power in the bin.

Identify the frequency bin with the strongest evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The t-Cauchy test combines the two-sided bin p-values into one p-value for
the gene or pair. ``T_per_bin`` gives the signed t statistics and
``P_value_per_bin`` gives the unadjusted bin p-values. ``Best_bin`` is the
zero-based index of the bin with the smallest p-value in the combination.
The multiple-testing adjustment applies to the gene or pair as a whole.

.. code-block:: python

   gene_bins = comparison.test_diff_freq(groups, statistic="log_t_cauchy")
   pair_bins = comparison.test_cross_diff_freq(
       groups, representation="coherency", statistic="t_cauchy",
   )

Map a selected bin in one sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the selected bin to ``explain_spectrum``. This example maps each spot's
contribution to signed coherency in the CD8-FoxP3 pair's ``Best_bin``,
using a sample where both genes are present.

.. code-block:: python

   pair = ("CD8", "FoxP3")
   best_bin = pair_bins.set_index(["Feature_1", "Feature_2"]).at[pair, "Best_bin"]
   pair_map = comparison.explain_spectrum(
       pair, sample_index=0, bands=[best_bin], representation="coherency",
   )
   regions = pair_map.select_regions(map_index=0, size=100.0, direction="absolute")
   figure = pair_map.plot(map_index=0, regions=regions)

The map describes the source expression in the selected sample. For one gene,
use ``"power"`` for DF or ``"power_norm"`` for DF-norm. For a pair, choose
``"coherence"``, ``"coherency"``, or ``"cross"`` (real cross-spectrum).
See :doc:`quickstart` for plotting options and :doc:`/comparator_api` for the
method reference.

Adjust for covariates or test a continuous predictor
----------------------------------------------------

Supply a design matrix to adjust for other predictors or to test a continuous
predictor. Use one row per sample in the same order as ``samples``.
A contrast specifies the coefficient or combination of coefficients to test.

.. code-block:: python

   import pandas as pd

   design = pd.DataFrame({"condition": groups, "batch": ["A", "B", "C"] * 2})
   adjusted = comparison.test_diff_freq(design, contrast="condition")
   adjusted_pairs = comparison.test_cross_diff_freq(design, contrast="condition")

Use ``contrast="condition"`` here to test condition 1 minus condition 0 after
adjustment for batch. Naming the coefficient avoids having to know its position.
For a data frame, SONIC adds an intercept and encodes categorical predictors.
Contrast vectors follow the column order of that encoded design matrix.
For a numeric matrix, use its column order and include any intercept yourself.

For a continuous predictor, its coefficient gives the change per unit increase
with the other predictors held fixed. The design must contain enough independent
information to estimate the requested contrast.

Advanced options
----------------

Choose a frequency grid and adjust the spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set the spatial units and bin directions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``unit_scales`` to convert irregular coordinates to common physical units.
For regular rectangular grids, pass :class:`spatialdata.SpatialData` samples
and specify the physical bin height and width with ``spacing``.

.. code-block:: python

   gridded = Comparator(
       spatial_samples, bins="square_008um", table_name="table",
       col_key="array_col", row_key="array_row", spacing=(8.0, 8.0),
   ).compute_spectra()

Use the default radial bins to compare spatial scales across orientations.
Use ``feature_mode="2d"`` to also compare direction. SONIC aligns the
directional spectra across samples. Supply genes with stable directional
patterns in ``landmark_genes`` to define the alignment.

.. code-block:: python

   directional = Comparator(samples, feature_mode="2d").compute_spectra(
       landmark_genes=["EPCAM", "KRT8", "KRT18"],
   )

Adjust gene spectra before testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``normalize_background()`` divides each sample's power by the geometric mean
  across observed genes in each bin. Use it when sample-wide differences are
  nuisance variation. It can also remove shared biological signal.
- ``normalize_covariates(["celltype_astro", "MALAT1"])`` removes the part of
  each gene's log spectrum explained by the spectra of known spatial maps.
  Use the test's design matrix for covariates defined at the sample level.

Choose between L2 distance and t-Cauchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For L2, ``Statistic`` is the weighted root mean square of the signed bin
estimates. Effects in opposite directions both contribute to this distance.
The default is ``statistic="log_l2"`` for DF and DF-norm, and
``statistic="l2"`` for gene pairs.

For t-Cauchy, ``Statistic`` reports the largest absolute t statistic. The
p-value combines the two-sided p-values from bins with a nonzero estimated
effect or variation unexplained by the model. A single small bin p-value can
contribute strongly to the result.

For two independent groups, each bin uses Welch's t test. The groups can
have different variances. With a design matrix, each bin uses an ordinary
least squares t test. Its usual t reference assumes independent Gaussian
errors with a common variance across samples in that bin. The Cauchy
reference is approximate when bin p-values are dependent. Both t-Cauchy
options use ``null="analytic"`` and require ``covariance=None``.

Both L2 and t-Cauchy give equal weight to bins by default. Supply positive,
finite ``freq_weights`` to emphasize spatial scales chosen before testing.
For t-Cauchy, weights affect the combined p-value. The individual bin tests
and the choice of ``Best_bin`` use the original bin p-values.

Choose the null calibration for L2
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Conditional rotation (default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``null="rotation"`` is the default for all four L2 tests. It compares the
observed distance with the distances expected if the contrast direction were
random, given the variation across bins after adjustment for nuisance predictors.
The calculation is deterministic.

The conditional null distribution is exact for independent Gaussian errors on
the tested scale when samples share the same cross-bin covariance for that gene
or pair. Bins can be correlated. Different genes or pairs can have different
covariances. With non-Gaussian errors, calibration depends on the design and
the error distribution.

Gaussian plug-in calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``null="plugin"`` uses a Gaussian reference distribution with an estimated
residual covariance. It treats that estimate as fixed. With few independent
samples, uncertainty in the covariance estimate can affect the p-value.
The model assumes a common residual covariance across samples for each gene
or pair.

By default, plug-in calibration estimates covariance across bins separately for
each gene (``"per_gene"``) or pair (``"per_pair"``). Use ``covariance="pooled"``
when a common covariance across genes or pairs is plausible. Pooling uses more
observations to estimate the bin variances and correlations. Specifying a
covariance selects plug-in calibration if you omit ``null``.

.. code-block:: python

   per_gene = comparison.test_diff_freq(groups, null="plugin")
   pooled = comparison.test_cross_diff_freq(groups, covariance="pooled")

To inspect covariance matrices and fitted coefficients for selected genes, use
``comparison.estimate_null_covariance(groups, features=["CD8", "CD20"])``.

Permutation of binary group labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``null="permutation"`` compares the observed L2 distance with distances from
other assignments of samples to the two groups. Use it when the group labels
are exchangeable under the null hypothesis. Pass one-dimensional binary labels
and omit ``contrast``. For sampled permutations, ``n_perm`` sets the number of
assignments and ``random_state`` sets the seed.
