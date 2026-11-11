Quick Start
===========

Use SONIC for two main tasks:

1. Detect spatial patterns within one sample.
2. Compare spatial patterns across several samples.

The examples use the :func:`~sonic.Detector` and
:func:`~sonic.Comparator` factories. They select the coordinate or grid
implementation from the input type.


Pattern detection
-----------------

The Q-test finds features whose values depend on location. The R-test finds
pairs of features that share a spatial pattern. Use
:func:`~sonic.Detector` to run either test across an entire dataset.


AnnData with coordinates
~~~~~~~~~~~~~~~~~~~~~~~~

The input is an :class:`anndata.AnnData` with expression values in ``adata.X``
or a layer and coordinates in ``adata.obsm["spatial"]``. Start with NUFFT and a
Matérn kernel for two-dimensional spatial coordinates. Choose ``bandwidth`` in
coordinate units to match the spatial scale of interest.

.. code-block:: python

   import anndata as ad
   from sonic import Detector

   adata = ad.read_h5ad("spatial_tissue.h5ad")
   print(f"{adata.n_obs} spots × {adata.n_vars} genes")

   detector = Detector(
       adata,
       kernel_method="matern",
       backend="nufft",
       bandwidth=2.0,
       nu=1.5,
   ).setup_data(
       adata,
       obsm_key="spatial",
       min_cells_frac=0.05,
   )

   q_results = detector.compute_qstat(n_jobs=4)
   svgs = q_results[q_results["P_adj"] < 0.05]
   print(f"Found {len(svgs)} spatially variable genes")

``q_results`` is indexed by feature name and sorted by Q. It contains the test
statistic, a p-value, and a Benjamini-Hochberg adjusted p-value.

After finding spatially variable genes, test a smaller set for spatial
co-expression:

.. code-block:: python

   top_genes = q_results.nlargest(100, "Q").index.tolist()
   r_results = detector.compute_rstat(
       features_x=top_genes,
       features_y=None,
       n_jobs=4,
   )

``features_y=None`` tests pairs within ``features_x``. Pass a second list to
test every pair between two feature sets.


Explain an observed Q or R score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After ``setup_data``, request a spatial explanation for one feature or pair:

.. code-block:: python

   q_map = detector.explain_qstat("FoxP3")
   r_map = detector.explain_rstat("CD8", "FoxP3", memory_budget_bytes="512 MiB")

   print(q_map.scores[0])  # Sum of the Q contributions
   print(r_map.scores[0])  # Sum of the R contributions
   contributions = r_map.contributions[0]

Calling ``compute_qstat`` or ``compute_rstat`` first is optional. Explanations
use the same fitted kernel and feature preparation, and check reconstruction
against an independently evaluated score. They do not fit a null distribution
or provide regional p-values. R uses the symmetric allocation
``(z_x * Kz_y + z_y * Kz_x) / 2`` and requires a symmetric real kernel.

Both detector classes return ``sonic.explanation.SpatialExplanation``:

- ``feature_names``: the selected name or ordered pair.
- ``values``: original selected values, shape ``(n_observations, n_features)``.
- ``filtered``: responses to the test-prepared values, shape
  ``(1, n_observations, n_features)``. These are kernel-filtered fields, not
  reconstructed expression intensities.
- ``contributions``: signed allocations, shape ``(1, n_observations)``.
- ``observation_indices``: original setup-time AnnData row positions, including
  alignment after isolated graph nodes are removed; for grids, these are
  C-order flattened raster indices, not source table row indices.
- ``coordinates``: original AnnData coordinate order, or grid ``(y, x)``
  coordinates from kernel spacing and origin ``(0, 0)``. These grid coordinates
  are local kernel coordinates, not a SpatialData world-coordinate transform.
  Graph-only explanations may have no coordinates.
- ``metadata``: statistic, backend, source and normalization information.

``scores`` is computed from the contributions, without storing a duplicate
score array. Structural raster holes are omitted from returned rows while the
reported Q/R retains its existing full-grid standardization scale.

AnnData explanations accept ``source="obs"`` for individual numeric metadata
columns, and ``layer=`` with ``source="var"``. Encode categorical levels into
numeric columns before selecting one for explanation. Features excluded by the
detector's expression/variance filters raise an error. Grid explanations use
the raster prepared by ``setup_data``.

Only the selected one/two features are densified. The memory budget estimates
workspace and returned arrays; it excludes resident data, kernel storage, lazy
factorization and library overhead. No Fourier coefficients or source-container
references are retained. Comparator band maps use the same container; see
:doc:`multisample`.

Plot maps and select zoom regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plotting and region selection operate on the returned ``SpatialExplanation``
for either detector Q/R or comparator bins. They do not rerun the kernel or
Fourier transforms:

.. code-block:: python

   regions = r_map.select_regions(size=100.0, n_regions=2)
   figure = r_map.plot(regions=regions)
   figure.savefig("r_explanation.png", dpi=180)

``size`` is the square window width in the explanation's **stored coordinate
units**. Choose it for your dataset; no micrometre or millimetre conversion is
assumed. Q/R maps require an explicit size. Graph-only explanations without
coordinates cannot be plotted or searched for regions.

Region candidates advance by one quarter of the width. By default they need
at least 64 observations and 35% occupied tiles in a 20-by-20 subdivision.
``min_observations`` and ``min_coverage`` adjust those eligibility checks.
Occupancy is a coordinate diagnostic, not a tissue mask. Candidates must fit
inside the coordinate bounding box; sparse tissue and holes can leave no
eligible crop. The requested size is never silently reduced.

``direction="absolute"`` ranks the absolute **signed sum** of contributions;
``"positive"`` and ``"negative"`` select the respective direction. Opposite
signs can cancel within a window. Greedy overlap suppression returns up to
``n_regions`` windows. These are contribution-enriched examples, not unbiased
samples or evidence of regional significance. Region selection does not infer
which direction is associated with a condition; use your sample-level contrast
to interpret it. ``memory_budget_bytes`` limits estimated tiling scratch;
a very small window in a large coordinate domain can require more memory.

The returned table records ``coord0_min``, ``coord1_min``, ``coord0_max`` and
``coord1_max`` in original column order, plus occupancy, observation count and
the exact signed contribution sum. Bounds are half-open: lower edges are
included and upper edges excluded. Use these bounds with ``coordinates`` to
select rows and ``observation_indices`` to map them to the source data.

``plot()`` returns a Matplotlib figure. Columns show original values, filtered
responses and signed contributions; the first row is an overview and each
provided region adds a zoom row. Without ``regions``, only the overview is
shown. Overview tiles contain observation means and leave empty tiles blank;
zoom panels show every observation. All rows in a column share color limits.
The default limits clip to the 1st/99th percentiles for original values and
symmetric 99th-percentile absolute limits for signed fields. Pass ``limits``
as one (low, high) pair per column to compare multiple samples consistently.
Reported sums always use individual, unclipped contributions.

Plots retain coordinate units. Known grid maps put column 1 horizontally and
column 0 downward; other sources put column 0 horizontally and column 1 upward.
``coordinate_order=(1, 0)`` or ``(0, 1)`` overrides the axis-column mapping.
Axes label the source column numbers. Crops inherit full-sample filtering and
normalization, including contributions from locations outside the crop.


Which backend should I use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Data
     - Starting point
     - Use when
   * - Two-dimensional coordinates, smooth patterns
     - ``backend="nufft"`` with Matérn
     - Default starting point for spatial Q-tests.
   * - Coordinates or a graph
     - ``backend="matrix"`` with CAR
     - You want graph neighbourhoods or have a matrix in ``adata.obsp``.
   * - Regular rasterized bins
     - :class:`~sonic.DetectorGrid` through :func:`~sonic.Detector`
     - The input is :class:`spatialdata.SpatialData`, such as Visium HD.

For a precomputed adjacency or connectivity matrix, replace ``obsm_key`` with
``obsp_key``:

.. code-block:: python

   detector = Detector(
       adata,
       kernel_method="car",
       backend="matrix",
       rho=0.9,
   ).setup_data(adata, obsp_key="connectivities")

See :doc:`/guides/kernels` for other kernels and tuning parameters.


SpatialData with regular bins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a regular grid, pass a :class:`spatialdata.SpatialData` object and the keys
needed to rasterize its table. The row and column fields must define a
contiguous rectangular grid.

.. code-block:: python

   import spatialdata as sd
   from sonic import Detector

   sdata = sd.read_zarr("visium_hd.zarr")
   detector = Detector(
       sdata,
       kernel_method="car",
       rho=0.9,
       neighbor_degree=1,
   ).setup_data(
       sdata,
       bins="square_008um",
       table_name="square_008um",
       col_key="array_col",
       row_key="array_row",
       min_count=10,
   )

   q_results = detector.compute_qstat()

The grid backend uses periodic boundaries. Standard Visium spot layouts are
hexagonal; use the coordinate workflow for those samples. FFT acceleration is
limited to rectangular grids.


Observed expression and numeric metadata must be finite; NaN and infinity are
rejected before testing. Missing categorical metadata is also rejected.
Categorical ``obs`` features are expanded into indicator columns for both Q
and R tests on the matrix and NUFFT backends. Structural holes in a grid
remain supported and are excluded from the observed domain.

Test one feature directly
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the lower-level functions when you already have arrays and want to run one
test.

.. code-block:: python

   from sonic import NUFFTKernel, spatial_q_test, spatial_r_test

   kernel = NUFFTKernel(
       coords,
       method="matern",
       bandwidth=2.0,
       nu=1.5,
   )
   Q, q_pvalue = spatial_q_test(gene_x, kernel)
   R, r_pvalue = spatial_r_test(gene_x, gene_y, kernel)


Pattern comparison
------------------

Pattern comparison asks whether a gene's spatial layout changes across samples
or conditions. SONIC compares spatial spectra, so samples can have different
positions and orientations.

The shortest workflow takes a list of :class:`anndata.AnnData` samples with
shared ``var_names`` and coordinates in ``obsm["spatial"]``:

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

``pattern_results`` ranks genes by changes in spatial layout.
``normalize_shape=True`` removes overall spectrum amplitude before testing, so
the comparison focuses on how power moves across spatial scales.
``expression_results`` tests each gene's sample-level mean expression.
Background normalization is optional. Use it when sample-wide spectral
differences are nuisance variation; it can also remove shared biological signal.

A gene can therefore show:

- a pattern change without a mean-expression change;
- a mean-expression change without a pattern change;
- both changes.

For regular grids, pass a list of :class:`spatialdata.SpatialData` samples and
the rasterization keys:

.. code-block:: python

   comparison = Comparator(
       spatial_samples,
       bins="square_008um",
       table_name="square_008um",
       col_key="array_col",
       row_key="array_row",
   ).compute_spectra()

   pattern_results = comparison.test_diff_freq(groups)

See :doc:`/guides/multisample` for covariate maps, physical coordinate units,
permutation tests, and multi-factor designs.


Next steps
----------

- :doc:`/guides/kernels` for choosing a kernel.
- :doc:`/guides/multisample` for cross-sample analysis.
- :doc:`/guides/scaling` for memory and runtime controls.
- :doc:`/guides/theory` for the statistical derivations.
- :doc:`/autoapi/sonic/index` for the full API.
