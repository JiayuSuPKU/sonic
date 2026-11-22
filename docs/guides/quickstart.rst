Quick Start
===========

Use :func:`~sonic.api.Detector` to find spatial patterns in one sample.
Use :func:`~sonic.api.Comparator` to test whether those patterns change between
groups of biological samples.
Both accept irregular coordinates in :class:`anndata.AnnData` or a regular
rectangular grid in :class:`spatialdata.SpatialData`.

.. _pattern-detection:

Detect spatial patterns in one sample
-------------------------------------

Find spatially variable genes with the Q-test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Q-test asks whether a gene's expression varies with location. The kernel
determines which patterns the test detects. This example uses a Matérn kernel
for smooth patterns at irregular coordinates. Choose ``bandwidth`` in the
same units as the coordinates.

.. code-block:: python

   import anndata as ad
   from sonic import Detector

   adata = ad.read_h5ad("spatial_tissue.h5ad")
   detector = Detector(
       adata, backend="nufft", kernel_method="matern", bandwidth=2.0,
   ).setup_data(adata, obsm_key="spatial")

   q_results = detector.compute_qstat()
   spatial_genes = q_results.loc[q_results["P_adj"] < 0.05, "Feature"]

Here ``adata.X`` contains expression, with spots in rows and genes in columns.
Use ``layer`` on the test and explanation methods to select an expression layer.
The table reports Q, its p-value (``P_value``), and the Benjamini-Hochberg
adjusted p-value (``P_adj``). The last line selects genes with ``P_adj < 0.05``.

Test spatial association between genes with the R-test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The R-test asks whether two genes share a spatial pattern. With the Matérn
kernel, a positive R indicates that high expression of one gene tends to occur
near high expression of the other. A negative R indicates the opposite pattern.
Use gene names from your dataset for this example and the map below.

.. code-block:: python

   r_results = detector.compute_rstat(features_x=["CD8"], features_y=["FoxP3"])

Pass two lists to test pairs between gene sets. Pass only ``features_x`` to
test pairs within one set. For example, the names of the 100 genes with
the highest Q scores in this coordinate workflow are in
``q_results.nlargest(100, "Q")["Feature"]``.

.. _plot-maps-and-select-zoom-regions:

Map the spots that contribute to a Q or R score
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Q and R summarize a whole sample. A contribution map shows how each spot
contributes to the score. The signed contributions sum to Q for one gene or
to R for a pair.

.. code-block:: python

   q_map = detector.explain_qstat("FoxP3")
   r_map = detector.explain_rstat("CD8", "FoxP3")
   regions = r_map.select_regions(size=100.0, n_regions=2)
   figure = r_map.plot(regions=regions)

The plot shows expression, the kernel-filtered response, and signed
contributions. Region selection ranks windows by the absolute value of their
signed contribution sum. The test p-value applies to the whole sample.
Choose the window width (``size``) in the sample's coordinate units.
Use ``r_map.plot()`` for the overview alone. See
:class:`~sonic.explanation.SpatialExplanation` for map fields and plotting
options.

.. _pattern-comparison:

Compare spatial patterns across samples
---------------------------------------

Use one sample per biological replicate. Supply the same genes in the same
order and use common spatial units. The default radial bins summarize variation
by spatial scale, averaging over directions.

Compare a gene's power, power distribution, or mean expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DF compares mean log power at each scale. DF-norm first divides each bin's
power by the gene's total across bins. A uniform fold change in expression
preserves these shares. Use the expression test to compare the gene's mean
expression between groups.

.. code-block:: python

   import anndata as ad
   from sonic import Comparator

   paths = [
       "control_1.h5ad", "control_2.h5ad", "control_3.h5ad",
       "case_1.h5ad", "case_2.h5ad", "case_3.h5ad",
   ]
   samples = [ad.read_h5ad(path) for path in paths]
   groups = [0, 0, 0, 1, 1, 1]
   comparison = Comparator(samples).compute_spectra()

   df = comparison.test_diff_freq(groups)
   df_norm = comparison.test_diff_freq(groups, normalize_shape=True)
   expression = comparison.test_diff_expr(groups)

In the DF tables, ``P_adj`` gives the adjusted p-value for each gene.
``Estimate_per_bin`` gives the signed difference per bin on the natural-log
scale. With the labels above, the difference is control minus case.

Compare spatial coupling between gene pairs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit cross-spectra for the pairs of interest. DF-coherence tests whether
coupling strength changes between groups. DF-coherency tests whether signed
alignment changes.

.. code-block:: python

   comparison.compute_cross_spectra([("CD8", "FoxP3")])
   strength = comparison.test_cross_diff_freq(groups)
   signed = comparison.test_cross_diff_freq(groups, representation="coherency")

See :doc:`multisample` to identify the frequency bin with the strongest
evidence, map its spatial contributions, or adjust a comparison for covariates.

Other inputs and advanced options
---------------------------------

Use a neighborhood graph for detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If an AnnData object contains a connectivity matrix in ``obsp``, use the
matrix backend to define neighbors from that graph.

.. code-block:: python

   graph_detector = Detector(adata, backend="matrix", kernel_method="car")
   graph_detector.setup_data(adata, obsp_key="connectivities")
   graph_results = graph_detector.compute_qstat()

Use a regular grid in SpatialData
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a rectangular grid, specify the bin shapes, expression table, and columns
that identify grid rows and columns. Use keys from your SpatialData object.
The grid kernel uses periodic boundaries. For hexagonal spots, use the
coordinate workflow.

.. code-block:: python

   import spatialdata as sd

   sdata = sd.read_zarr("visium_hd.zarr")
   grid_detector = Detector(sdata, kernel_method="car").setup_data(
       sdata, bins="square_008um", table_name="table",
       col_key="array_col", row_key="array_row",
   )
   grid_results = grid_detector.compute_qstat()

To compare several grids, supply the same keys to ``Comparator`` and set
``spacing`` to the physical height and width of a grid cell. See
:doc:`multisample` for an example.

Test expression arrays directly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have expression vectors and coordinates in the same spot
order, pass them to the Q-test or R-test with a kernel.

.. code-block:: python

   from sonic import NUFFTKernel, spatial_q_test, spatial_r_test

   kernel = NUFFTKernel(adata.obsm["spatial"], method="matern", bandwidth=2.0)
   values = adata[:, ["CD8", "FoxP3"]].to_df()
   x = values["CD8"].to_numpy()
   y = values["FoxP3"].to_numpy()
   q, q_pvalue = spatial_q_test(x, kernel)
   r, r_pvalue = spatial_r_test(x, y, kernel)

See :doc:`kernels` for kernel choices, :doc:`scaling` for memory and runtime
options, and :doc:`theory` for calibration assumptions.
