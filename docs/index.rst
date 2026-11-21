SONIC
=====

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   self
   guides/installation
   guides/quickstart
   guides/kernels
   guides/multisample
   guides/scaling
   guides/theory
   guides/faq

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   comparator_api
   autoapi/sonic/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Development

   changelog

**SONIC** (Spatial Organization through Nonrandom-pattern Inference and
Comparison) detects spatial patterns in omics data and compares those patterns
across samples. It accepts coordinates, graphs, and regular grids through
:class:`anndata.AnnData` and :class:`spatialdata.SpatialData`.


Pattern detection
-----------------

Use :func:`~sonic.Detector` to screen every feature in one sample. The Q-test
finds spatially variable features; the R-test finds feature pairs that share a
spatial pattern.

.. code-block:: python

   from sonic import Detector

   detector = Detector(
       adata,
       kernel_method="matern",
       backend="nufft",
       bandwidth=2.0,
       nu=1.5,
   ).setup_data(adata, obsm_key="spatial")

   q_results = detector.compute_qstat()

Start with NUFFT and Matérn for two-dimensional spatial coordinates; choose
``bandwidth`` in coordinate units. The kernel determines which patterns
receive a high score. CAR and Matérn favour smooth spatial changes;
the graph Laplacian favours sharp differences
between neighbours. The :doc:`guides/kernels` guide gives practical defaults.


Pattern comparison
------------------

Use :func:`~sonic.Comparator` when you have several samples. SONIC compares
per-gene spatial spectra across samples with different spots, orientations, or
coordinate systems.

.. code-block:: python

   from sonic import Comparator

   comparison = Comparator(samples).compute_spectra()

   # Optional: remove the sample-wide spectral background before testing.
   # comparison.normalize_background()
   pattern_results = comparison.test_diff_freq(groups, normalize_shape=True)
   expression_results = comparison.test_diff_expr(groups)

With ``normalize_shape=True``, the pattern test compares each gene's share of
power at each spatial scale. The expression test compares its mean expression
between groups. See :doc:`guides/multisample` for test options, assumptions,
and examples with sample-level covariates.


Choose your input
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Input
     - Data layout
     - SONIC interface
   * - :class:`anndata.AnnData`
     - Coordinates or a precomputed graph
     - :func:`~sonic.Detector` or :func:`~sonic.Comparator`
   * - :class:`spatialdata.SpatialData`
     - Regular rasterized bins
     - :func:`~sonic.Detector` or :func:`~sonic.Comparator`
   * - NumPy arrays
     - One feature or feature pair plus a kernel
     - :func:`~sonic.spatial_q_test` or :func:`~sonic.spatial_r_test`


Start here
----------

- :doc:`guides/installation`
- :doc:`guides/quickstart`
- :doc:`guides/kernels`
- :doc:`guides/multisample`
- :doc:`guides/theory` for derivations and proofs


Citation
--------

Su, Jiayu, et al.
*On the consistent and scalable detection of spatial patterns.*
`arXiv:2602.02825 (2026) <https://arxiv.org/pdf/2602.02825>`_.

Please report bugs and feature requests through
`GitHub Issues <https://github.com/JiayuSuPKU/sonic/issues>`_.
