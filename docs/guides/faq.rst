FAQ
===

What does SONIC stand for?
--------------------------

Spatial Organization through Nonrandom-pattern Inference and Comparison.


What is the difference between the Q-test and R-test?
-----------------------------------------------------

The Q-test asks whether one feature has a spatial pattern. Use it to find
spatially variable genes or other spatially structured measurements.

The R-test asks whether two features share a spatial pattern. Use it after the
Q-test to study spatial co-expression among a smaller set of features.

See :doc:`/guides/quickstart` for both workflows and :doc:`/guides/theory` for
the definitions.


Which backend should I use?
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Backend
     - Use it for
   * - ``backend="nufft"``
     - The starting workflow for two-dimensional spatial Q-tests, using Matérn.
   * - ``backend="matrix"``
     - Precomputed graphs or explicit graph neighbourhoods with CAR, Moran, or graph Laplacian kernels.
   * - Grid detector
     - Rectangular rasterized ``SpatialData`` bins, such as Visium HD square bins.

The :func:`~sonic.Detector` factory chooses the detector from the input type;
``backend`` chooses how an ``AnnData`` coordinate dataset is processed. See
:doc:`/guides/kernels` for examples.


Does the FFT backend support hexagonal grids?
---------------------------------------------

No. The FFT backend supports rectangular grids with periodic boundaries. A
staggered hex grid has row-dependent neighbours and cannot be represented by
the scalar FFT operator used by SONIC.

For standard Visium spots, keep the physical coordinates and use the Matrix or
NUFFT path. For an exact hex adjacency, build that graph explicitly and use a
Matrix kernel.


Why prefer CAR over Moran's I for detection?
--------------------------------------------

Moran's I can let different spatial components cancel, which can hide a real
pattern. CAR avoids that cancellation and is the recommended starting point
for smooth graph-based patterns.

Use Moran's I when you need comparability with an existing analysis. The
mathematical explanation is in :doc:`/guides/theory`.


Can SONIC analyze non-spatial data?
-----------------------------------

Yes, if closeness can be represented by coordinates or a graph. Examples
include a k-nearest-neighbour graph in a latent space, a pseudotime ordering,
or a lineage graph. Store a precomputed matrix in ``adata.obsp`` and pass its
key to :meth:`~sonic.DetectorIrregular.setup_data`.


Does SONIC support three-dimensional coordinates?
--------------------------------------------------

:class:`~sonic.MatrixKernel` supports three-dimensional coordinates. The FFT
and NUFFT backends are currently two-dimensional.


Where should I report a problem?
--------------------------------

Open an issue on `GitHub <https://github.com/JiayuSuPKU/sonic/issues>`_ and
include the SONIC version, input type, backend, and smallest example that
reproduces the problem.
