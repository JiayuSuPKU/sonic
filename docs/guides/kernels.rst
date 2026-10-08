Choosing a Kernel
=================

The kernel tells SONIC which spatial patterns to reward. Choose it from the
data layout and the type of pattern you want to detect.


Recommended starting points
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 32 36

   * - Data
     - Pattern
     - Start with
   * - Two-dimensional coordinates
     - Smooth gradients or domains
     - :class:`~sonic.NUFFTKernel` with ``method="matern"``
   * - Coordinates or a graph
     - Smooth graph neighbourhoods
     - :class:`~sonic.MatrixKernel` with ``method="car"``
   * - Coordinates or a graph
     - Sharp changes between neighbours
     - :class:`~sonic.MatrixKernel` with ``method="graph_laplacian"``
   * - Rectangular rasterized grid
     - Smooth gradients or domains
     - :class:`~sonic.FFTKernel` with Matérn or CAR
   * - Small dataset, under about 5,000 observations
     - Any supported pattern
     - :class:`~sonic.MatrixKernel`

Matérn is a good default for physical coordinates. CAR is a good default when
neighbourhoods are defined by a graph.

.. code-block:: python

   from sonic import FFTKernel, MatrixKernel, NUFFTKernel

   smooth_coords = NUFFTKernel(
       coords,
       method="matern",
       bandwidth=25.0,
       nu=1.5,
   )

   smooth_graph = MatrixKernel.from_coordinates(
       coords,
       method="car",
       k_neighbors=4,
       rho=0.9,
   )

   sharp_graph = MatrixKernel.from_coordinates(
       coords,
       method="graph_laplacian",
       k_neighbors=4,
   )

   smooth_grid = FFTKernel(
       shape=(1000, 1000),
       method="car",
       neighbor_degree=1,
       rho=0.9,
   )


What each method detects
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 46 30

   * - Method
     - Pattern
     - Main parameters
   * - ``"matern"``
     - Smooth spatial changes with adjustable scale and smoothness.
     - ``bandwidth``, ``nu``
   * - ``"gaussian"``
     - Very smooth, distance-based patterns concentrated at larger scales.
     - ``bandwidth``
   * - ``"car"``
     - Smooth patterns defined by graph neighbours.
     - ``rho``, ``k_neighbors`` or ``neighbor_degree``
   * - ``"graph_laplacian"``
     - Boundaries, textures, and local differences between neighbours.
     - ``k_neighbors`` or ``neighbor_degree``
   * - ``"moran"``
     - Classical spatial autocorrelation for comparisons with older methods.
     - ``k_neighbors`` or ``neighbor_degree``

Moran's I can miss patterns because positive and negative parts of its
spectrum can cancel. Prefer CAR for smooth graph-based detection. The
:doc:`/guides/theory` page gives the mathematical argument.


.. _q-null-defaults:

Default Q-test calibration
--------------------------

The default depends on **both the kernel method and the backend**.
``compute_null_params(kernel)``, ``spatial_q_test`` and the detector classes
use the same selection:

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Kernel method
     - MatrixKernel
     - FFTKernel
     - NUFFTKernel
   * - ``"gaussian"``
     - ``"welch"``
     - ``"moments"``
     - ``"moments"``
   * - ``"matern"``
     - ``"welch"``
     - ``"moments"``
     - ``"moments"``
   * - ``"car"``
     - ``"welch"``
     - ``"moments"``
     - ``"moments"``
   * - ``"graph_laplacian"``
     - ``"welch"``
     - ``"moments"``
     - ``"moments"``
   * - ``"moran"``
     - ``"clt"``
     - ``"clt"``
     - ``"clt"``
   * - Precomputed matrix (default label)
     - ``"welch"``; assumes PSD
     - Not applicable
     - Not applicable

``welch`` matches the mean and variance with a scaled chi-square distribution.
``moments`` uses up to four moments to select a Liu noncentral chi-square,
bounded beta, central chi-square, or normal fit. **Both test the upper tail**
(unusually large Q). ``clt`` uses a **two-sided** normal approximation and
can detect unusually small as well as unusually large Q. A normal fallback
inside ``moments`` still tests only the upper tail.

All defaults account for sample standardization (sample standard deviation,
``ddof=1``), using finite-sample moments under an iid central Gaussian null.
They are analytic approximations, not permutation tests. Moment matching
corrects variance, skewness and kurtosis before fitting; Welch and CLT use
the corrected mean and variance. A zero-variance null or constant input
feature returns p=1.

FFT uses all modes for either ``fft2`` or ``rfft2``. NUFFT moments describe the
operator at the observed points, with sample size ``n=number of points``;
the internal Fourier grid does not define the null sample size. NUFFT moment
calibration defaults to analytic lower traces and 60 probes for higher traces,
even when eigenvalues are cached. Reduced eigendecomposition is opt-in through
``compute_null_params(kernel, method="moments", nufft_spectrum=True)``. Explicit
matrix Welch/CLT traces are exact; precision-backed kernels use 15 probes by
default. See :ref:`q-null-calibration` for fit selection and probe controls.

The signed-spectrum check overrides the table for a modified FFT/NUFFT kernel:
negative Fourier weights beyond numerical roundoff select ``clt`` regardless
of the method name. A custom precomputed matrix is **not** eigendecomposed
just to detect its signs. Label Moran adjacency matrices with
``method="moran"``; for other symmetric indefinite matrices, choose ``clt``
or finite-sample ``moments`` explicitly. Welch assumes PSD.

To override the default, prepare a cache and pass it to the array-level Q-test:

.. code-block:: python

   from sonic import spatial_q_test
   from sonic.statistics import compute_null_params

   params = compute_null_params(kernel, method="moments")
   Q, p = spatial_q_test(data, kernel, null_params=params)
   print(params["method"], params["model"], params["tail"])
   print(params["q_fit"]["family"], params["q_fit"]["fit"])

``method="moments"`` also permits signed symmetric kernels with the default
``dirichlet_correction=True``. On Moran this deliberately changes the test
from two-sided CLT to an upper-tail test. ``dirichlet_correction=False``
changes the null model to an unstandardized Gaussian quadratic form; it does
not disable standardization of the input data. Detector ``compute_qstat``
methods use automatic calibration; overrides go through ``spatial_q_test``.


Tune the parameters
-------------------

``bandwidth``
   Used by Gaussian and Matérn kernels. It has the same unit as the spatial
   coordinates. Smaller values focus on short-range structure; larger values
   favour broader patterns.

``nu``
   Controls Matérn smoothness. ``nu=1.5`` is a practical starting point.

``rho``
   Controls CAR smoothing. ``rho=0.9`` is the default. Values closer to one
   put more weight on broad, smooth patterns and must remain below one.

``k_neighbors``
   Sets graph connectivity for coordinate and graph kernels. Start with a
   small local neighbourhood, then check whether the top results remain stable
   across a few nearby values.

``neighbor_degree``
   Sets the number of neighbour rings on a rectangular FFT grid. One uses the
   nearest horizontal and vertical neighbours; larger values include more
   distant grid cells.


Coordinates, graphs, and grids
------------------------------

:class:`~sonic.MatrixKernel` accepts coordinates or a precomputed matrix. Use
it for custom graphs, data with more than two coordinate dimensions, and exact
graph neighbourhoods.

:class:`~sonic.NUFFTKernel` accepts irregular two-dimensional coordinates. It
avoids a dense observation-by-observation matrix and is the usual choice for
large coordinate datasets with Gaussian or Matérn kernels.

:class:`~sonic.FFTKernel` accepts rectangular two-dimensional grids with
periodic boundaries. It does not support staggered hexagonal grids. Standard
Visium spots should stay in coordinate form and use the Matrix or NUFFT path.


Use a precomputed graph
-----------------------

``Detector`` can read an adjacency or connectivity matrix from ``adata.obsp``:

.. code-block:: python

   detector = Detector(
       adata,
       kernel_method="car",
       backend="matrix",
       rho=0.9,
   ).setup_data(
       adata,
       obsp_key="connectivities",
   )

Connectivity graphs are symmetrized before isolated observations are removed,
so a node with incoming edges is retained. For ``is_distance=True``, Gaussian
and Matérn kernels require complete pairwise distances. A sparse distance
matrix must explicitly store every off-diagonal entry, including true zero
distances; an omitted neighbour edge is ambiguous and is rejected. Use spatial
coordinates when only a neighbour-distance graph is available.

At the lower level, construct a kernel directly from a matrix:

.. code-block:: python

   from sonic import MatrixKernel

   kernel = MatrixKernel.from_matrix(adjacency, method="moran")

For CAR with a custom graph, pass its precision matrix and set
``is_precision=True``. See :class:`~sonic.MatrixKernel.from_matrix` for the
accepted matrix types.


See also
--------

- :doc:`/guides/quickstart` for complete detection examples.
- :doc:`/guides/scaling` for backend memory and runtime.
- :doc:`/guides/theory` for kernel spectra, consistency, and null distributions.
- :doc:`/autoapi/sonic/kernels/index` for constructor parameters.
