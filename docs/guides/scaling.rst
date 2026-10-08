Scaling to Large Datasets
=========================

SONIC builds the spatial operator once, then applies it to features in batches.
The main choice is the backend; parallelism and chunk sizes are usually safe to
leave on ``"auto"``.


Choose the backend first
------------------------

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Backend
     - Best fit
     - Main tradeoff
   * - Matrix, dense
     - Small coordinate datasets, usually under about 5,000 observations
     - Simple and flexible, but memory grows quadratically.
   * - Matrix, sparse graph or CAR precision
     - Large neighbourhood graphs
     - Keeps graph geometry; setup can dominate for large CAR systems.
   * - NUFFT
     - Large irregular two-dimensional coordinates
     - Avoids a dense matrix; intended for Gaussian and Matérn coordinate kernels.
   * - FFT
     - Rectangular rasterized grids
     - Fastest grid path, but assumes periodic boundaries and does not support hex grids.

See :doc:`/guides/kernels` for the corresponding kernel choices.


Use the high-level defaults
---------------------------

For pattern detection, ``n_jobs=-1`` uses the available CPU cores and
``chunk_size="auto"`` limits temporary feature batches:

.. code-block:: python

   q_results = detector.compute_qstat(
       n_jobs=-1,
       chunk_size="auto",
   )

Use the same controls for the R-test. Restrict ``features_x`` and
``features_y`` before increasing parallelism: an all-pairs R-test grows much
faster than a Q-test over the same genes.

Fourier detectors balance job-level parallelism with FFT/NUFFT threads:

.. code-block:: python

   q_results = detector.compute_qstat(
       n_jobs="auto",
       workers="auto",
       chunk_size="auto",
   )

For pattern comparison, ``n_jobs`` splits work across samples when
``progress=False``. The default progress bar runs the outer loop sequentially.
The constructor controls the number of genes processed in each Fourier batch:

.. code-block:: python

   from sonic import Comparator

   comparison = Comparator(
       samples,
       nufft_chunk_size="auto",  # use fft_chunk_size for SpatialData grids
   ).compute_spectra(n_jobs=-1, progress=False)

Automatic batches are capped at 32 features for FFT, NUFFT and explicit matrix
kernels, and 4 for precision solves. These caps come from the scheduling sweep
in ``benchmarks/benchmark_scheduling.py``; they are conservative defaults, not
runtime autotuning. A shared 2 GiB estimated transient-workspace budget can
reduce them further. It excludes stored inputs, kernel matrices/factorizations
and retained results, so it is not a total-process memory limit.

Override that fixed budget with ``memory_budget_bytes`` on ``compute_qstat``,
``compute_rstat``, ``compute_spectra``, or the standalone ``spatial_q_test`` /
``spatial_r_test`` functions. Supply a positive integer number of bytes or a
size string such as ``"512 MB"``, ``"2 GiB"``, or ``"16 Gb"``. Units are
case-insensitive: KB/MB/GB/TB use powers of 1000, while KiB/MiB/GiB/TiB use powers
of 1024. Fractional sizes such as ``"1.5 GiB"`` are accepted if they resolve to
whole bytes. There is no adaptive ``"auto"`` memory budget:

.. code-block:: python

   q_results = detector.compute_qstat(
       n_jobs=8,
       chunk_size="auto",
       memory_budget_bytes="4 GiB",  # across all jobs
   )
   comparison.compute_spectra(
       n_jobs=8,
       progress=False,
       memory_budget_bytes="4 GiB",
   )

More jobs divide the budget rather than increase it automatically. For example,
eight jobs share the default 2 GiB estimate at 256 MiB per job; a 4 GiB override
allows 512 MiB per job. Raising the budget only increases chunks when the memory
limit is binding; the backend chunk caps still apply. Each call defaults to
2 GiB again if the override is omitted. Null-calibration workspace, process
overhead and the NUFFT R-test's retained standardized X block are also outside
the batch budget. Standalone bipartite NUFFT R-tests bypass chunk sizing.

``n_jobs`` is an upper bound on outer jobs. Automatic scheduling favors outer
jobs when enough batches or samples exist, then uses spare CPUs for transform
threads, capped at four. ``n_jobs * workers`` stays within the CPU count.
For NUFFT, the detector/comparator ``workers`` option sets FINUFFT's ``nthreads``;
low-level ``NUFFTKernel`` and ``power_spectrum_2d_nufft`` expose ``nthreads``
directly and default to one. ``workers=None`` selects one thread in high-level
calls. Explicit thread counts can override the cap of four.

The shared helpers live in :func:`sonic.utils.auto_chunk_size`,
:func:`sonic.utils.resolve_chunk_size`, and
:func:`sonic.utils.resolve_parallelism`. Chunk budgets use the resolved outer
job count; transform threads share their job's batch. Comparators with
``progress=True`` therefore budget for one outer job.

Start with these defaults. Explicit chunk sizes bypass automatic memory sizing;
raise them only after profiling shows that small batches are the bottleneck.


Sparse expression matrices
--------------------------

``AnnData.X`` may remain sparse. SONIC densifies only the current feature
batch, not the complete observation-by-gene matrix. A smaller ``chunk_size``
or ``nufft_chunk_size`` reduces this temporary allocation.

Filtering low-prevalence features during setup also saves work:

.. code-block:: python

   detector.setup_data(
       adata,
       obsm_key="spatial",
       min_cells_frac=0.05,
   )


Reuse work in low-level loops
-----------------------------

The detector classes build the kernel once and compute null parameters once per
bulk test. When using the array-level API yourself, compute null parameters
once and reuse them:

.. code-block:: python

   from sonic import compute_null_params, spatial_q_test

   null_params = compute_null_params(kernel, method="moments")
   results = [
       spatial_q_test(values, kernel, null_params=null_params)
       for values in feature_vectors
   ]

Reuse the same kernel and null parameters across features.


Rough cost guide
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Operator
     - Setup
     - Memory
     - Per feature
   * - Dense matrix
     - Up to ``O(n²)``
     - ``O(n²)``
     - ``O(n²)``
   * - Sparse neighbourhood graph
     - ``O(nk)`` after graph construction
     - ``O(nk)``
     - ``O(nk)``
   * - CAR sparse precision
     - Sparse factorization
     - Sparse factors
     - Sparse solve
   * - FFT or NUFFT
     - Frequency-grid setup
     - Near ``O(n)``
     - Near ``O(n log n)``

Here ``n`` is the number of observations or grid cells and ``k`` is the number
of graph neighbours. These are planning estimates; grid shape, sparsity, and
threading determine actual runtime.


When a run is slow or memory-heavy
----------------------------------

1. Check that the backend matches the data layout.
2. Filter features before the test rather than after it.
3. Keep ``chunk_size`` on ``"auto"`` or lower it if memory is the limit.
4. Avoid nested oversubscription: reduce ``n_jobs`` when each FFT or NUFFT call
   already uses several threads.
5. For R-tests, reduce the feature sets before changing compute settings.

FFT uses periodic rectangular boundaries. If that geometry is not appropriate,
use coordinates or an explicit graph even when the grid backend would be
faster.


See also
--------

- :doc:`/guides/quickstart` for complete workflows.
- :doc:`/guides/kernels` for backend and kernel selection.
- :doc:`/guides/theory` for null approximations and computational details.
