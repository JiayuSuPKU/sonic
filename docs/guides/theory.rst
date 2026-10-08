Theoretical Results
===================

Our `accompanying paper <https://arxiv.org/pdf/2602.02825>`_ shows
that virtually every spatially-variable-gene (SVG) detection method
reduces to a single quadratic-form statistic, the Q-statistic. This
includes graph-based methods like Moran's I, parametric models, and
non-parametric dependence tests. The Q-statistic is

.. math::

   Q_n = \mathbf{z}^\top \mathbf{K} \mathbf{z},

where :math:`\mathbf{z}` is the standardised feature vector and
:math:`\mathbf{K}` is a kernel matrix that encodes spatial
structure. Under the null hypothesis of spatial independence,
:math:`Q_n` is often approximated by a weighted :math:`\chi^2`
distribution. Sample standardization requires finite-sample moment
corrections, described below. Moment-matching approximations give a fast
p-value for the Q-test. The kernel choice critically affects the consistency
and the power of the resulting test.

Theorem 1: Q-tests detect mean shifts only
------------------------------------------

All spatial Q-tests detect mean-shift patterns
(:math:`\mathbb{E}[\mathbf{x} \mid S = \mathbf{s}] \neq \mathbb{E}[\mathbf{x}]`).

This follows directly from using a linear kernel
:math:`l(x_i, x_j) = x_i x_j` in the quadratic form, which reduces
the conditional :math:`X \mid S = s_i` to its mean. To probe higher
moments (variance, distributional changes), swap in a non-linear
kernel. For example, apply a Gaussian or polynomial kernel to
:math:`\mathbf{z}^2` rather than :math:`\mathbf{z}`.

In spatial transcriptomics the distributional information is
typically absent. We observe only one realisation
:math:`(x_i, s_i)` per location, which blurs the line between mean
independence and statistical independence. Treating the signal as a
deterministic element of a Hilbert space
:math:`f \in L^2(\mathcal{S})` and applying spectrum theory of
kernel operators yields the consistency condition below.


Theorem 2: Consistency requires positive definiteness
-----------------------------------------------------

A spatial Q-test is universally consistent (power approaches 1 as
:math:`n \to \infty`) for every non-constant deterministic pattern
*if and only if* :math:`\mathbf{K}` is strictly positive definite.

Under :math:`H_0`, :math:`Q_n \approx \sum_i \lambda_i \chi^2_1`.
When some :math:`\lambda_i < 0` (indefinite kernel), signals aligned
with the negative eigenspace cancel signals aligned with the
positive eigenspace. We call this *spectral cancellation*, and it
costs the test power on composite patterns.

The implication is that you should pick a kernel with a non-negative
spectrum.

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Kernel
     - Spectrum
     - Consistency
   * - Gaussian
     - Strictly positive
     - Guaranteed.
   * - Matérn
     - Strictly positive
     - Guaranteed.
   * - Moran's I
     - Indefinite
     - Spectral cancellation.
   * - Graph Laplacian
     - Non-negative
     - Guaranteed (high-frequency Moran).
   * - CAR (inverse Laplacian)
     - Strictly positive
     - Guaranteed (low-frequency Moran).


CAR is a scalable correction to Moran's I
-----------------------------------------

The Conditional Autoregressive (CAR) kernel is strictly positive
definite:

.. math::

   \mathbf{K} = (\mathbf{I} - \rho \tilde{\mathbf{W}})^{-1},

where :math:`\tilde{\mathbf{W}}` is the row-normalised adjacency
matrix and :math:`0 < \rho < 1` is the autoregressive parameter
(default :math:`0.9`). The matrix
:math:`\mathbf{I} - \rho \tilde{\mathbf{W}}` is the CAR *precision*
matrix. It is sparse with :math:`\mathcal{O}(nk)` non-zeros even
though :math:`\mathbf{K}` itself is dense, which is what makes CAR
scalable on large graphs.

Key properties:

- Strictly positive definite for any :math:`0 < \rho < 1`.
- Theoretically consistent (Theorem 2).
- Scales via sparse-precision LU solves, with no
  :math:`\mathcal{O}(n^2)` materialisation.
- Polynomial spectral decay that emphasises smooth, large-scale
  patterns while keeping a heavy tail for mid- and high-frequency
  components.

Use CAR as the default for graph-flavoured spatial-pattern
detection.


.. _q-null-calibration:

Null calibration paths
----------------------

With sample-standardized features, Q is a **ratio** of quadratic forms.
Under an iid central Gaussian null, with :math:`m=n-1` and eigenvalues of
:math:`A=HKH` on the centered subspace,

.. math::

   Q = m\frac{\sum_i \lambda_i Z_i^2}{\sum_i Z_i^2},
   \qquad Z_i \overset{\mathrm{iid}}{\sim} N(0,1).

The unstandardized statistic instead has the Gaussian mixture null
:math:`\sum_i\lambda_i\chi^2_1`. SONIC keeps these models separate:
``compute_null_params(..., dirichlet_correction=True)`` uses finite-sample
ratio moments, while ``liu_sf(...)`` retains the mixture model
used by analytic comparison. Setting ``dirichlet_correction=False`` opts
into the unstandardized moment approximation for detection.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Path
     - Calibration
     - Tail
   * - Q, ``method="welch"``
     - Scaled central chi-square matching mean and variance.
     - Upper
   * - Q, ``method="moments"``
     - Corrected moments; fit selection below.
     - Upper
   * - Q, ``method="clt"``
     - Normal approximation, also used for recognized signed kernels.
     - Two-sided
   * - R
     - Zero-mean normal approximation using ``var_R``.
     - Two-sided
   * - ``liu_sf(...)``
     - Gaussian quadratic-form mixture approximation.
     - Upper

PSD matrix kernels default to Welch; PSD FFT/NUFFT kernels default to moment matching.
``compute_null_params(kernel)`` and ``spatial_q_test`` use the same automatic selection.
The :ref:`kernel-by-kernel default table <q-null-defaults>` includes all built-in
methods, precomputed matrices, and the detector defaults.
Recognized signed kernels, including Moran, default to two-sided CLT. They also
accept explicit ``method="moments"`` with ``dirichlet_correction=True``:
the finite-sample moment formulas require a symmetric kernel, not PSD.
Welch and the unstandardized moment fit remain restricted to PSD kernels.
A normal fallback within moment matching retains an upper tail; it does not become the
CLT test. A zero-variance null or constant input feature returns p=1.

For a Moran kernel, opt into upper-tail calibration with::

   from sonic import spatial_q_test
   from sonic.statistics import compute_null_params

   params = compute_null_params(kernel, method="moments")
   Q, p = spatial_q_test(data, kernel, null_params=params)

This tests unusually large Q (positive spatial association). The default CLT
test also detects unusually small Q. Both are analytic Gaussian-null
approximations, rather than a permutation calibration of Moran's statistic.
NUFFT Moran kernels use centered trace probes because their full signed
spectrum is unavailable; FFT kernels retain all signed spectral modes.

For finite-sample moment matching, the implementation applies this order:

1. Match four moments with Liu when its noncentral chi-square parameters
   are admissible (``fit="liu4"``).
2. Otherwise try a bounded four-moment beta fit (``fit="beta4"``).
3. Otherwise match positive skewness with a central chi-square
   (``fit="chi2_skewness"``).
4. Remaining symmetric/left-skewed shapes use ``fit="normal_fallback"``.
   This is an approximation, not a guarantee of tail calibration for
   strongly asymmetric shapes.

The cache records the statistical ``model`` and ``tail``. For moment matching,
``source`` identifies spectrum versus probe estimates, and
``params["q_fit"]["family"]`` / ``["fit"]`` identify the distribution
and matching rule. Null preparation happens once before feature chunking;
all backends use the same p-value evaluator.
Detector Z-scores use the same corrected mean and variance as their p-values;
for a non-normal fit, a Z-score does not imply a normal-tail p-value.

The cache contains the prepared fit and reporting moments, without
intermediate trace estimates. Rebuild it with ``compute_null_params`` after
changing the kernel or calibration settings. Incomplete caches are rejected.
Finite-sample fitting uses direct scaled traces of :math:`B=A-\operatorname{tr}(A)H/m`
to avoid cancellation near constant spectra. FFT kernels use their full
spectrum. Matrix moment fits use a full spectrum only for dense kernels with
at most 2000 observations, an already cached full spectrum, or an explicit
``k_eigen`` request. Sparse and precision-backed kernels use probes. NUFFT
moment calibration always defaults to analytic lower traces and 60 probes for
higher traces, even with a cached spectrum. To request its reduced
eigendecomposition, prepare the null explicitly:

.. code-block:: python

   null_params = compute_null_params(kernel, method="moments", nufft_spectrum=True)
   q, p = spatial_q_test(values, kernel, null_params=null_params)

This opt-in uses at most 2000 retained Fourier modes and falls back to probes
when the reduced spectrum is unavailable. It is approximate because of
spectral truncation and NUFFT numerical accuracy. An integer ``k_eigen``
requests top-k Lanczos instead; explicit ``n_probes`` overrides either spectrum
request. NUFFT Welch/CLT use only the analytic lower traces.

``n_probes`` controls a shared budget: 60 by default for moment matching and
15 for precision-backed Welch/CLT. Explicit Welch/CLT traces remain exact.
Precision kernels reuse the probes and first solves for higher traces, with
one additional application per probe for third and fourth powers. Probe
blocks target 256 MiB of temporary workspace, excluding the kernel and
existing caches. When the global centering offset must first be estimated,
large jobs replay the initial solves instead of retaining all solutions.
The same unshifted second trace supplies ``var_R``. Moment and probe
approximations need particular care for extreme tails.


R-test: bivariate spatial co-expression
---------------------------------------

The R-statistic extends the Q-test to two features at a time:

.. math::

   R_{xy} = \mathbf{x}^\top \mathbf{K} \mathbf{y},

where :math:`\mathbf{x}` and :math:`\mathbf{y}` are standardised.
Under :math:`H_0`,
:math:`R_{xy} \sim \mathcal{N}\bigl(0, \operatorname{tr}(\mathbf{K}^2)\bigr)`,
which gives a fast Normal p-value.

A typical workflow:

1. Identify SVGs via the univariate Q-test.
2. Test pairwise R-statistics among the top SVGs.
3. Control FDR across comparisons.


Drop-in replacement for Moran's I
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Method
     - Test consistency
     - Use case
   * - Moran's I
     - Spectral cancellation.
     - Classical autocorrelation; backwards compatibility only.
   * - Graph Laplacian
     - Guaranteed.
     - High-frequency, local variation.
   * - CAR
     - Guaranteed.
     - Low-frequency, smooth patterns.

In practice:

- On a graph, use the CAR kernel for consistent, high-power
  detection across functional patterns.
- On 2-D physical space, use the FFT- and NUFFT-accelerated forms
  of any PSD kernel. Matérn is a common starting point.


See also
--------

- :doc:`/guides/quickstart` for practical recipes.
- :doc:`/guides/kernels` for kernel selection and design.
- :doc:`/guides/scaling` for practical runtime and memory controls.
- :doc:`/autoapi/sonic/statistics/index` for the statistical-test
  API.
