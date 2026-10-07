"""Spatial statistics and null calibration.

Q calibration has four steps: collect kernel traces, compute null moments,
fit a distribution, and evaluate its tail. ``compute_null_params`` prepares
the fit; ``_prepare_q_null`` resolves defaults and validates supplied fits;
``_q_pvalues`` evaluates the prepared fit for every backend.

The null paths are deliberately distinct:

* Sample-standardized Q: finite-sample ratio moments. ``moments`` selects a
  four-moment Liu or beta fit, a central chi-square fit, or a normal fallback.
  Symmetric indefinite kernels, including Moran, can use this path explicitly.
* Gaussian quadratic forms: ``liu_sf(...)`` retains the original
  eigenvalue-mixture approximation used by comparison.
* Welch Q: mean/variance chi-square fit, upper tail.
* CLT Q: normal approximation, two-sided. A normal fallback within moment matching
  still uses the upper tail; it is not the CLT test.
* R: zero-mean normal approximation, two-sided, using ``var_R``.

Finite-sample fitting computes centered/scaled traces directly for numerical
stability. Zero-variance nulls return p=1.

Automatic Q defaults (also used by detectors) are Welch for MatrixKernel and
moment matching for FFTKernel/NUFFTKernel with Gaussian, Matérn, CAR or graph
Laplacian kernels. Moran, and any detected signed Fourier spectrum, use CLT.
Custom precomputed matrices otherwise default to Welch under the caller's PSD
assumption; their signs are not inferred by an extra eigendecomposition.
NUFFT moment calibration defaults to analytic lower traces and 60 probes
for higher traces; its reduced eigendecomposition requires an explicit opt-in.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import beta, chi2, ncx2, norm
from sklearn.utils.sparsefuncs import mean_variance_axis
from tqdm import tqdm

from sonic.kernels import Kernel
from sonic.utils import (
    _DEFAULT_CHUNK_BUDGET,
    _parse_memory_budget,
    auto_chunk_size,
    resolve_chunk_size,
)

__all__ = [
    "apply_bh_correction",
    "auto_chunk_size",
    "cauchy_combine",
    "resolve_chunk_size",
    "compute_null_params",
    "liu_sf",
    "spatial_q_test",
    "spatial_r_test",
]

# Bound new probe workspace, allowing for eight live float64 blocks.
_TRACE_PROBE_BUDGET_BYTES = 256 * (1 << 20)
# Full dense spectra are useful for small problems; larger nulls use probes.
_DENSE_NULL_SPECTRUM_LIMIT = 2000


def apply_bh_correction(p_values: np.ndarray | pd.Series | Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values for raw p-values.

    Non-finite input values (NaN / inf) are ignored during correction and remain
    ``NaN`` in the output. Finite raw p-values must lie in ``[0, 1]``. The
    returned array preserves the input shape.

    Parameters
    ----------
    p_values : array-like of float
        Raw p-values to adjust.

    Returns
    -------
    numpy.ndarray
        Benjamini-Hochberg adjusted p-values with the same shape as
        ``p_values``.
    """
    pvals = np.asarray(p_values, dtype=float)
    flat = pvals.ravel()
    p_adj = np.full(flat.shape, np.nan, dtype=float)

    valid_mask = np.isfinite(flat)
    m = valid_mask.sum()
    if m == 0:
        return p_adj.reshape(pvals.shape)

    valid_pvals = flat[valid_mask]
    if np.any((valid_pvals < 0.0) | (valid_pvals > 1.0)):
        raise ValueError("raw p-values must be finite values in [0, 1], or NaN/inf.")

    p_sorted_idx = np.argsort(valid_pvals)
    p_sorted = valid_pvals[p_sorted_idx]
    ranks = np.arange(1, m + 1)

    bh_vals = p_sorted * m / ranks
    bh_adj = np.minimum.accumulate(bh_vals[::-1])[::-1]
    bh_adj = np.clip(bh_adj, 0, 1)

    p_adj_indices = np.where(valid_mask)[0][p_sorted_idx]
    p_adj[p_adj_indices] = bh_adj

    return p_adj.reshape(pvals.shape)


def cauchy_combine(
    pvals: np.ndarray, axis: int = -1, *, where: np.ndarray | None = None
) -> np.ndarray:
    """
    Cauchy combination test.

    For p-values :math:`p_1, \\dots, p_K`, forms
    :math:`T = \\frac{1}{K}\\sum_k \\cot(\\pi p_k)` and returns
    the analytic upper-tail probability under the standard Cauchy null,
    :math:`p = \\arctan2(1, T) / \\pi`. Robust to arbitrary dependence
    between the input p-values — that is the whole point of Cauchy
    combination — so it is safe to apply over correlated frequency bins
    without decorrelating them first.

    Parameters
    ----------
    pvals : np.ndarray
        Input p-values in ``[0, 1]``. Values at the exact endpoints are
        clipped away from them to keep :math:`\\tan` finite.
    axis : int, default -1
        Axis along which to combine.
    where : np.ndarray of bool, optional
        Broadcastable mask selecting informative tests. Slices with no
        selected tests return 1. Unselected tests receive no weight.

    Returns
    -------
    np.ndarray
        Combined p-value(s); one less axis than ``pvals``.
    """
    pvals = np.asarray(pvals, dtype=float)
    clipped = np.clip(pvals, np.finfo(float).tiny, np.nextafter(1.0, 0.0))
    # cot(pi * p) is equivalent to tan(pi * (0.5 - p)) but avoids precision
    # loss for the ultra-small p-values produced by analytic Welch tests.
    cot = 1.0 / np.tan(np.pi * clipped)
    if where is None:
        T = np.mean(cot, axis=axis)
    else:
        where = np.broadcast_to(where, pvals.shape)
        count = where.sum(axis=axis)
        T = np.divide(
            np.sum(cot, axis=axis, where=where),
            count,
            out=np.full(np.shape(count), -np.inf),
            where=count > 0,
        )
    return np.arctan2(1.0, T) / np.pi


def _fit_standardized_q(
    mean: float, b: dict[int, float], scale: float, n: int, kurtosis: bool = False
) -> dict:
    r"""Fit the sample-standardized Q from traces of ``B / scale``.

    With ``m = n-1``, ``B = HKH - tr(HKH) H/m`` and ``b_j = tr(B^j)``,
    the Gaussian-null cumulants are

    .. math::

        \kappa_2 &= 2m b_2/(m+2), \\
        \kappa_3 &= 8m^2 b_3/((m+2)(m+4)), \\
        \kappa_4 &= \frac{48m^3}{(m+2)(m+4)(m+6)}
          \left[b_4-\frac{2(m+3)b_2^2}{m(m+2)}\right].

    Positive skew uses Liu when admissible, otherwise its three-moment
    central-chi-square fallback. Shapes with insufficient kurtosis for Liu
    (including negative excess kurtosis) use a four-moment beta fit when
    possible. Remaining symmetric/left-skewed shapes use a normal fallback.
    These formulas require symmetry, not positive semidefiniteness. All fits
    are moment approximations, not exact tail probabilities.
    """
    # 1. Convert scaled B traces to the moments of sample-standardized Q.
    m = n - 1
    coef = {
        "model": "standardized_q",
        "family": "normal",
        "fit": "degenerate",
        "mu_Q": float(mean),
        "sigma_Q": 0.0,
    }
    if b[2] <= 0 or scale == 0:
        return coef
    coef["sigma_Q"] = float(scale * np.sqrt(2 * m * b[2] / (m + 2)))
    s1 = np.sqrt(m * (m + 2)) / (m + 4) * b[3] / b[2] ** 1.5
    s2 = (m * (m + 2) * b[4] / b[2] ** 2 - 2 * (m + 3)) / ((m + 4) * (m + 6))
    if not np.isfinite(s1) or not np.isfinite(s2):
        raise ValueError("Non-finite standardized Q moments.")

    coef["fit"] = "normal_fallback"

    # 2. Match all four moments with Liu only when its parameters are valid.
    # Noncentral chi-square requires 1 <= s1^2/s2 < 9/8 and positive skew.
    if s1 > 1e-7 and s2 > 0 and s2 <= s1 * s1 < 1.125 * s2:
        fit = "liu4"
        a = 1 / (s1 - np.sqrt(max(s1 * s1 - s2, 0.0)))
        delta = max(s1 * a**3 - a**2, 0.0)
        dof = a**2 - 2 * delta
    else:
        skew, excess = np.sqrt(8) * s1, 12 * s2
        # 3. Try a bounded four-moment beta fit for shapes Liu cannot represent.
        # Solve its equations for a+b and (b-a)/(a+b).
        denominator = 3 * skew**2 - 2 * excess
        if denominator > 1e-12:
            total = 6 * (2 + excess - skew**2) / denominator
            if total > 0:
                ratio = skew * (total + 2) / np.sqrt(skew**2 * (total + 2) ** 2 + 16 * (total + 1))
                alpha, beta_shape = total * (1 - ratio) / 2, total * (1 + ratio) / 2
                if min(alpha, beta_shape) > 0:
                    coef.update(
                        family="beta",
                        fit="beta4",
                        alpha=float(alpha),
                        beta=float(beta_shape),
                        mu_x=float(alpha / total),
                        sigma_x=float(np.sqrt(alpha * beta_shape / (total + 1)) / total),
                    )
                    return coef
        # 4. Preserve the existing fallback policy: normal for remaining
        # symmetric/left-skewed shapes, central chi-square for positive skew.
        if s1 <= 1e-7:
            return coef
        dof = 1 / (s2 if kurtosis and s2 > 0 else s1 * s1)
        fit = "chi2_kurtosis" if kurtosis and s2 > 0 else "chi2_skewness"
        delta = 0.0
    if dof <= 0 or not np.isfinite(dof + delta):
        return coef
    coef.update(
        family="ncx2",
        fit=fit,
        mu_x=float(dof + delta),
        sigma_x=float(np.sqrt(2 * (dof + 2 * delta))),
        dof_x=float(dof),
        delta_x=float(delta),
    )
    return coef


def _spectrum_traces(lambs: np.ndarray, n: int, dofs: np.ndarray | None = None) -> dict:
    """Collect the Q mean and scaled B traces from a centered spectrum.

    Return ``mean``, ``powers``, ``scale`` and ``source`` in the same format
    as :func:`_estimate_kernel_traces`. Include omitted zero modes up to the
    n-1 centered dimensions; remove only the structural constant mode.
    """
    if n < 2:
        raise ValueError("Sample-standardized Q requires n >= 2.")
    lam = np.asarray(lambs, dtype=float).ravel()
    weights = np.ones_like(lam) if dofs is None else np.broadcast_to(dofs, lam.shape).copy()
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Kernel eigenvalues and non-negative degrees of freedom must be finite.")
    m = n - 1
    missing = m - float(weights.sum())
    if missing < 0 and lam.size:
        # A full HKH spectrum includes the removed constant direction once.
        dc = int(np.argmin(np.abs(lam)))
        tolerance = 64 * np.finfo(float).eps * np.max(np.abs(lam))
        if missing == -1 and abs(lam[dc]) <= tolerance and weights[dc] >= 1:
            weights[dc] -= 1
            missing = 0.0
    if missing < 0:
        raise ValueError("The kernel spectrum exceeds the n-1 centered dimensions.")
    keep = weights > 0
    lam, weights = lam[keep], weights[keep]
    mean = float(np.sum(weights * lam))
    offset = mean / m
    shifted = lam - offset
    scale = max(float(np.max(np.abs(shifted), initial=0)), abs(offset) if missing else 0.0)
    if scale <= 64 * np.finfo(float).eps * np.max(np.abs(lam), initial=0):
        return {
            "mean": mean,
            "powers": dict.fromkeys((2, 3, 4), 0.0),
            "scale": 0.0,
            "source": "spectrum",
        }
    shifted /= scale
    b = {
        p: float(np.sum(weights * shifted**p) + missing * (-offset / scale) ** p) for p in (2, 3, 4)
    }
    return {"mean": mean, "powers": b, "scale": scale, "source": "spectrum"}


def _fit_gaussian_quadratic(
    mean: float, powers: dict[int, float], kurtosis: bool = False, *, scale: float = 1.0
) -> dict:
    r"""Fit the unstandardized Gaussian quadratic form with Liu's approximation.

    ``powers[p]`` is the power sum of eigenvalues divided by ``scale`` (including
    degrees of freedom and noncentrality weights); the statistical cumulant is
    ``2**(p-1) * (p-1)! * powers[p] * scale**p``. No standardization correction belongs
    here. Inputs from ``liu_sf`` are scaled before taking powers; all shape
    calculations are dimensionless and use no additive numerical offsets.
    """
    coef = {
        "model": "gaussian_quadratic",
        "family": "normal",
        "fit": "degenerate" if powers[2] <= 0 else "normal_fallback",
        "mu_Q": float(mean),
        "sigma_Q": float(np.sqrt(2.0) * np.sqrt(max(powers[2], 0.0)) * scale),
    }
    if powers[2] <= 0:
        return coef
    root = np.sqrt(powers[2])
    s1 = powers[3] / root / powers[2]
    s2 = powers[4] / powers[2] / powers[2]
    if s1 <= 0 or s2 <= 0:
        return coef
    s12 = s1**2
    if s12 > s2:
        # Rationalize 1 / (s1 - sqrt(s1² - s2)) to avoid cancellation.
        a = (s1 + np.sqrt(s12 - s2)) / s2
        delta_x = max(s1 * a**3 - a**2, 0.0)
        dof_x = a**2 - 2.0 * delta_x
        fit = "liu4"
    else:
        fit = "chi2_kurtosis" if kurtosis else "chi2_skewness"
        delta_x = 0.0
        dof_x = 1.0 / (s2 if kurtosis else s12)
    if dof_x <= 0 or not np.isfinite(dof_x + delta_x):
        return coef
    coef.update(
        {
            "family": "ncx2",
            "fit": fit,
            "mu_x": float(dof_x + delta_x),
            "sigma_x": float(np.sqrt(2 * (dof_x + 2 * delta_x))),
            "dof_x": float(dof_x),
            "delta_x": float(delta_x),
        }
    )
    return coef


def _prepare_moment_fit(
    lambs: np.ndarray,
    dofs: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
    kurtosis: bool = False,
    n: int | None = None,
) -> dict:
    """Prepare a moment fit from the kernel eigenvalue spectrum.

    With ``n``, accumulate centered/scaled powers directly. Omitted zero
    eigenvalues are included up to the ``n-1`` centered dimensions; a full
    ``HKH`` spectrum may include its one extra zero constant mode. Without
    ``n``, preserve the unstandardized Gaussian-mixture fit.

    Parameters
    ----------
    lambs : np.ndarray
        Eigenvalues of ``K``, shape ``(n_evals,)``.
    dofs, deltas : np.ndarray, optional
        Per-eigenvalue degrees of freedom and non-centrality parameters.
        Default to central chi-squared (ones, zeros).
    kurtosis : bool, default False
        Use the kurtosis-based edge-case approximation.
    n : int, optional
        Sample size for finite-sample moment correction. Noncentral variables
        are unsupported in this mode; ``dofs`` must fit in ``n-1`` dimensions.
    """
    lambs = np.asarray(lambs, dtype=float)
    if n is not None:
        if deltas is not None and np.any(np.asarray(deltas) != 0):
            raise ValueError("Sample-standardized Q requires central variables (deltas=0).")
        traces = _spectrum_traces(lambs, n, dofs)
        return _fit_standardized_q(traces["mean"], traces["powers"], traces["scale"], n, kurtosis)
    if dofs is None:
        dofs = np.ones_like(lambs)
    else:
        dofs = np.asarray(dofs, dtype=float)
    if deltas is None:
        deltas = np.zeros_like(lambs)
    else:
        deltas = np.asarray(deltas, dtype=float)
    if not (np.isfinite(lambs).all() and np.isfinite(dofs).all() and np.isfinite(deltas).all()):
        raise ValueError(
            "Mixture eigenvalues, degrees of freedom and noncentralities must be finite."
        )
    if np.any(dofs < 0) or np.any(deltas < 0):
        raise ValueError("Mixture degrees of freedom and noncentralities must be non-negative.")
    scale = float(np.max(np.abs(lambs), initial=0.0))
    scaled = lambs / scale if scale else lambs
    mean = float(np.sum(scaled * (dofs + deltas)) * scale)
    powers = {p: float(np.sum(scaled**p * (dofs + p * deltas))) for p in (2, 3, 4)}
    return _fit_gaussian_quadratic(mean, powers, kurtosis=kurtosis, scale=scale)


def _centered_kernel(kernel: Kernel) -> Kernel:
    """Use the standardized operator's trace view without changing caller state."""
    if getattr(kernel, "centering", True):
        return kernel
    centered = copy(kernel)
    centered.centering = True
    return centered


def _estimate_kernel_traces(  # noqa: C901
    kernel: Kernel,
    n_probes: int = 60,
    rng_seed: int = 0,
    use_analytic_traces: bool = True,
    *,
    centered: bool = False,
    max_order: int = 4,
) -> dict:
    r"""Collect Q traces and the R variance with one shared probe budget.

    ``max_order=2`` returns the mean and second trace of A=HKH for Welch/CLT.
    Order four estimates scaled powers of A, or of the contrast
    B=A-tr(A)H/(n-1) when ``centered=True``. These are spectral power sums,
    not statistical cumulants. ``var_R`` is the second trace of HKH.

    Explicit kernels retain analytic lower traces where available. Precision
    kernels reuse cached raw probes and solves, then project them to HKH.
    Higher powers use a second application to the same probes. Large probe
    sets are processed in bounded blocks; when the global mean is estimated,
    replaying the first application costs a third pass instead of storing all
    solutions. Every block uses the same offset. Draws are independent of
    block size. FFT kernels always use their inexpensive full spectrum.
    """
    from sonic.kernels.fft import FFTKernel

    kernel = _centered_kernel(kernel)
    if (
        isinstance(n_probes, (bool, np.bool_))
        or not np.isfinite(n_probes)
        or n_probes < 1
        or int(n_probes) != n_probes
    ):
        raise ValueError("n_probes must be a positive integer.")
    if max_order not in (2, 4):
        raise ValueError("max_order must be 2 or 4.")
    n_probes = int(n_probes)
    n, m = int(kernel.n), int(kernel.n) - 1
    if m < 1:
        raise ValueError("Spatial null calibration requires n >= 2.")
    if isinstance(kernel, FFTKernel):
        lam = np.asarray(kernel.eigenvalues(return_full_layout=True), dtype=float).copy()
        lam[0] = 0.0  # Standardized Q and R project out the constant mode.
        var_R = float(np.sum(lam**2))
        if centered and max_order == 4:
            return {**_spectrum_traces(lam, n), "var_R": var_R}
        scale = float(np.max(np.abs(lam))) if max_order == 4 else 1.0
        scaled = lam / scale if scale else lam
        return {
            "mean": float(lam.sum()),
            "powers": {p: float(np.sum(scaled**p)) for p in range(2, max_order + 1)},
            "scale": scale,
            "source": "spectrum",
            "var_R": var_R,
        }

    precision = bool(getattr(kernel, "stores_precision", False))
    c1 = c2 = var_R = None
    analytic = False
    # The constant-vector solve also converts cached KV into HKHV, without
    # repeating the first batch of precision solves or changing kernel state.
    K1 = None
    if precision:
        K1 = np.asarray(kernel._apply_K_dense(np.ones((n, 1)))).ravel()
    if use_analytic_traces and not precision:
        try:
            c2 = float(kernel.square_trace())
            c1 = float(kernel.trace())
            var_R = c2
            analytic = True
        except (ValueError, NotImplementedError):
            c1 = c2 = var_R = None

    block_size = min(n_probes, max(1, _TRACE_PROBE_BUDGET_BYTES // (8 * n * 8)))
    cached = getattr(kernel, "_trace_rvs_cache", None) if precision and rng_seed == 0 else None
    if cached is not None and cached["n_vectors"] != n_probes:
        cached = None
    if precision and rng_seed == 0 and cached is None and block_size == n_probes:
        cached = kernel._get_rvs_trace_cache(n_probes)

    def probe_blocks():
        # This iterator is replayed only when the global offset is not known
        # and the first-pass solutions do not fit the workspace budget.
        rng = np.random.default_rng(rng_seed)
        for start in range(0, n_probes, block_size):
            count = min(block_size, n_probes - start)
            if cached is not None:
                V = cached["rvs"][:, start : start + count]
                Y = cached["Y"][:, start : start + count]
            else:
                V = rng.choice([-1.0, 1.0], size=(count, n)).T
                Y = np.asarray(kernel._apply_K_dense(V) if precision else kernel.Kx(V))
            means = V.mean(axis=0)
            U = Y if K1 is None else Y - K1[:, None] * means
            U = U - U.mean(axis=0)
            yield V - means, U

    # One extra pass is needed only to learn the global B offset. For small
    # jobs retain its single block; large jobs replay identical probe draws.
    mean_pass = centered and max_order == 4 and not analytic
    saved = None
    sums = np.zeros(3)  # <V,AV>, ||AV||², ||V||²
    powers = dict.fromkeys(range(2, max_order + 1), 0.0)
    scale = 0.0 if max_order == 4 else 1.0
    largest_action = 0.0
    for phase in range(2 if mean_pass else 1):
        blocks = [saved] if phase == 1 and saved is not None else probe_blocks()
        for V, U in blocks:
            if phase == 0:
                sums += [np.sum(V * U), np.sum(U * U), np.sum(V * V)]
                largest_action = max(largest_action, float(np.max(np.abs(U))))
            if mean_pass and phase == 0:
                if block_size == n_probes:
                    saved = (V, U)
                continue
            if max_order == 2:
                continue
            if centered:
                offset = c1 / m
                U = U - offset * V
            block_scale = float(np.max(np.abs(U)))
            if block_scale == 0 or (
                centered
                and block_scale <= 64 * np.finfo(float).eps * max(largest_action, abs(offset))
            ):
                continue
            # Rescale accumulated sums when a later block has a larger
            # action, avoiding fourth-power overflow/underflow.
            new_scale = max(scale, block_scale)
            for p in powers:
                powers[p] *= (scale / new_scale) ** p
            scale = new_scale
            U /= scale
            W = np.asarray(kernel.Kx(U))
            W -= W.mean(axis=0)
            if centered:
                W -= offset * U
            W /= scale
            powers[2] += float(np.sum(U * U))
            powers[3] += float(np.sum(U * W))
            powers[4] += float(np.sum(W * W))
        if phase == 0:
            if centered and sums[2] == 0:
                raise ValueError("All centered probes were constant; increase n_probes.")
            weight = m / sums[2] if centered else 1.0 / n_probes
            if not analytic:
                c1, c2 = float(weight * sums[0]), float(weight * sums[1])
                var_R = c2

    powers = {p: float(weight * value) for p, value in powers.items()}
    if max_order == 2:
        powers[2] = c2
    elif scale > 0:
        if not centered:
            powers[2] = c2 / scale / scale
        else:
            b2_exact = c2 - c1**2 / m
            if analytic and b2_exact > np.sqrt(np.finfo(float).eps) * c2:
                powers[2] = b2_exact / scale / scale
    return {"mean": c1, "powers": powers, "scale": scale, "source": "probes", "var_R": var_R}


def _moment_sf(t: float | np.ndarray, coef: dict) -> np.ndarray:
    """Evaluate ``Pr(Q > t)`` from a prepared moment fit.

    ``coef`` is the dict returned by :func:`_prepare_moment_fit`. Broadcasts across
    ``t`` using the fitted chi-square, beta, or normal survival function.
    """
    t = np.asarray(t, dtype=float)
    if coef["sigma_Q"] <= 0:
        if coef.get("model") == "gaussian_quadratic":
            return np.asarray(t <= coef["mu_Q"], dtype=float)
        return np.ones_like(t)
    z = (t - coef["mu_Q"]) / coef["sigma_Q"]
    if coef["family"] == "normal":
        return norm.sf(z)
    x = z * coef["sigma_x"] + coef["mu_x"]
    if coef["family"] == "beta":
        return beta.sf(x, coef["alpha"], coef["beta"])
    if coef["delta_x"] == 0:
        return chi2.sf(x, coef["dof_x"])
    return ncx2.sf(x, coef["dof_x"], coef["delta_x"])


def _prepare_q_null(kernel: Kernel, null_params: dict | None = None) -> dict:  # noqa: C901
    """Resolve defaults or validate a prepared Q cache before feature chunking.

    Matrix kernels default to Welch; Fourier kernels default to moments;
    recognized signed kernels default to CLT. Supply either a method-only
    request or the cache returned by ``compute_null_params``. Partial caches
    are rejected instead of silently rebuilding a different null model.
    """
    # Check the method even for cached fits, so signed-kernel checks apply.
    requested = None if null_params is None else null_params.get("method")
    fit = None if null_params is None else null_params.get("q_fit")
    fit_model = fit.get("model") if isinstance(fit, dict) else None
    model = None if null_params is None else null_params.get("model", fit_model)
    method = _resolve_q_null_method(
        kernel,
        requested,
        dirichlet_correction=model != "gaussian_quadratic",
    )
    if null_params is None or null_params.keys() <= {"method"}:
        if not hasattr(kernel, "square_trace"):
            raise ValueError(
                "A raw kernel matrix requires null_params; wrap it in MatrixKernel "
                "or pass a prepared null from compute_null_params."
            )
        return compute_null_params(kernel, method=method)

    # Reuse fitted parameters without modifying the caller's dictionary.
    required = {"method", "mean_Q", "var_Q"}
    if method == "moments":
        required.add("q_fit")
    elif method == "welch":
        required.update(("scale_g", "df_h"))
    missing = sorted(key for key in required if null_params.get(key) is None)
    if missing:
        raise ValueError(
            f"Incomplete null_params for method={method!r}: missing {', '.join(missing)}. "
            "Build the cache with compute_null_params."
        )
    if model is not None and model not in {"standardized_q", "gaussian_quadratic"}:
        raise ValueError("Invalid null_params model; rebuild with compute_null_params.")
    expected_tail = "two-sided" if method == "clt" else "upper"
    if null_params.get("tail", expected_tail) != expected_tail:
        raise ValueError(
            "null_params tail does not match its method; rebuild with compute_null_params."
        )
    if method == "moments":
        if not isinstance(fit, dict) or fit.get("family") not in {"normal", "beta", "ncx2"}:
            raise ValueError(
                "Incomplete or invalid q_fit family; rebuild with compute_null_params."
            )
        required_fit = {"model", "mu_Q", "sigma_Q"}
        if fit["family"] != "normal":
            required_fit.update(("mu_x", "sigma_x"))
            required_fit.update(
                ("alpha", "beta") if fit["family"] == "beta" else ("dof_x", "delta_x")
            )
        if any(fit.get(key) is None for key in required_fit):
            raise ValueError("Incomplete q_fit; rebuild with compute_null_params.")
        if fit_model != model or (fit_model == "gaussian_quadratic" and fit["family"] == "beta"):
            raise ValueError(
                "q_fit model does not match null_params; rebuild with compute_null_params."
            )
    return null_params


def _q_pvalues(Q: float | np.ndarray, params: dict) -> np.ndarray:
    """Evaluate a prepared Q null without estimating traces or fitting again.

    Welch and moment fits use the upper tail. CLT is two-sided. A zero
    variance returns p=1; positive variances use the same rule on all backends.
    """
    Q = np.asarray(Q, dtype=float)
    method = params["method"]
    if method == "moments":
        return _moment_sf(Q, params["q_fit"])
    if method == "welch":
        return _welch_apply(Q, params)
    if method == "clt":
        sigma = np.sqrt(max(params["var_Q"], 0.0))
        if sigma <= 0:
            return np.ones_like(Q)
        z = (Q - params["mean_Q"]) / sigma
        return chi2.sf(z * z, df=1)  # Two-sided normal tail; moment fits always use the upper tail.
    raise ValueError(f"Unknown null approximation method: {method!r}")


def _welch_apply(t: float | np.ndarray, params: dict) -> np.ndarray:
    """Apply the Welch null, treating a deterministic Q as uninformative."""
    t = np.asarray(t, dtype=float)
    if params["var_Q"] <= 0 or params["mean_Q"] <= 0:
        return np.ones_like(t)
    g, h = params["scale_g"], params["df_h"]
    if g <= 0 or h <= 0:
        return np.ones_like(t)
    return chi2.sf(t / g, df=h)


def liu_sf(
    t: float | np.ndarray,
    lambs: np.ndarray,
    dofs: np.ndarray | None = None,
    deltas: np.ndarray | None = None,
    kurtosis: bool = False,
) -> float | np.ndarray:
    """Approximate the upper tail of a Gaussian quadratic form using Liu's fit.

    ``lambs`` are eigenvalue weights, ``dofs`` are degrees of freedom
    (default ones), and ``deltas`` are noncentralities (default zeros).
    ``kurtosis=True`` selects the kurtosis-based central chi-square fallback.
    The returned probability has the same shape as ``t``. Multiplying both
    ``t`` and ``lambs`` by a positive constant preserves the fitted tail.
    A zero-variance mixture returns p=1 at or below its mean and p=0 above it.

    This is the unstandardized eigenvalue-mixture model used by comparison.
    For sample-standardized spatial Q, use ``compute_null_params`` with
    ``method="moments"`` and pass that cache to ``spatial_q_test``.
    """
    coef = _prepare_moment_fit(lambs, dofs=dofs, deltas=deltas, kurtosis=kurtosis)
    return _moment_sf(t, coef)


def _resolve_q_null_method(
    kernel: Kernel,
    method: str | None = None,
    *,
    dirichlet_correction: bool = True,
) -> str:
    """Resolve the same automatic Q method for direct calls and detectors."""
    from sonic.kernels.fft import FFTKernel
    from sonic.kernels.nufft import NUFFTKernel

    if method is not None and method not in {"clt", "welch", "moments"}:
        raise ValueError(f"Unknown null method {method!r}; choose 'clt', 'welch', or 'moments'.")
    indefinite = getattr(kernel, "method", None) == "moran"
    # NUFFT uses this same Fourier spectrum to apply the irregular-point kernel.
    fourier_kernel = getattr(kernel, "_fft_kernel", kernel)
    spectrum = getattr(fourier_kernel, "spectrum", None)
    if spectrum is not None:
        tolerance = 64 * np.finfo(float).eps * float(np.max(np.abs(spectrum)))
        indefinite |= bool(np.min(spectrum) < -tolerance)
    if method is None:
        if indefinite:
            return "clt"
        return "moments" if isinstance(kernel, (FFTKernel, NUFFTKernel)) else "welch"
    # Finite-sample moments depend on centered eigenvalue contrasts, not their signs.
    # Welch and the unstandardized Gaussian-mixture fit still require PSD kernels.
    if indefinite and (method == "welch" or (method == "moments" and not dirichlet_correction)):
        raise ValueError(
            "Welch and unstandardized moment fits require a PSD kernel. "
            "For indefinite kernels, use method='clt' or method='moments' "
            "with dirichlet_correction=True."
        )
    return method


def compute_null_params(  # noqa: C901
    kernel: Kernel,
    method: str | None = None,
    k_eigen: int | None = None,
    dirichlet_correction: bool = True,
    n_probes: int | None = None,
    *,
    nufft_spectrum: bool = False,
) -> dict:
    r"""Prepare and cache the Q null, plus the variance needed by the R-test.

    Calibration proceeds once per kernel: collect traces, compute moments,
    then fit the requested distribution. All Q backends share the resulting
    cache and p-value evaluator. See the module docstring for the path table.

    Parameters
    ----------
    kernel : Kernel
        Matrix, FFT, NUFFT, or a compatible kernel object.
    method : {'clt', 'welch', 'moments'} or None, default None
        Automatic selection matches ``spatial_q_test``: Welch for matrix
        kernels, moments for Fourier kernels, and CLT for recognized signed kernels.
        ``clt`` uses a two-sided normal approximation; ``welch`` uses an
        upper-tail chi-square fit to mean/variance; ``moments`` uses the first
        four moments to select Liu, beta, central chi-square, or normal fits.
        All moment fits use the upper tail. Symmetric indefinite kernels,
        including Moran, support ``moments`` with ``dirichlet_correction=True``.
        Their automatic default remains two-sided CLT; Welch requires PSD.
    k_eigen : int, optional
        Use a truncated eigenvalue spectrum for moment matching. This is approximate:
        omitted modes are treated as zero. Automatic spectra are limited to
        FFT and dense matrices with at most 2000 samples (or a cached full
        spectrum). An explicit integer selects top-k Lanczos for NUFFT;
        otherwise NUFFT uses probes unless ``nufft_spectrum=True``.
        Ignored when ``n_probes`` is specified.
    dirichlet_correction : bool, default True
        Use moments of sample-standardized Q (a ratio of quadratic forms).
        Correct variance for every method, and skewness/kurtosis for moment matching.
        ``False`` retains the unstandardized Gaussian quadratic-form model;
        recognized indefinite kernels reject this option with ``moments``.
    n_probes : int, optional
        Probe budget: defaults to 60 for moments and 15 for precision-backed
        Welch/CLT. Explicit Welch/CLT traces remain exact. For moments, supplying
        this skips eigenvalues except for FFT, which always uses its full spectrum.
        Precision solves are shared across trace orders. Probe workspace is
        blocked above 256 MiB (excluding existing kernel/cache storage, and with
        at least one probe). A global-offset pass may require replaying solves.
        Probe error decreases as the inverse square root of the count;
        no fixed count guarantees relative p-value accuracy.
    nufft_spectrum : bool, default False
        Opt into the reduced eigendecomposition for centered NUFFT moment
        calibration. By default, NUFFT uses analytic lower traces and 60 probes
        for higher traces, even when eigenvalues are cached. ``True`` requests
        the bounded PSD spectrum (at most 2000 retained Fourier modes), falling
        back to probes if unavailable. Spectral truncation and NUFFT accuracy
        limit this approximation. ``n_probes`` overrides this flag;
        ``k_eigen`` selects top-k Lanczos instead. Other backends and Welch/CLT
        ignore this flag.

    Returns
    -------
    dict
        ``method``, ``model`` and ``tail`` identify the calibration path.
        ``mean_Q`` and ``var_Q`` describe its Q moments; ``var_R`` is the
        square trace of HKH used by the separate, two-sided normal R-test,
        including when the kernel exposes raw traces with ``centering=False``.

        Welch also stores ``scale_g`` and ``df_h``. Moment matching stores ``q_fit``:
        its ``family`` names the distribution, while ``fit`` distinguishes
        ``liu4``, ``beta4``, ``chi2_skewness``, ``chi2_kurtosis``,
        ``normal_fallback`` and ``degenerate`` fits. ``source`` records
        whether the traces came from the spectrum or probes.

        Intermediate trace estimates are not returned. Recompute this cache
        when changing the kernel or calibration settings; partial caches
        are not rebuilt automatically.

    Notes
    -----
    Defaults for Gaussian, Matérn, CAR and graph-Laplacian kernels are Welch
    on MatrixKernel and moment matching on FFTKernel/NUFFTKernel. Moran and
    detected signed Fourier spectra use CLT. Precomputed matrix kernels
    default to Welch unless labeled ``method="moran"``; the caller must
    select CLT or finite-sample moments for other symmetric indefinite
    matrices. No extra eigendecomposition is done just to choose the default.

    For ``H = I - 11'/n``, ``A = HKH`` and ``m = n-1``, standardization
    gives ``Q = m X'AX / (X'HX)``. Under the iid central Gaussian null,

    .. math::

        E[Q] = \operatorname{tr}(A), \qquad
        \operatorname{Var}(Q) =
        \frac{2[m\operatorname{tr}(A^2)-\operatorname{tr}(A)^2]}{m+2}.

    Finite-sample higher moments use ``B = A - tr(A) H/m``; their formulas and
    fallback order live together in :func:`_fit_standardized_q`.
    Without the correction, variance is ``2 tr(A^2)``. These moment fits
    remain approximations, particularly in extreme tails.

    Examples
    --------
    >>> params = compute_null_params(kernel, method="moments")
    >>> Q, pval = spatial_q_test(data, kernel, null_params=params)
    """
    from sonic.kernels.base import MatrixKernelBase
    from sonic.kernels.fft import FFTKernel
    from sonic.kernels.nufft import NUFFTKernel

    # 1. Resolve policy and validate budgets before doing any kernel work.
    kernel = _centered_kernel(kernel)
    method = _resolve_q_null_method(kernel, method, dirichlet_correction=dirichlet_correction)
    n = int(kernel.n)
    if n < 2:
        raise ValueError("Spatial null calibration requires n >= 2.")
    if n_probes is not None and (
        isinstance(n_probes, (bool, np.bool_))
        or not np.isfinite(n_probes)
        or n_probes < 1
        or int(n_probes) != n_probes
    ):
        raise ValueError("n_probes must be a positive integer.")
    probe_count = int(n_probes) if n_probes is not None else (60 if method == "moments" else 15)
    precision = bool(getattr(kernel, "stores_precision", False))
    params = {
        "method": method,
        "model": "standardized_q" if dirichlet_correction else "gaussian_quadratic",
        "tail": "two-sided" if method == "clt" else "upper",
    }

    # 2. Choose the source before collecting traces. A failed spectrum attempt
    # must be cheap; sparse/implicit full-spectrum requests fail immediately.
    if method == "moments":
        try:
            if n_probes is not None:
                raise NotImplementedError("Use trace probes for this request.")
            if isinstance(kernel, NUFFTKernel) and k_eigen is None and not nufft_spectrum:
                raise NotImplementedError("NUFFT moments default to analytic traces and probes.")
            if isinstance(kernel, MatrixKernelBase) and k_eigen is None:
                cached = getattr(kernel, "_spectrum_centered", None)
                if n > _DENSE_NULL_SPECTRUM_LIMIT and (cached is None or len(cached) != n):
                    raise NotImplementedError("Large matrix nulls use trace probes.")
            if isinstance(kernel, FFTKernel):
                vals = kernel.eigenvalues(k=k_eigen, return_full_layout=True)
            else:
                vals = kernel.eigenvalues(k=k_eigen)
            if dirichlet_correction and k_eigen is None and len(vals) < n - 1:
                raise NotImplementedError("Incomplete centered spectrum; use probes.")
            # Keep the full second trace for R even when Q explicitly uses a
            # truncated spectrum. Do not use centered B powers as an R variance.
            if len(vals) == n and (precision or isinstance(kernel, FFTKernel)):
                params["var_R"] = float(np.sum(vals**2))
            elif precision:
                params["var_R"] = float(kernel.square_trace(n_probes=probe_count))
            else:
                # In particular, NUFFT's reduced spectrum may be truncated;
                # keep its separate analytic R variance authoritative.
                params["var_R"] = float(kernel.square_trace())
            if dirichlet_correction:
                if len(vals) == n:
                    vals = np.delete(vals, np.argmin(np.abs(vals)))
                traces = _spectrum_traces(vals, n)
            else:
                scale = float(np.max(np.abs(vals), initial=0.0))
                scaled = vals / scale if scale else vals
                traces = {
                    "mean": float(np.sum(scaled) * scale),
                    "powers": {p: float(np.sum(scaled**p)) for p in (2, 3, 4)},
                    "scale": scale,
                    "source": "spectrum",
                }
        except NotImplementedError:
            traces = _estimate_kernel_traces(
                kernel, n_probes=probe_count, centered=dirichlet_correction
            )
            params["var_R"] = traces["var_R"]
        # 3. Fit once; every feature consumes this small prepared cache.
        params["source"] = traces["source"]
        if dirichlet_correction:
            params["q_fit"] = _fit_standardized_q(
                traces["mean"], traces["powers"], traces["scale"], n
            )
        else:
            params["q_fit"] = _fit_gaussian_quadratic(
                traces["mean"], traces["powers"], scale=traces["scale"]
            )
        params["mean_Q"] = params["q_fit"]["mu_Q"]
        params["var_Q"] = params["q_fit"]["sigma_Q"] ** 2
    else:
        # Welch/CLT need only two traces. Explicit kernels calculate these
        # exactly; precision kernels share one (possibly blocked) probe solve.
        if precision:
            traces = _estimate_kernel_traces(
                kernel, n_probes=probe_count, centered=dirichlet_correction, max_order=2
            )
            mean_Q, second = traces["mean"], traces["powers"][2]
            params["var_R"] = traces["var_R"]
        else:
            second = float(kernel.square_trace())
            mean_Q = float(kernel.trace())
            params["var_R"] = second
        m = n - 1
        var_Q = 2.0 * (m * second - mean_Q**2) / (m + 2) if dirichlet_correction else 2.0 * second
        var_Q = max(var_Q, 0.0)
        params["mean_Q"] = float(mean_Q)
        params["var_Q"] = float(var_Q)
        if method == "welch":
            if var_Q > 0 and mean_Q > 0:
                params["scale_g"] = var_Q / (2.0 * mean_Q)
                params["df_h"] = (2.0 * mean_Q**2) / var_Q
            else:
                params["scale_g"] = 0.0
                params["df_h"] = 1.0
    return params


def _sparse_mean_std(X: sp.spmatrix, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Stable column moments without densification or input mutation."""
    X = X.astype(np.float64, copy=False)
    if X.format not in ("csr", "csc"):
        X = X.tocsc()
    if not X.has_canonical_format:
        X = X.copy()
        X.sum_duplicates()
    means, var = mean_variance_axis(X, axis=0)
    var *= X.shape[0] / max(X.shape[0] - ddof, 1)
    return means, np.sqrt(var)


def _q_test_matrix(  # noqa: C901
    Xn: np.ndarray | sp.spmatrix,
    kernel: Kernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """Single-batch Q-test on a MatrixKernel (no chunking).

    Parallel to :func:`sonic.kernels.fft._q_test_fft` /
    :func:`sonic.kernels.nufft._q_test_nufft`: takes whatever batch size is
    handed in and processes it in one call. The chunking loop lives in
    :func:`spatial_q_test`, which dispatches here per chunk.
    """
    is_sparse = sp.issparse(Xn)
    if is_sparse:
        # Promote before squaring integer counts, while preserving sparsity.
        Xn = Xn.astype(np.float64, copy=False)
        n, M = Xn.shape if Xn.ndim == 2 else (Xn.shape[0], 1)
        if Xn.ndim == 1 or M == 1:
            Xn = Xn.reshape(-1, 1)
            M = 1
    else:
        Xn = np.asarray(Xn, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        n, M = Xn.shape

    # Sparse moments stay sparse; the kernel selects a sparse product or a
    # directly centered dense workspace according to numerical safety and kernel type.
    if is_sparse and not is_standardized and hasattr(kernel, "xtKx_standardized"):
        means, stds = _sparse_mean_std(Xn)
        valid_mask = stds > 1e-12
        Q = kernel.xtKx_standardized(Xn, means, stds)
    else:
        if is_standardized:
            z = Xn
            valid_mask = (
                np.asarray((z != 0).sum(axis=0)).ravel() > 0 if is_sparse else np.any(z, axis=0)
            )
        else:
            if is_sparse:
                Xn = Xn.toarray()
            means = np.mean(Xn, axis=0)
            stds = np.std(Xn, axis=0, ddof=1)
            valid_mask = stds > 1e-12
            z = np.zeros_like(Xn)
            if np.any(valid_mask):
                z[:, valid_mask] = (Xn[:, valid_mask] - means[valid_mask]) / stds[valid_mask]
        if hasattr(kernel, "xtKx"):
            Q = kernel.xtKx(z)
        else:
            # Fallback for raw matrices.
            Kz = kernel.dot(z) if sp.issparse(kernel) else np.dot(kernel, z)
            Q = np.sum(z * Kz, axis=0)

    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    if M == 1 and Q.size == 1:
        Q = Q.item()

    if not return_pval:
        return Q

    # Calibration is shared with both Fourier backends; this helper only computes Q.
    if null_params is None:
        null_params = _prepare_q_null(kernel)
    pval = _q_pvalues(np.atleast_1d(Q), null_params)
    # A constant feature has no sample-standardized null realization. In
    # particular, Q=0 can lie in a significant tail when the kernel is signed.
    pval = np.where(valid_mask, pval, 1.0)

    pval = np.atleast_1d(pval)
    if M == 1 and pval.size == 1:
        pval = pval.item()
    return Q, pval


def _chunk_last_axis(X, start: int, end: int):
    """Slice ``X`` along its trailing (feature) axis. Works on numpy
    arrays and ``scipy.sparse`` matrices alike."""
    if sp.issparse(X) or X.ndim == 2:
        return X[:, start:end]
    return X[:, :, start:end]


def _feature_count(X, is_fft: bool) -> int:
    """Return the number of features ``M`` in the trailing axis of ``X``.

    FFTKernel input shape is ``(ny, nx)`` / ``(ny, nx, M)``; everything
    else is ``(n,)`` / ``(n, M)``.
    """
    if is_fft:
        return X.shape[2] if X.ndim == 3 else 1
    if sp.issparse(X):
        return X.shape[1] if X.ndim == 2 else 1
    return X.shape[1] if X.ndim == 2 else 1


def _resolve_chunk_size(
    chunk_size: int | str,
    kernel: Kernel,
    M: int,
    n_jobs: int = 1,
    memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
) -> int:
    """Turn ``chunk_size='auto' | -1 | int`` into a concrete batch size.

    ``'auto'`` → :func:`auto_chunk_size`. ``-1`` or ``>= M`` → ``M``
    (no chunking). Everything else is coerced to a positive int.
    """
    if isinstance(chunk_size, str):
        if chunk_size != "auto":
            raise ValueError(f"chunk_size must be 'auto', -1, or int, got {chunk_size!r}.")
        resolved = auto_chunk_size(kernel, n_jobs=n_jobs, budget_bytes=memory_budget_bytes)
    elif int(chunk_size) == -1:
        resolved = M
    else:
        resolved = max(1, int(chunk_size))
    return max(1, min(resolved, M))


def spatial_q_test(  # noqa: C901
    Xn: np.ndarray | sp.spmatrix,
    kernel: Kernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
    chunk_size: int | str = "auto",
    show_progress: bool = False,
    *,
    memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """
    Univariate spatial Q-test for detecting spatial variability.

    Top-level chunking wrapper — splits the feature batch along the
    trailing axis into blocks of ``chunk_size`` features, dispatches
    each block to the backend-specific per-chunk helper
    (:func:`sonic.kernels.fft._q_test_fft`,
    :func:`sonic.kernels.nufft._q_test_nufft`, or :func:`_q_test_matrix`), and
    concatenates the results. The per-chunk helpers do **not** handle
    chunking themselves.

    Parameters
    ----------
    Xn : np.ndarray or scipy.sparse matrix
        Input data of shape ``(n,)`` / ``(n, M)`` for MatrixKernel and
        NUFFTKernel, or ``(ny, nx)`` / ``(ny, nx, M)`` for FFTKernel.
        Can be dense numpy array or sparse matrix (CSC/CSR recommended)
        for MatrixKernel; FFT/NUFFT paths require dense input.
    kernel : Kernel
        Pre-constructed :class:`~sonic.kernels.Kernel` (``MatrixKernel`` /
        ``FFTKernel`` / ``NUFFTKernel``) or a raw dense / sparse kernel
        matrix.
    null_params : dict, optional
        Pre-computed null distribution parameters from
        :func:`compute_null_params`. Resolved once at the top level if
        ``None`` and shared across chunks (no redundant recomputation).
    return_pval : bool, default True
        If True, returns ``(Q, pval)``; else returns ``Q`` only.
    is_standardized : bool, default False
        If True, skips Z-score standardization internally.
    chunk_size : int or ``'auto'``, default ``'auto'``
        Number of features processed per per-chunk dispatch call.
        ``'auto'`` defers to :func:`auto_chunk_size` (backend-specific
        cache cap under ``memory_budget_bytes``); ``-1``
        processes the full batch in a single call. For a cross-backend
        cost model see :doc:`/guides/scaling`.
    show_progress : bool, default False
        If True, displays a tqdm bar over chunks (only when ``M > chunk_size``).
    memory_budget_bytes : int or str, default 2147483648 (2 GiB)
        Bytes or a size string, e.g. ``"512 MB"`` or ``"2 GiB"``. Units are
        case-insensitive: GB uses powers of 1000, GiB powers of 1024.
        Positive estimated batch-workspace budget for ``chunk_size='auto'``.
        Excludes inputs, retained outputs, kernel storage and null calibration;
        not a total-process memory limit. Explicit chunk sizes bypass sizing.

    Returns
    -------
    Q : float or np.ndarray
        Test statistic value(s). Shape ``(M,)`` for 2-D / 3-D inputs,
        scalar for 1-D.
    pval : float or np.ndarray, optional
        Null p-value, returned only if ``return_pval=True``. Welch and moments
        use the upper tail; CLT uses a two-sided normal tail. Same shape as Q.

    Notes
    -----
    Inputs are centered and scaled by their sample standard deviation.
    Under the iid Gaussian null, Q is a ratio of quadratic forms; the chosen
    null fit approximates its tail. PSD matrix kernels default to Welch,
    PSD Fourier kernels to moment matching, and recognized signed kernels to CLT.
    See :func:`compute_null_params` and :doc:`/guides/theory`.

    Examples
    --------
    >>> coords = np.random.randn(100, 2)
    >>> kernel = MatrixKernel.from_coordinates(coords, method='gaussian')
    >>> data = np.random.randn(100)
    >>> Q, pval = spatial_q_test(data, kernel)
    >>> # Sparse-matrix batch of features (auto-chunked):
    >>> from scipy.sparse import csr_matrix
    >>> sparse_data = csr_matrix(np.random.randn(100, 1000))
    >>> Q, pval = spatial_q_test(sparse_data, kernel, show_progress=True)
    """
    # Lazy imports — avoid circular dependency with the FFT / NUFFT modules.
    from sonic.kernels.fft import FFTKernel, _q_test_fft
    from sonic.kernels.nufft import NUFFTKernel, _q_test_nufft

    memory_budget_bytes = _parse_memory_budget(memory_budget_bytes)
    is_fft = isinstance(kernel, FFTKernel)
    is_nufft = isinstance(kernel, NUFFTKernel)
    # Prepare the null once for every backend before splitting features.
    # Score-only calls never estimate traces or fit a distribution.
    if return_pval:
        null_params = _prepare_q_null(kernel, null_params)

    # Determine M on the trailing axis.
    M = _feature_count(Xn, is_fft=is_fft)
    resolved_chunk = _resolve_chunk_size(
        chunk_size, kernel, M, memory_budget_bytes=memory_budget_bytes
    )

    if is_fft:

        def _dispatch(X):
            return _q_test_fft(
                X,
                kernel,
                null_params=null_params,
                return_pval=return_pval,
                is_standardized=is_standardized,
            )

    elif is_nufft:

        def _dispatch(X):
            return _q_test_nufft(
                X,
                kernel,
                null_params=null_params,
                return_pval=return_pval,
                is_standardized=is_standardized,
            )

    else:

        def _dispatch(X):
            return _q_test_matrix(
                X,
                kernel,
                null_params=null_params,
                return_pval=return_pval,
                is_standardized=is_standardized,
            )

    # Single-batch shortcut.
    if resolved_chunk >= M:
        return _dispatch(Xn)

    # Chunk loop.
    starts = list(range(0, M, resolved_chunk))
    iterator = starts
    if show_progress and len(starts) > 1:
        iterator = tqdm(
            starts,
            desc="Q-test chunks",
            total=len(starts),
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

    Q_parts: list[np.ndarray] = []
    P_parts: list[np.ndarray] = []
    for start in iterator:
        end = min(start + resolved_chunk, M)
        block = _chunk_last_axis(Xn, start, end)
        result = _dispatch(block)
        if return_pval:
            Q_b, P_b = result
            Q_parts.append(np.atleast_1d(Q_b))
            P_parts.append(np.atleast_1d(P_b))
        else:
            Q_parts.append(np.atleast_1d(result))

    Q = np.concatenate(Q_parts)
    if return_pval:
        return Q, np.concatenate(P_parts)
    return Q


def _r_null_variance(kernel: Kernel) -> float:
    """Variance of standardized X'KY, independent of the kernel's trace view."""
    return max(float(_centered_kernel(kernel).square_trace()), 0.0)


def _r_test_matrix(  # noqa: C901
    Xn: np.ndarray | sp.spmatrix,
    Yn: np.ndarray | sp.spmatrix,
    kernel: Kernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """Single-batch R-test on a MatrixKernel (no chunking).

    Parallel to :func:`sonic.kernels.fft._r_test_fft` /
    :func:`sonic.kernels.nufft._r_test_nufft`: takes whatever batch size is
    handed in and processes it in one call. The chunking loop lives in
    :func:`spatial_r_test`, which dispatches here per chunk.
    """

    # Normalize shapes; preserve sparsity of inputs.
    def _prep(A):
        if sp.issparse(A):
            return A.reshape(-1, 1) if A.ndim == 1 else A
        arr = np.asarray(A, dtype=float)
        return arr.reshape(-1, 1) if arr.ndim == 1 else arr

    Xn, Yn = _prep(Xn), _prep(Yn)
    if Xn.shape != Yn.shape:
        raise ValueError(f"Xn and Yn shapes must match, got {Xn.shape} vs {Yn.shape}.")
    n, M = Xn.shape
    if n != kernel.n:
        raise ValueError(f"Kernel.n={kernel.n} does not match data rows {n}.")

    def _standardize(A):
        """Z-score A (sparse or dense) column-wise with ddof=1. Returns dense."""
        if sp.issparse(A):
            means, stds = _sparse_mean_std(A)
            Z = A.toarray() - means
        else:
            means = np.mean(A, axis=0)
            stds = np.std(A, axis=0, ddof=1)
            Z = A - means
        valid = stds > 1e-12
        Z[:, ~valid] = 0.0
        if np.any(valid):
            Z[:, valid] /= stds[valid]
        return Z

    if is_standardized:
        Zx = np.asarray(Xn.toarray() if sp.issparse(Xn) else Xn, dtype=float)
        Zy = np.asarray(Yn.toarray() if sp.issparse(Yn) else Yn, dtype=float)
    else:
        Zx = _standardize(Xn)
        Zy = _standardize(Yn)

    # R = diag(Zx^T K Zy) via the kernel's public bilinear primitive.
    R = np.atleast_1d(np.asarray(kernel.xtKy(Zx, Zy)))

    if M == 1 and R.size == 1:
        R = R.item()

    if not return_pval:
        return R

    # P-value (Normal Approximation). Both X, Y are z-scored before
    # R = Zₓᵀ K Zᵧ, so R ~ N(0, trace((HKH)²)) — NOT trace(K²).
    if null_params is not None and "var_R" in null_params:
        var_R = float(null_params["var_R"])
    else:
        var_R = _r_null_variance(kernel)
    sigma = np.sqrt(var_R)
    if sigma > 0:
        z_score = R / sigma
        pval = 2 * norm.sf(np.abs(z_score))
    else:
        pval = np.ones_like(R) if isinstance(R, np.ndarray) else 1.0
    return R, pval


def spatial_r_test(  # noqa: C901
    Xn: np.ndarray | sp.spmatrix,
    Yn: np.ndarray | sp.spmatrix,
    kernel: Kernel,
    null_params: dict | None = None,
    return_pval: bool = True,
    is_standardized: bool = False,
    chunk_size: int | str = "auto",
    show_progress: bool = False,
    *,
    memory_budget_bytes: int | str = _DEFAULT_CHUNK_BUDGET,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """
    Bivariate spatial R-test for correlation between two spatial variables.

    Top-level chunking wrapper — splits the paired feature batch along
    the trailing axis into blocks of ``chunk_size`` features, dispatches
    each block to the backend-specific per-chunk helper
    (:func:`sonic.kernels.fft._r_test_fft`,
    :func:`sonic.kernels.nufft._r_test_nufft`, or :func:`_r_test_matrix`), and
    concatenates the results. The per-chunk helpers do **not** handle
    chunking themselves.

    Parameters
    ----------
    Xn : np.ndarray or scipy.sparse matrix
        First input. Shape ``(n,)`` / ``(n, M)`` for MatrixKernel and
        NUFFTKernel, ``(ny, nx)`` / ``(ny, nx, M)`` for FFTKernel.
    Yn : np.ndarray or scipy.sparse matrix
        Second input, same shape as ``Xn`` (paired R-test).
        For ``NUFFTKernel`` a bipartite mode with ``M_x != M_y`` is
        passed through without chunking.
    kernel : Kernel
        Pre-constructed :class:`~sonic.kernels.Kernel`.
    null_params : dict, optional
        Pre-computed null parameters; only ``'var_R'`` is consumed.
        Resolved once at the top level if ``None`` and shared across
        chunks.
    return_pval : bool, default True
        If True, returns ``(R, pval)``; else returns ``R`` only.
    is_standardized : bool, default False
        If True, skips Z-score standardization internally.
    chunk_size : int or ``'auto'``, default ``'auto'``
        Number of feature pairs processed per per-chunk dispatch call.
        ``'auto'`` defers to :func:`auto_chunk_size`; ``-1`` processes
        the full batch in a single call. For a cross-backend cost model
        see :doc:`/guides/scaling`.
    show_progress : bool, default False
        If True, displays a tqdm bar over chunks (only when
        ``M > chunk_size``).
    memory_budget_bytes : int or str, default 2147483648 (2 GiB)
        Bytes or a size string, e.g. ``"512 MB"`` or ``"2 GiB"``. Units are
        case-insensitive: GB uses powers of 1000, GiB powers of 1024.
        Positive estimated batch-workspace budget for ``chunk_size='auto'``.
        Excludes inputs, retained outputs, kernel storage and null calibration;
        not a total-process memory limit. Explicit chunk sizes and bipartite
        NUFFT inputs (``M_x != M_y``) bypass sizing.

    Returns
    -------
    R : float or np.ndarray
        Test statistic value(s). Shape ``(M,)`` for 2-D / 3-D inputs,
        scalar for 1-D. For bipartite NUFFT input
        (``M_x != M_y``), shape ``(M_x, M_y)``.
    pval : float or np.ndarray, optional
        Two-tailed tail probability under null hypothesis; returned
        only if ``return_pval=True``.

    Notes
    -----
    Under H₀, ``R = xᵀ K y`` is approximated as
    :math:`\\mathcal{N}(0, \\mathrm{trace}((HKH)^2))` on z-scored inputs.
    See :doc:`/guides/theory` and :doc:`/guides/scaling`.

    Examples
    --------
    >>> coords = np.random.randn(100, 2)
    >>> kernel = MatrixKernel.from_coordinates(coords, method='gaussian')
    >>> x_data = np.random.randn(100)
    >>> y_data = np.random.randn(100)
    >>> R, pval = spatial_r_test(x_data, y_data, kernel)
    """
    # Lazy imports — avoid circular dependency with the FFT / NUFFT modules.
    from sonic.kernels.fft import FFTKernel, _r_test_fft
    from sonic.kernels.nufft import NUFFTKernel, _r_test_nufft

    memory_budget_bytes = _parse_memory_budget(memory_budget_bytes)
    is_fft = isinstance(kernel, FFTKernel)
    is_nufft = isinstance(kernel, NUFFTKernel)

    # Resolve var_R once (cached across chunks).
    if return_pval and (null_params is None or "var_R" not in null_params):
        null_params = {**(null_params or {}), "var_R": _r_null_variance(kernel)}

    Mx = _feature_count(Xn, is_fft=is_fft)
    My = _feature_count(Yn, is_fft=is_fft)

    if is_fft:

        def _dispatch(X, Y):
            return _r_test_fft(
                X,
                Y,
                kernel,
                null_params=null_params,
                return_pval=return_pval,
                is_standardized=is_standardized,
            )

    elif is_nufft:

        def _dispatch(X, Y):
            return _r_test_nufft(
                X,
                Y,
                kernel,
                null_params=null_params,
                return_pval=return_pval,
                is_standardized=is_standardized,
            )

    else:

        def _dispatch(X, Y):
            return _r_test_matrix(
                X,
                Y,
                kernel,
                null_params=null_params,
                return_pval=return_pval,
                is_standardized=is_standardized,
            )

    # Bipartite NUFFT (M_x != M_y) doesn't fit the paired-chunk pattern —
    # pass the full batch through and let the backend handle it.
    if Mx != My:
        return _dispatch(Xn, Yn)

    M = Mx
    resolved_chunk = _resolve_chunk_size(
        chunk_size, kernel, M, memory_budget_bytes=memory_budget_bytes
    )

    # Single-batch shortcut.
    if resolved_chunk >= M:
        return _dispatch(Xn, Yn)

    # Chunk loop.
    starts = list(range(0, M, resolved_chunk))
    iterator = starts
    if show_progress and len(starts) > 1:
        iterator = tqdm(
            starts,
            desc="R-test chunks",
            total=len(starts),
            bar_format="{l_bar}{bar:30}{r_bar}{bar:-30b}",
        )

    R_parts: list[np.ndarray] = []
    P_parts: list[np.ndarray] = []
    for start in iterator:
        end = min(start + resolved_chunk, M)
        Xblock = _chunk_last_axis(Xn, start, end)
        Yblock = _chunk_last_axis(Yn, start, end)
        result = _dispatch(Xblock, Yblock)
        if return_pval:
            R_b, P_b = result
            R_parts.append(np.atleast_1d(R_b))
            P_parts.append(np.atleast_1d(P_b))
        else:
            R_parts.append(np.atleast_1d(result))

    R = np.concatenate(R_parts)
    if return_pval:
        return R, np.concatenate(P_parts)
    return R
