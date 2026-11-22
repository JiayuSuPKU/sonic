#ifndef CHI2COMB_CHI2COMB_H
#define CHI2COMB_CHI2COMB_H

#define CHI2COMB_VERSION "0.0.3"
#define CHI2COMB_VERSION_MAJOR 0
#define CHI2COMB_VERSION_MINOR 0
#define CHI2COMB_VERSION_PATCH 3

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#ifdef CHI2COMB_API_EXPORTS
#define CHI2COMB_API __declspec(dllexport)
#else
#define CHI2COMB_API __declspec(dllimport)
#endif
#else
#define CHI2COMB_API
#endif

struct chi2comb_chisquareds {
    const double *coefs;  /* chi-squared coefficients */
    const double *ncents; /* noncentrality parameters */
    const int *dofs;      /* degree of freedoms */
    int n;                /* number of terms */
};

struct chi2comb_info {
    double emag; /* error magnitude */
    int niterms; /* total number of integration terms */
    int nints;   /* number of integrations */
    double intv; /* integration interval in final integration */
    double truc; /* truncation point in initial integration */
    double sd;   /* s.d. of initial convergence factor */
    int ncycles; /* cycles to locate integration parameters */
};

/** Cumulative distribution function of non-central chi-squared distributions.
 *
 * Computes P[Q<q] where Q=λ₁X₁ + ... + λₙXₙ + σ X₀. Xᵢ is an independent random variable
 * having a non-central chi-squared distribution with nᵢ degrees of freedom and
 * non-centrality parameter δ²ⱼ. X₀ is an independent random variable having a standard
 * Gaussian distribution.
 *
 * Parameters
 * ----------
 * q : double
 *     Value point at which distribution function is to be evaluated.
 * chi2s : struct chi2comb_chisquareds
 *     Chi-squared distributions.
 * gcoef : double
 *     Coefficient of the standard Normal distribution.
 * lim : int
 *     Maximum number of integration terms.
 * abstol : double
 *     Absolute error tolerance.
 * info : struct chi2comb_info*
 *     Algorithm information.
 * result : double *
 *     Estimated c.d.f. evaluated at `q`.
 *
 * Returns
 * -------
 * error : int
 *     0: completed successfully
 *     1: required accuracy not achieved
 *     2: round-off error possibly significant
 *     3: invalid parameters
 *     4: unable to locate integration parameters
 *     5: out of memory
 */
CHI2COMB_API int chi2comb_cdf(double q, struct chi2comb_chisquareds *chi2s, double gcoef,
                              int lim, double abstol, struct chi2comb_info *info,
                              double *result);

#ifdef __cplusplus
}
#endif

#endif /* end of include guard: CHI2COMB_CHI2COMB_H */