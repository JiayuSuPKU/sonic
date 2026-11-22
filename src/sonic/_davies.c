/* Python binding for the MIT-licensed chi2comb Davies core. See licenses/chi2comb.txt. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>
#include <math.h>
#define CHI2COMB_API_EXPORTS
#include "_vendor/chi2comb.h"

static PyObject *davies_sf(PyObject *self, PyObject *args)
{
    PyObject *input, *sequence = NULL, *output = NULL;
    double accuracy, result;
    int limit, fault;
    double *coefficients = NULL, *noncentrality = NULL;
    int *dofs = NULL;
    Py_ssize_t size, i;
    struct chi2comb_chisquareds terms;
    struct chi2comb_info info;

    if (!PyArg_ParseTuple(args, "Odi:sf", &input, &accuracy, &limit))
        return NULL;
    if (!isfinite(accuracy) || accuracy <= 0 || accuracy >= 1 || limit < 1) {
        PyErr_SetString(PyExc_ValueError, "Davies requires 0 < accuracy < 1 and limit >= 1.");
        return NULL;
    }
    sequence = PySequence_Fast(input, "Davies coefficients must be a finite sequence.");
    if (!sequence)
        return NULL;
    size = PySequence_Fast_GET_SIZE(sequence);
    if (size < 1 || size > INT_MAX) {
        PyErr_SetString(PyExc_ValueError, "Davies requires 1 through INT_MAX coefficients.");
        goto cleanup;
    }
    coefficients = PyMem_Calloc((size_t)size, sizeof(double));
    noncentrality = PyMem_Calloc((size_t)size, sizeof(double));
    dofs = PyMem_Calloc((size_t)size, sizeof(int));
    if (!coefficients || !noncentrality || !dofs) {
        PyErr_NoMemory();
        goto cleanup;
    }
    for (i = 0; i < size; ++i) {
        coefficients[i] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sequence, i));
        if (PyErr_Occurred())
            goto cleanup;
        if (!isfinite(coefficients[i])) {
            PyErr_SetString(PyExc_ValueError, "Davies coefficients must be finite.");
            goto cleanup;
        }
        dofs[i] = 1;
    }
    terms.coefs = coefficients;
    terms.ncents = noncentrality;
    terms.dofs = dofs;
    terms.n = (int)size;
    /* Keep the GIL: the upstream routine's static state is not reentrant. */
    fault = chi2comb_cdf(0.0, &terms, 0.0, limit, accuracy, &info, &result);
    if (fault == 5) {
        PyErr_NoMemory();
        goto cleanup;
    }
    output = Py_BuildValue("di", 1.0 - result, fault);
cleanup:
    PyMem_Free(coefficients);
    PyMem_Free(noncentrality);
    PyMem_Free(dofs);
    Py_DECREF(sequence);
    return output;
}

static PyMethodDef methods[] = {
    {"sf", davies_sf, METH_VARARGS,
     "sf(coefficients, accuracy, limit) -> (P(sum(a*Z**2) >= 0), ifault)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_davies", "Native Davies AS 155 integration.", -1, methods
};

PyMODINIT_FUNC PyInit__davies(void)
{
    return PyModule_Create(&module);
}
