Comparator methods
==================

:class:`~sonic.comparators.irregular.ComparatorIrregular` and
:class:`~sonic.comparators.grid.ComparatorGrid` provide the same methods.
See :doc:`guides/multisample` to choose a test.

.. currentmodule:: sonic.comparators.base

Compare samples
---------------

.. automethod:: _ComparatorBase.test_diff_freq
.. automethod:: _ComparatorBase.test_cross_diff_freq
.. automethod:: _ComparatorBase.test_diff_expr

Prepare spectra
---------------

.. automethod:: _ComparatorBase.compute_spectra
.. automethod:: _ComparatorBase.compute_cross_spectra
.. automethod:: _ComparatorBase.subset
.. automethod:: _ComparatorBase.normalize_background
.. automethod:: _ComparatorBase.normalize_covariates

Inspect results
---------------

.. automethod:: _ComparatorBase.estimate_null_covariance
.. automethod:: _ComparatorBase.effective_rank
.. automethod:: _ComparatorBase.explain_spectrum
