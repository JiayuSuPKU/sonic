Installation
============

From PyPI
---------

.. code-block:: bash

   pip install sonic

Two optional extras are available for development and documentation:

.. code-block:: bash

   pip install 'sonic[dev]'    # tests, linting, jupyter, matplotlib
   pip install 'sonic[docs]'   # Sphinx + theme + autoapi


From source
-----------

.. code-block:: bash

   git clone https://github.com/JiayuSuPKU/sonic.git
   cd sonic
   pip install -e '.[dev,docs]'


Requirements
------------

- **Python** 3.10+.
- **Runtime dependencies** (installed automatically): ``scanpy``,
  ``spatialdata``, ``finufft``, ``joblib``, ``tqdm``. Through
  ``scanpy`` you also get ``anndata``, ``numpy``, ``scipy``,
  ``scikit-learn`` and ``pandas``.

``spatialdata`` is needed by :class:`~sonic.DetectorGrid` and
:class:`~sonic.ComparatorGrid`. ``finufft`` is needed by
:class:`~sonic.NUFFTKernel`, by :class:`~sonic.DetectorIrregular`
when you set ``backend="nufft"``, and by
:class:`~sonic.ComparatorIrregular`.


Verify the install
------------------

.. code-block:: python

   import sonic

   print(sonic.__version__)
   print(sorted(sonic.__all__))

You should see 17 public names organised into four layers (see
:doc:`/guides/quickstart` for what each layer does). The top-level
package is the user-facing surface. The canonical submodule paths
(``sonic.kernels.*``, ``sonic.detectors.*``,
``sonic.comparators.multisample``, ``sonic.statistics``) are
documented under :doc:`/autoapi/sonic/index`.
