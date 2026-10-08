Installation
============

Release candidate
-----------------

.. code-block:: bash

   pip install sonic-spatial==1.0.0rc2

The Python import remains ``sonic``.

Install the current GitHub source with:

.. code-block:: bash

   pip install 'sonic-spatial @ git+https://github.com/JiayuSuPKU/sonic.git'


Development install
-------------------

.. code-block:: bash

   git clone https://github.com/JiayuSuPKU/sonic.git
   cd sonic
   pip install -e '.[dev,docs]'

The ``dev`` and ``docs`` extras include the test and documentation toolchains,
respectively.


Legacy ``quadsv`` compatibility
-------------------------------

The compatibility distribution installs SONIC while preserving deprecated
``quadsv`` imports:

.. code-block:: bash

   pip install quadsv==1.0.0rc2


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
   from sonic import Comparator, Detector

   print(sonic.__version__)

If these imports succeed, continue with :doc:`/guides/quickstart`.


Development troubleshooting
---------------------------

If Numba reports ``no locator available`` while importing a dependency, point
it to a writable cache directory:

.. code-block:: bash

   mkdir -p /private/tmp/sonic-numba-cache
   NUMBA_CACHE_DIR=/private/tmp/sonic-numba-cache python -c 'import sonic'

On macOS, a test process that crashes inside FINUFFT may have loaded more than
one OpenMP runtime. First retry with one OpenMP thread:

.. code-block:: bash

   OMP_NUM_THREADS=1 python -m pytest -q

A clean conda-forge environment is the quickest way to confirm whether either
problem comes from the current environment.
