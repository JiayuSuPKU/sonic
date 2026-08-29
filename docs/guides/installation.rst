Installation
============

Release candidate
-----------------

.. code-block:: bash

   pip install sonic-spatial==1.0.0rc1

The Python import remains ``sonic``.


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

   pip install quadsv==1.0.0rc1


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
