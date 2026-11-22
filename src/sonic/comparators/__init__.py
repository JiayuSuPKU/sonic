"""Compare gene expression and spatial spectra across biological samples.

Use :class:`ComparatorIrregular` for irregular coordinates in AnnData.
Use :class:`ComparatorGrid` for regular grids in SpatialData. Both support
single-gene and gene-pair comparisons. Fit spectra once and supply a design
for each test. See :doc:`/guides/multisample` for examples.
"""

from sonic.comparators.grid import ComparatorGrid
from sonic.comparators.irregular import ComparatorIrregular

__all__ = ["ComparatorIrregular", "ComparatorGrid"]
