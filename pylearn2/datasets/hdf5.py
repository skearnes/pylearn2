"""Objects for datasets serialized in HDF5 format (.h5)."""
import warnings
try:
    import h5py
except ImportError:
    warnings.warn("Could not import h5py")
import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils.iteration import FiniteDatasetIterator, safe_izip


class HDF5Dataset(DenseDesignMatrix):
    """Dense dataset loaded from an HDF5 file."""
    def __init__(self, filename, X=None, topo_view=None, y=None, **kwargs):
        """
        Loads data and labels from HDF5 file.

        Parameters
        ----------
        filename: str
            HDF5 file name.
        X: str
            Key into HDF5 file for dataset design matrix.
        topo_view: str
            Key into HDF5 file for topological view of dataset.
        y: str
            Key into HDF5 file for dataset targets.
        kwargs: dict
            Keyword arguments passed to `DenseDesignMatrix`.
        """
        self._file = h5py.File(filename)
        if X is not None:
            X = self._get_dataset(X)
        if topo_view is not None:
            topo_view = self._get_dataset(topo_view)
        if y is not None:
            y = self._get_dataset(y)
        super(HDF5Dataset, self).__init__(X=X, topo_view=topo_view, y=y,
                                          **kwargs)

    def _get_dataset(self, dataset, load_all=False):
        """
        Get a handle for an HDF5 dataset.

        Parameters
        ----------
        dataset : str
            Name or path of HDF5 dataset.
        """
        if load_all:
            data = self._file[dataset][:]
        else:
            data = self._file[dataset]
            data.ndim = len(data.shape)
        return data

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        iterator = super(HDF5Dataset, self).iterator(mode, batch_size,
                                                     num_batches, topo, targets,
                                                     rng, data_specs,
                                                     return_tuple)
        iterator.__class__ = HDF5DatasetIterator
        return iterator


class HDF5DatasetIterator(FiniteDatasetIterator):
    def next(self):
        next_index = self._subset_iterator.next()

        # convert to boolean selection
        sel = np.zeros(self.num_examples, dtype=bool)
        sel[next_index] = True
        next_index = sel

        rval = []
        for data, fn in safe_izip(self._raw_data, self._convert):
            if data.ndim > 1:
                this_data = data[next_index, :]
            else:
                this_data = data[next_index]
            if fn:
                this_data = fn(this_data)
            rval.append(this_data)
        rval = tuple(rval)
        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval
