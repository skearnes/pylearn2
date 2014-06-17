"""
Grid search helper functions and classes.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__credits__ = ["Steven Kearnes", "Bharath Ramsundar"]
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
try:
    from sklearn.grid_search import ParameterSampler
except ImportError:
    ParameterSampler = None
import sys


class UniqueParameterSampler(object):
    """
    This class is a wrapper for ParameterSampler that attempts to return
    a unique grid point on each iteration. ParameterSampler will sometimes
    yield the same parameters multiple times, especially when sampling
    small grids.

    Parameters
    ----------
    param_distribution : dict
        Parameter grid or distributions.
    n_iter : int
        Number of points to sample from the grid.
    n_attempts : int, optional
        Maximum number of samples to take from the grid.
    random_state : int or RandomState, optional
        Random state.
    """
    def __init__(self, param_distribution, n_iter, n_attempts=None,
                 random_state=None):
        if n_attempts is None:
            n_attempts = 100 * n_iter
        if ParameterSampler is None:
            raise RuntimeError("Could not import from sklearn.")
        self.sampler = ParameterSampler(param_distribution, n_attempts,
                                        random_state)
        self.n_iter = n_iter
        self.params = []

    def __iter__(self):
        """
        Return the next grid point from ParameterSampler unless we have
        seen it before. The ParameterSampler will raise StopIteration after
        n_attempts samples.
        """
        for params in self.sampler:
            if len(self.params) >= self.n_iter:
                break
            if params not in self.params:
                self.params.append(params)
                yield params

    def __len__(self):
        return self.n_iter


def random_seeds(size, random_state=None):
    """
    Generate random seeds. This function is intended for use in a pylearn2
    YAML config file.

    Parameters
    ----------
    size : int
        Number of seeds to generate.
    random_state : int or None
        Seed for random number generator.
    """
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(0, sys.maxint, size)
    return seeds
