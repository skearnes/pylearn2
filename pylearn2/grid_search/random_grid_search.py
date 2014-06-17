"""
Random grid search.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__credits__ = ["Steven Kearnes", "Bharath Ramsundar"]
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

try:
    from sklearn.grid_search import ParameterSampler
except ImportError:
    ParameterSampler = None

from pylearn2.grid_search import GridSearch, GridSearchCV


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


class RandomGridSearch(GridSearch):
    """
    Hyperparameter grid search using a YAML template and random selection
    of a subset of the grid points.

    Parameters
    ----------
    n_iter : int
        Number of grid points to sample.
    random_state : int, optional
        Random seed.
    kwargs : dict, optional
        Keyword arguments for GridSearch.
    """
    def __init__(self, n_iter, random_state=None, **kwargs):
        self.n_iter = n_iter
        self.random_state = random_state
        super(RandomGridSearch, self).__init__(**kwargs)

    def get_param_grid(self, param_grid):
        """
        Construct a parameter grid.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        return UniqueParameterSampler(param_grid, self.n_iter, None,
                                      self.random_state)


class RandomGridSearchCV(GridSearchCV):
    """
    GridSearchCV with random selection of parameter grid points.

    Parameters
    ----------
    n_iter : int
        Number of grid points to sample.
    random_state : int, optional
        Random seed.
    kwargs : dict, optional
        Keyword arguments for GridSearchCV.
    """
    def __init__(self, n_iter, random_state=None, **kwargs):
        self.n_iter = n_iter
        self.random_state = random_state
        super(RandomGridSearchCV, self).__init__(**kwargs)

    def get_param_grid(self, param_grid):
        """
        Construct a parameter grid.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        return UniqueParameterSampler(param_grid, self.n_iter, None,
                                      self.random_state)
