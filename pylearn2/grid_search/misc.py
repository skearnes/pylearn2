"""
Grid search helper functions and classes.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__credits__ = ["Steven Kearnes", "Bharath Ramsundar"]
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
import sys


class Empty(object):
    """
    Empty class, when None won't do (for example, when attributes are
    assigned to the object).
    """
    pass


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
