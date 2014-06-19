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

from pylearn2.cross_validation import TrainCV
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest


def get_models(trainer, channel_name=None, higher_is_better=False):
    """
    Extract the model from this trainer, possibly taking the best model
    from MonitorBasedSaveBest.

    Parameters
    ----------
    trainer : Train or TrainCV
        Trainer.
    channel_name : str, optional
        Monitor channel to match in MonitorBasedSaveBest.
    higher_is_better : bool, optional
        Whether higher channel values indicate better models (default
        False).
    """
    if isinstance(trainer, TrainCV):
        trainers = trainer.trainers
    else:
        trainers = [trainer]
    models = []
    for trainer in trainers:
        model = None
        for extension in trainer.extensions:
            if (isinstance(extension, MonitorBasedSaveBest) and
                    extension.channel_name == channel_name):

                # These are assertions and not part of the conditional
                # since failures are likely to indicate errors in the
                # input YAML.
                assert extension.higher_is_better == higher_is_better
                assert extension.store_best_model
                model = extension.best_model
                break
        if model is None:
            model = trainer.model
        models.append(model)
    if len(models) == 1:
        models, = models
    return models


def get_scores(models, channel_name):
    """
    Get scores for models.

    Parameters
    ----------
    model : Model
        Model.
    channel_name : str
        Monitor channel.
    """
    scores = []
    for model in np.atleast_1d(models):
        monitor = model.monitor
        score = monitor.channels[channel_name].val_record[-1]
        scores.append(score)
    if len(scores) == 1:
        scores, = scores
    else:
        scores = np.asarray(scores)
    return scores


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
