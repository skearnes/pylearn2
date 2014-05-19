"""
Train ensemble models.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np

from pylearn2.cross_validation import TrainCV
from pylearn2.train import Train


class TrainEnsembleGridSearch(object):
    """
    Train an ensemble model using models derived from grid search.

    Parameters
    ----------
    dataset : Dataset
        Training dataset.
    grid_search : GridSearch
        Grid search object that trains models and sets a best_models
        attribute. The best_models attribute possibly contains results for
        each fold of cross-validation.
    algorithm : TrainingAlgorithm
        Training algorithm.
    save_path : str or None
        Output filename for trained models. Also used (with modification)
        for individual models if save_folds is True.
    save_freq : int
        Save frequency, in epochs. Only used if save_folds is True.
    extensions : list or None
        TrainExtension objects for individual Train objects.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    save_folds: bool
        Whether to write individual files for each cross-validation fold.
    cv_extensions : list or None
        TrainCVExtension objects for the parent TrainCV object.
    """
    def __init__(self, dataset, grid_search, algorithm=None, save_path=None,
                 save_freq=0, extensions=None, allow_overwrite=True):
        self.dataset = dataset
        self.grid_search = grid_search
        self.algorithm = algorithm
        self.save_path = save_path
        self.save_freq = save_freq
        self.extensions = extensions
        self.allow_overwrite = allow_overwrite

        self.ensemble_trainers = None

    def main_loop(self, time_budget):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int or None
            Maximum time (in seconds) before interrupting training.
        """
        self.grid_search.main_loop(time_budget)
        self.build_ensemble()
        for trainer in self.ensemble_trainers:
            trainer.main_loop(time_budget)

    def build_ensemble(self):
        trainers = []
        for models in np.atleast_2d(self.grid_search.best_models):
            trainer = Train(self.dataset, models, self.algorithm,
                            self.save_path, self.save_freq, self.extensions,
                            self.allow_overwrite)
            trainers.append(trainer)
        self.ensemble_trainers = trainers

class Ensemble(object):
    """
    Train an ensemble model.

    Parameters
    ----------
    models : list
        Models to combine.
    ensemble : str or None
        Ensemble type. If None, defaults to 'average'.
    kwargs : dict
        Keyword arguments for ensemble Train object.
    """
    def __init__(self, models, ensemble=None, **kwargs):
        self.models = models
        self.ensemble = ensemble
        self.trainer_kwargs = kwargs

        self.trainer = None
        self.build_ensemble()

    def main_loop(self, time_budget):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int or None
            Maximum time (in seconds) before interrupting training.
        """
        self.trainer.main_loop(time_budget)

    def build_ensemble(self):
        """Construct ensemble trainer."""



class EnsembleGridSearch(Ensemble):
    """
    Train an ensemble model using models derived from grid search.

    Parameters
    ----------
    grid_search : GridSearch
        Grid search object that trains models and sets a best_models
        attribute. The best_models attribute possibly contains results for
        each fold of cross-validation.
    ensemble : str or None
        Ensemble type. If None, defaults to 'average'.
    kwargs : dict
        Keyword arguments for ensemble Train object.
    """
    def __init__(self, grid_search, ensemble=None, **kwargs):
        self.grid_search = grid_search
        self.ensemble = ensemble
        self.trainer_kwargs = kwargs

        self.models = None
        self.trainer = None

    def main_loop(self, time_budget=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int or None
            Maximum time (in seconds) before interrupting training.
        """
        self.grid_search.main_loop(time_budget)
        self.build_ensemble()
        self.ensemble_trainer.main_loop(time_budget)

    def build_ensemble(self):
        """Construct ensemble model trainer."""


class TrainEnsemble2(object):
    """
    Train an ensemble model. Child models are trained first and selected
    by performance on validation set. Then the ensemble model is trained
    on the combined training and validation sets. Test set performance is
    only calculated for the ensemble model.

    The ensemble model is a Train object with the model defined by an MLP
    containing an EnsembleLayer.

    Parameters
    ----------
    ensemble : str or None
        Ensemble type. If None, defaults to 'average'.
    n_best : int or None
        Number of models to include in the ensemble. If None, all models
        are included.
    kwargs : dict
        Keyword arguments for ensemble Train object.
    """
    def __init__(self, trainers, ensemble=None, n_best=None, **kwargs):
        self.trainers = trainers
        self.n_best = n_best
        self.ensemble_kwargs = kwargs
        self.ensemble = ensemble
        self.ensemble_trainer = None

    def main_loop(self, time_budget=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int or None
            Maximum time (in seconds) before interrupting training.
        """
        # train children
        for trainer in self.trainers:
            trainer.main_loop(time_budget)

        # construct and train ensemble model
        self.build_ensemble()
        self.ensemble_trainer.main_loop(time_budget)
