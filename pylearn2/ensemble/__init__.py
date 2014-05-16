"""
Train ensemble models.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

from pylearn2.cross_validation import TrainCV
from pylearn2.train import Train


class TrainEnsemble(object):
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

    def build_ensemble(self):
        """Construct ensemble model."""

