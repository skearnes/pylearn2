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

    The ensemble model is a Train (or TrainCV) object with the model
    defined by an MLP containing an EnsembleLayer.

    How would models trained on different datasets be combined? By making
    a new dataset with two sets of features and using the inputs_to_layers
    functionality of EnsembleLayer.

    Parameters
    ----------
    trainers : list
        Train or TrainCV children.
    algorithm : TrainingAlgorithm
        Training algorithm.
    save_path : str or None
        Output filename for the model.
    save_freq : int
        Save frequency, in epochs.
    extensions : list or None
        Training extensions.
    allow_overwrite : bool
        Whether to overwrite a pre-existing output file matching save_path.
    """
    def __init__(self, trainers, algorithm=None, save_path=None, save_freq=0,
                 extensions=None, allow_overwrite=True):
        self.trainers = trainers
        self.algorithm = algorithm
        self.save_path = save_path
        self.save_freq = save_freq
        self.extensions = extensions
        self.allow_overwrite = allow_overwrite

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

        # select best models
