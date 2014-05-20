"""
Ensemble models.

TODO:
* More complicated layers: weighted average.
* Combinations of models trained on different input. This can be handled
with inputs_to_layers and ensemble_args.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np

from pylearn2.cross_validation import TrainCV
from pylearn2.ensemble.mlp import resolve_ensemble_layer
from pylearn2.models.mlp import MLP
from pylearn2.train import Train
from pylearn2.utils import safe_zip


class GridSearchEnsemble(object):
    """
    Train an ensemble model using models derived from grid search.

    Parameters
    ----------
    grid_search : GridSearch
        Grid search object that trains models and sets a best_models
        attribute. The best_models attribute possibly contains results for
        each fold of cross-validation.
    ensemble : str
        Ensemble type. Passed to resolve_ensemble_layer.
    dataset : Dataset or None
        Training dataset.
    dataset_iterator : iterable or None
        Cross validation dataset iterator.
    ensemble_args : dict or None
        Keyword arguments for ensemble layer.
    model_args : dict or None
        Keyword arguments for MLP. If None, nvis is extracted from one of
        the component models.
    algorithm : TrainingAlgorithm
        Training algorithm.
    save_path : str or None
        Output filename for trained model(s).
    save_freq : int
        Save frequency, in epochs.
    extensions : list or None
        TrainExtension objects for individual Train objects.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    cv_extensions : list or None
        TrainCVExtension objects for the parent TrainCV object.
    """
    def __init__(self, grid_search, ensemble, dataset=None,
                 dataset_iterator=None, ensemble_args=None, model_args=None,
                 algorithm=None, save_path=None, save_freq=0, extensions=None,
                 allow_overwrite=True):
        assert dataset is not None or dataset_iterator is not None, (
            "One of dataset or dataset_iterator must be provided.")
        assert dataset is None or dataset_iterator is None
        self.dataset = dataset
        self.dataset_iterator = dataset_iterator
        self.grid_search = grid_search
        self.ensemble = ensemble
        if ensemble_args is None:
            ensemble_args = {}
        self.ensemble_args = ensemble_args
        if model_args is None:
            model_args = {}
        self.model_args = model_args
        self.algorithm = algorithm
        self.save_path = save_path
        self.save_freq = save_freq
        self.extensions = extensions
        self.allow_overwrite = allow_overwrite

        # placeholder
        self.ensemble_trainer = None

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
        """
        Construct ensemble model trainer, possibly one for each fold of
        cross-validation.
        """
        if 'layer_name' not in self.ensemble_args:
            self.ensemble_args['layer_name'] = 'ensemble'
        if ('nvis' not in self.model_args or
                'input_space' not in self.model_args):
            self.model_args['input_space'] = np.atleast_2d(
                self.grid_search.best_models)[0, 0].input_space
        klass = resolve_ensemble_layer(self.ensemble)
        trainers = []
        if self.dataset_iterator is not None:
            assert self.grid_search.cv
            models = np.asarray(self.grid_search.best_models)
            if models.ndim == 1:
                models = np.atleast_2d(models).T
            for dataset, this_models in safe_zip(list(self.dataset_iterator),
                                                 models):
                layer = klass(layers=this_models, **self.ensemble_args)
                model = MLP(layers=[layer], **self.model_args)
                trainer = Train(dataset['train'], model, self.algorithm,
                                self.save_path, self.save_freq,
                                self.extensions, self.allow_overwrite)
                trainer.algorithm._set_monitoring_dataset(dataset)
                trainers.append(trainer)
        else:
            layer = klass(layers=self.grid_search.best_models,
                          **self.ensemble_args)
            model = MLP(layers=[layer], **self.model_args)
            trainer = Train(self.dataset, model, self.algorithm,
                            self.save_path, self.save_freq,
                            self.extensions, self.allow_overwrite)
            trainer.algorithm._set_monitoring_dataset(self.dataset)
            trainers = [trainer]
        self.ensemble_trainers = trainers
