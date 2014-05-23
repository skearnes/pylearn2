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
from pylearn2.ensemble.mlp import resolve_ensemble_type
from pylearn2.models.mlp import MLP
from pylearn2.train import Train


class GridSearchEnsemble(object):
    """
    Train an ensemble model using models derived from grid search.

    Parameters
    ----------
    grid_search : GridSearch
        Grid search object that trains models and sets a best_models
        attribute. The best_models attribute possibly contains results for
        each fold of cross-validation.
    ensemble_type : str
        Ensemble type. Passed to resolve_ensemble_type.
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
    save_folds: bool
        Whether to write individual files for each cross-validation fold.
        Only used if dataset_iterator is used.
    cv_extensions : list or None
        TrainCVExtension objects for the parent TrainCV object. Only used
        if dataset_iterator is used.
    """
    def __init__(self, grid_search, ensemble_type, dataset=None,
                 dataset_iterator=None, ensemble_args=None, model_args=None,
                 algorithm=None, save_path=None, save_freq=0, extensions=None,
                 allow_overwrite=True, save_folds=False, cv_extensions=None):
        assert dataset is not None or dataset_iterator is not None, (
            "One of dataset or dataset_iterator must be provided.")
        assert dataset is None or dataset_iterator is None
        self.dataset = dataset
        self.dataset_iterator = dataset_iterator
        self.grid_search = grid_search
        self.ensemble_type = ensemble_type
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
        self.save_folds = save_folds
        self.cv_extensions = cv_extensions

        # placeholder
        self.ensemble_trainer = None

    def main_loop(self, time_budget=None, parallel=False, client_kwargs=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int or None
            Maximum time (in seconds) before interrupting training.
        parallel : bool
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict or None
            Keyword arguments for IPython.parallel.Client.
        """
        self.grid_search.main_loop(time_budget, parallel, client_kwargs)
        self.build_ensemble()
        if isinstance(self.ensemble_trainer, TrainCV):
            self.ensemble_trainer.main_loop(time_budget, parallel,
                                            client_kwargs)
        else:
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
        klass = resolve_ensemble_type(self.ensemble_type)
        if self.dataset_iterator is not None:
            assert self.grid_search.cv
            models = self.grid_search.best_models
            if models.ndim == 1:
                models = np.atleast_2d(models).T
            model_iterator = []
            for this_models in models:
                layer = klass(layers=this_models, **self.ensemble_args)
                model = MLP(layers=[layer], **self.model_args)
                model_iterator.append(model)
            trainer = TrainCV(self.dataset_iterator, None, model_iterator,
                              self.algorithm, self.save_path, self.save_freq,
                              self.extensions, self.allow_overwrite,
                              self.save_folds, self.cv_extensions)
        else:
            layer = klass(layers=self.grid_search.best_models,
                          **self.ensemble_args)
            model = MLP(layers=[layer], **self.model_args)
            trainer = Train(self.dataset, model, self.algorithm,
                            self.save_path, self.save_freq,
                            self.extensions, self.allow_overwrite)
        self.ensemble_trainer = trainer
