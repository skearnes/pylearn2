"""
Hyperparameter grid search.

Template YAML files should use %()s substitutions for parameters.

Some fields are filled in automatically:
* seed_1, seed_2, ..., seed_n (random seeds)
* save_path
* best_save_path
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__credits__ = ["Steven Kearnes", "Bharath Ramsundar"]
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import numpy as np
import os
import sys
import warnings
try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    warnings.warn("Could not import from sklearn.")

from pylearn2.config import yaml_parse
from pylearn2.cross_validation import TrainCV
from pylearn2.train import SerializationGuard
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.utils import serial


def random_seeds(size, random_state=None):
    """
    Generate random seeds.

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


def batch_train(trainers, time_budget=None, parallel=False,
                client_kwargs=None):
    """
    Run main_loop of each trainer. When run in parallel, all child trainers
    (including children of TrainCV children) are added to the queue so
    grid points and cross-validation folds are all processed simultanously.

    Parameters
    ----------
    time_budget : int, optional
        The maximum number of seconds before interrupting
        training. Default is `None`, no time limit.
    parallel : bool
        Whether to train subtrainers in parallel using
        IPython.parallel.
    client_kwargs : dict or None
        Keyword arguments for IPython.parallel.Client.
    """
    if parallel:
        from IPython.parallel import Client

        def _train(trainer, time_budget=None):
            """
            Run main_loop of each trainer.

            Parameters
            ----------
            time_budget : int, optional
                The maximum number of seconds before interrupting
                training. Default is `None`, no time limit.
            """
            trainer.main_loop(time_budget)
            return trainer

        if client_kwargs is None:
            client_kwargs = {}
        client = Client(**client_kwargs)
        view = client.load_balanced_view()

        # get all child trainers and run them in parallel, which
        # simultaneously parallelizes over both grid points and
        # cross-validation folds
        calls = []
        for trainer in trainers:
            if isinstance(trainer, TrainCV):
                call = view.map(_train, trainer.trainers, block=False)
                calls.append(call)
            else:
                call = view.map(_train, [trainer], block=False)
                calls.append(call)
        for i, (trainer, call) in enumerate(zip(trainers, calls)):
            if isinstance(trainer, TrainCV):
                trainers[i].trainers = call.get()
            else:
                trainers[i], = call.get()
    else:
        for trainer in trainers:
            trainer.main_loop(time_budget)
    return trainers


class GridSearch(object):
    """
    Hyperparameter grid search using a YAML template. A trainer is
    constructed for each grid point using the template. If desired, the
    best models can be chosen by specifying a monitor channel to use for
    ranking. Additionally, if MonitorBasedStoreBest is used as a training
    extension in the template, rankings will be determined using the best
    models extracted from those extensions.

    Parameters
    ----------
    template : str
        YAML template, possibly containing % formatting fields.
    param_grid : dict
        Parameter grid, with keys matching template fields. Additional
        keys will also be used to generate additional models. For example,
        {'n': [1, 2, 3]} (when no %(n)s field exists in the template) will
        cause each model to be trained three times; useful when working
        with stochastic models.
    save_path : str or None
        Output filename for trained model(s). Also used (with modification)
        for individual models if template contains %(save_path)s or
        %(best_save_path)s fields.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    monitor_channel : str or None
        Monitor channel to use to compare models.
    higher_is_better : bool
        Whether higher monitor_channel values correspond to better models.
    n_best : int or None
        Maximum number of models to save, ranked by monitor_channel value.
    """
    def __init__(self, template, param_grid, save_path=None,
                 allow_overwrite=True, monitor_channel=None,
                 higher_is_better=False, n_best=None):
        self.template = template
        for key, value in param_grid.items():
            param_grid[key] = np.atleast_1d(value)  # must be iterable
        param_grid = ParameterGrid(param_grid)
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite
        self.monitor_channel = monitor_channel
        self.higher_is_better = higher_is_better
        self.n_best = n_best

        # construct a trainer for each grid point
        self.cv = False  # True if best_models is indexed by cv fold
        self.trainers = None
        self.params = None
        self.get_trainers(param_grid)

        # placeholders
        self.models = None
        self.scores = None
        self.best_models = None
        self.best_params = None
        self.best_scores = None

    def get_trainers(self, param_grid):
        """
        Construct a trainer for each grid point.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        trainers = []
        parameters = []
        for grid_point in param_grid:

            # build output filename
            if self.save_path is not None:
                prefix, ext = os.path.splitext(self.save_path)
                for key, value in grid_point.items():
                    prefix += '-{}_{}'.format(key, value)
                grid_point['save_path'] = prefix + ext
                grid_point['best_save_path'] = prefix + '-best' + ext

            # construct trainer
            trainer = yaml_parse.load(self.template % grid_point)
            trainers.append(trainer)
            parameters.append(grid_point)
        if isinstance(trainers[0], TrainCV):
            self.cv = True
        self.trainers = trainers
        self.params = parameters

    def score(self, models):
        """
        Score models.

        Parameters
        ----------
        models : list
            Models to score.
        """
        scores = None
        if self.monitor_channel is not None:
            scores = []
            for model in models:
                monitor = model.monitor
                score = monitor.channels[self.monitor_channel].val_record[-1]
                scores.append(score)
            scores = np.asarray(scores)
        return scores

    def get_best_models(self, trainers=None):
        """
        Get best models. If MonitorBasedStoreBest is used in the template
        with self.monitor_channel, then take the best model from that
        extension.

        Parameters
        ----------
        trainers : list or None
            Trainers from which to extract models. If None, defaults to
            self.trainers.
        """
        if trainers is None:
            trainers = self.trainers

        # special handling for TrainCV templates
        if isinstance(trainers[0], TrainCV):
            return self.get_best_cv_models()

        # test for MonitorBasedSaveBest
        models = []
        for trainer in trainers:
            found = False
            for extension in trainer.extensions:
                if found:
                    break
                if (isinstance(extension, MonitorBasedSaveBest) and
                        extension.channel_name == self.monitor_channel):
                    models.append(extension.best_model)
                    found = True
            if not found:
                models.append(trainer.model)
        models = np.asarray(models)
        params = np.asarray(self.params)
        scores = self.score(models)
        best_models = None
        best_params = None
        best_scores = None
        if scores is not None and self.n_best is not None:
            sort = np.argsort(scores)
            if self.higher_is_better:
                sort = sort[::-1]
            best_models = models[sort][:self.n_best]
            best_params = params[sort][:self.n_best]
            best_scores = scores[sort][:self.n_best]
            if len(best_models) == 1:
                best_models, = best_models
                best_params, = best_params
                best_scores, = best_scores
        self.models = models
        self.scores = scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores
        return models, scores, best_models, best_params, best_scores

    def get_best_cv_models(self):
        """
        Get best models from each cross-validation fold. This is different
        than using cross-validation to select hyperparameters.

        The first dimension of self.best_models is the fold index.
        """
        models = []
        params = []
        scores = []
        best_models = []
        best_params = []
        best_scores = []
        for k in xrange(len(self.trainers[0].trainers)):
            trainers = [trainer.trainers[k] for trainer in self.trainers]
            (this_models,
             this_scores,
             this_best_models,
             this_best_params,
             this_best_scores) = self.get_best_models(trainers)
            models.append(this_models)
            params.append(self.params)
            if this_scores is not None:
                scores.append(this_scores)
            if this_best_scores is not None:
                best_models.append(this_best_models)
                best_params.append(this_best_params)
                best_scores.append(this_best_scores)
        self.models = models
        self.params = params
        if not len(scores):
            scores = None
        if not len(best_scores):
            best_models = None
            best_params = None
            best_scores = None
        self.scores = scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores

    def main_loop(self, time_budget=None, parallel=False, client_kwargs=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict or None
            Keyword arguments for IPython.parallel.Client.
        """
        self.trainers = batch_train(self.trainers, time_budget, parallel,
                                    client_kwargs)
        self.get_best_models()
        self.save()

    def save(self, trainers=None):
        """
        Serialize best-scoring models.

        Parameters
        ----------
        trainers : list or None
            Parent trainer(s) of the model(s) to save. If None, defaults to
            self.trainers.
        """
        if trainers is None:
            trainers = self.trainers
            if self.best_models is not None:
                models = self.best_models
            else:
                models = self.models
        else:
            models = [trainer.model for trainer in trainers]

        # Train extensions
        # TrainCV calls on_save automatically
        for trainer in trainers:
            if not isinstance(trainer, TrainCV):
                for extension in trainer.extensions:
                    extension.on_save(trainer.model, trainer.dataset,
                                      trainer.algorithm)

        try:
            for trainer in trainers:
                if isinstance(trainer, TrainCV):
                    for t in trainer.trainers:
                        t.dataset._serialization_guard = SerializationGuard()
                else:
                    trainer.dataset._serialization_guard = SerializationGuard()
            if self.save_path is not None:
                if not self.allow_overwrite and os.path.exists(self.save_path):
                    raise IOError("Trying to overwrite file when not allowed.")
                serial.save(self.save_path, models, on_overwrite='backup')
        finally:
            for trainer in trainers:
                if isinstance(trainer, TrainCV):
                    for t in trainer.trainers:
                        t.dataset._serialization_guard = None
                else:
                    trainer.dataset._serialization_guard = None


class GridSearchCV(GridSearch):
    """
    Use a TrainCV template to select the best hyperparameters by cross-
    validation.

    Parameters
    ----------
    template : str
        YAML template, possibly containing % formatting fields.
    param_grid : dict
        Parameter grid, with keys matching template fields. Additional
        keys will also be used to generate additional models. For example,
        {'n': [1, 2, 3]} (when no %(n)s field exists in the template) will
        cause each model to be trained three times; useful when working
        with stochastic models.
    save_path : str or None
        Output filename for trained model(s). Also used (with modification)
        for individual models if template contains %(save_path)s or
        %(best_save_path)s fields.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    monitor_channel : str or None
        Monitor channel to use to compare models.
    higher_is_better : bool
        Whether higher monitor_channel values correspond to better models.
    n_best : int or None
        Maximum number of models to save, ranked by monitor_channel value.
    retrain : bool
        Whether to train the best model(s).
    retrain_dataset : Dataset, dict, or None
        Dataset or dict of datasets to use for training best model(s). If
        None, the dataset is extracted from TrainCV.dataset_iterator. If a
        dict, it must contain a 'train' key.
    """
    def __init__(self, template, param_grid, save_path=None,
                 allow_overwrite=True, monitor_channel=None,
                 higher_is_better=False, n_best=None, retrain=True,
                 retrain_dataset=None):
        super(GridSearchCV, self).__init__(template, param_grid, save_path,
                                           allow_overwrite, monitor_channel,
                                           higher_is_better, n_best)
        self.cv = False  # True if best_models is indexed by cv fold
        self.retrain = retrain
        self.retrain_dataset = retrain_dataset

    def get_best_cv_models(self):
        """
        Get best models by averaging scores over all cross-validation
        folds.
        """
        super(GridSearchCV, self).get_best_cv_models()
        if self.scores is None or self.n_best is None:
            return
        mean_scores = np.mean(self.scores, axis=0)
        sort = np.argsort(mean_scores)
        if self.higher_is_better:
            sort = sort[::-1]
        best_models = np.atleast_1d(self.models[0])[sort][:self.n_best]
        best_params = np.atleast_1d(self.params[0])[sort][:self.n_best]
        best_scores = mean_scores[sort][:self.n_best]
        if len(best_models) == 1:
            best_models, = best_models
            best_params, = best_params
            best_scores, = best_scores
        self.best_models = best_models
        self.best_params = best_params
        self.best_scores = best_scores

    def main_loop(self, time_budget=None, parallel=False, client_kwargs=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict or None
            Keyword arguments for IPython.parallel.Client.
        """
        self.trainers = batch_train(self.trainers, time_budget, parallel,
                                    client_kwargs)
        self.get_best_cv_models()
        trainers = None
        if self.retrain:
            trainers = self.retrain_best_models(time_budget, parallel,
                                                client_kwargs)
        self.save(trainers)

    def retrain_best_models(self, time_budget=None, parallel=False,
                            client_kwargs=None):
        """
        Train best models on full dataset.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict or None
            Keyword arguments for IPython.parallel.Client.
        """
        if self.retrain_dataset is not None:
            dataset = self.retrain_dataset
        else:
            dataset = self.trainers[0].dataset_iterator.dataset
        trainers = []
        for params in np.atleast_1d(self.best_params):
            parent = yaml_parse.load(self.template % params)
            trainer = parent.trainers[0]
            if isinstance(dataset, dict):
                trainer.dataset = dataset['train']
                trainer.algorithm._set_monitoring_dataset(dataset)
            else:
                trainer.dataset = dataset
                trainer.algorithm._set_monitoring_dataset({'train': dataset})
            trainers.append(trainer)
        trainers = batch_train(trainers, time_budget, parallel, client_kwargs)
        return trainers
