"""
Hyperparameter grid search.

Template YAML files should use %()s substitutions for parameters.

Some fields are filled in automatically:
* save_path
* best_save_path

Grid search protocol:
* Create a trainer for each sampled grid point.
* Run main_loop of each trainer.
* If monitor_channel is specified, extract score for each model.
* If retrain is True, create n_best new trainers (using retrain_kwargs) and
  run each main_loop.
* The output written to save_path is a dict containing grid points and
  scores.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__credits__ = ["Steven Kearnes", "Bharath Ramsundar"]
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

import gc
import numpy as np
import os
try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    ParameterGrid = None

from pylearn2.config import yaml_parse
from pylearn2.cross_validation import TrainCV
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.utils import serial


class GridSearch(object):
    """
    Hyperparameter grid search using a YAML template. A trainer is
    constructed for each grid point using the template. If desired, the
    best models can be chosen by specifying a monitor channel to use for
    ranking. Additionally, if MonitorBasedSaveBest is used as a training
    extension in the template, rankings will be determined using the model
    extracted from that extension.

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
    save_path : str, optional
        Output filename for scores. Also used (with modification) for
        individual models if template contains %(save_path)s or
        %(best_save_path)s.
    allow_overwrite : bool, optional (default True)
        Whether to overwrite pre-existing output file matching save_path.
    monitor_channel : str, optional
        Monitor channel to use to compare models.
    higher_is_better : bool, optional (default False)
        Whether higher monitor_channel values correspond to better models.
    n_best : int, optional
        Maximum number of models to save, ranked by monitor_channel value.
    retrain : bool, optional (default False)
        Whether to retrain the best model(s). The training dataset is the
        union of the training and validation sets (if any).
    retrain_kwargs : dict, optional
        Keyword arguments to modify the template trainer prior to
        retraining. Must contain 'dataset' or 'dataset_iterator'.
    """
    def __init__(self, template, param_grid, save_path=None,
                 allow_overwrite=True, monitor_channel=None,
                 higher_is_better=False, n_best=None, retrain=False,
                 retrain_kwargs=None):
        self.template = template
        for key, value in param_grid.items():
            param_grid[key] = np.atleast_1d(value)  # must be iterable
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite
        self.monitor_channel = monitor_channel
        self.higher_is_better = higher_is_better
        self.n_best = n_best
        self.retrain = retrain
        if retrain:
            assert n_best is not None
            assert retrain_kwargs is not None
            assert ('dataset' in retrain_kwargs or
                    'dataset_iterator' in retrain_kwargs)
        self.retrain_kwargs = retrain_kwargs

        # construct parameter grid
        self.params = None
        self.get_params(param_grid)

        # placeholders
        self.scores = None
        self.best_params = None
        self.best_scores = None

    def get_params(self, param_grid):
        """
        Construct parameter grid.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        parameters = []
        for grid_point in self.get_param_grid(param_grid):
            if self.save_path is not None:
                prefix, ext = os.path.splitext(self.save_path)
                for key, value in grid_point.items():
                    prefix += '-{}_{}'.format(key, value)
                grid_point['save_path'] = prefix + ext
                grid_point['best_save_path'] = prefix + '-best' + ext
            parameters.append(grid_point)
        assert len(parameters) > 1  # why are you doing a grid search?
        self.params = np.asarray(parameters)

    def get_param_grid(self, param_grid):
        """
        Construct a parameter grid.

        Parameters
        ----------
        param_grid : dict
            Parameter grid.
        """
        if ParameterGrid is None:
            raise RuntimeError("Could not import from sklearn.")
        return ParameterGrid(param_grid)

    def get_trainers(self):
        """Build a generator for trainers on each grid point."""
        trainers = (yaml_parse.load(self.template % params)
                    for params in self.params)
        return trainers

    def score_grid(self, time_budget=None, parallel=False, client_kwargs=None,
                   view_flags=None):
        """
        Construct trainers and run main_loop, then return scores, if
        requested.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool, optional (default False)
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict, optional
            Keyword arguments for IPython.parallel.Client.
        view_flags : dict, optional
            Flags for IPython.parallel LoadBalancedView.
        """
        scores = []

        if parallel:
            from IPython.parallel import Client

            if client_kwargs is None:
                client_kwargs = {}
            if view_flags is None:
                view_flags = {}
            client = Client(**client_kwargs)
            view = client.load_balanced_view()
            view.set_flags(**view_flags)

            # submit jobs
            calls = []
            for trainer in self.get_trainers():
                if isinstance(trainer, TrainCV):
                    trainer.setup()
                    call = view.map_async(
                        self.train, trainer.trainers,
                        [time_budget] * len(trainer.trainers))
                    calls.append(call)
                else:
                    call = view.map_async(self.train, [trainer], [time_budget])
                    calls.append(call)

                # cleanup
                del trainer
                gc.collect()

            # extract scores
            for trainer, call in zip(self.get_trainers(), calls):
                if isinstance(trainer, TrainCV):
                    trainer.trainers = call.get()
                    trainer.save()
                    models = self.get_models(trainer, self.monitor_channel,
                                             self.higher_is_better)
                    score = self.get_scores(models, self.monitor_channel)
                else:
                    trainer, = call.get()
                    models = self.get_models(trainer, self.monitor_channel,
                                             self.higher_is_better)
                    score, = self.get_scores(models, self.monitor_channel)
                scores.append(score)

                # cleanup
                view.purge_results(call.msg_ids)
                del trainer, call, models
                gc.collect()

        else:
            for trainer in self.get_trainers():
                trainer.main_loop(time_budget)
                if isinstance(trainer, TrainCV):
                    models = self.get_models(trainer, self.monitor_channel,
                                             self.higher_is_better)
                    score = self.get_scores(models, self.monitor_channel)
                else:
                    models = self.get_models(trainer, self.monitor_channel,
                                             self.higher_is_better)
                    score, = self.get_scores(models, self.monitor_channel)
                scores.append(score)

                # cleanup
                del trainer, models
                gc.collect()

        scores = np.asarray(scores)
        self.scores = scores

    @staticmethod
    def train(trainer, time_budget=None):
        """
        Run trainer main_loop and return trainer. This method is static so
        it can be used in parallel execution.

        Parameters
        ----------
        trainer : Train
            Trainer.
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        """
        trainer.main_loop(time_budget)

        # reduce memory footprint
        if isinstance(trainer, TrainCV):
            trainers = trainer.trainers
        else:
            trainers = [trainer]
        for trainer in trainers:
            trainer.dataset = type('Empty', (object,), {})
            trainer.algorithm = None
        gc.collect()

        return trainer

    @staticmethod
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
        return models

    @staticmethod
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
        for model in models:
            monitor = model.monitor
            score = monitor.channels[channel_name].val_record[-1]
            scores.append(score)
        scores = np.asarray(scores)
        return scores

    def get_best_params(self):
        """Get best grid parameters and associated scores."""
        best_scores = None
        best_params = None
        if self.n_best is not None:
            sort = np.argsort(self.scores, axis=0)
            if self.higher_is_better:
                sort = sort[::-1]
            if self.scores.ndim == 2:
                best_scores = np.zeros_like(self.scores)
                for i in xrange(self.scores.shape[1]):
                    best_scores[:, i] = self.scores[:, i][sort[:, i]]
                best_scores = best_scores[:self.n_best]
            else:
                best_scores = self.scores[sort][:self.n_best]
            best_params = self.params[sort][:self.n_best]

            # update save_path and best_save_path for best_params
            if self.save_path is not None:
                for i, params in enumerate(best_params):
                    for key in ['save_path', 'best_save_path']:
                        val = params[key]
                        prefix, ext = os.path.splitext(val)
                        prefix += '-retrain'
                        best_params[i][key] = prefix + ext

        self.best_scores = best_scores
        self.best_params = best_params

    def main_loop(self, time_budget=None, parallel=False, client_kwargs=None,
                  view_flags=None):
        """
        Run main_loop of each trainer.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool, optional (default False)
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict, optional
            Keyword arguments for IPython.parallel.Client.
        view_flags : dict, optional
            Flags for IPython.parallel LoadBalancedView.
        """
        self.score_grid(time_budget, parallel, client_kwargs, view_flags)
        self.get_best_params()
        if self.retrain:
            self.retrain_best_models(
                time_budget, parallel, client_kwargs, view_flags)
        self.save()

    def save(self):
        """Save params and scores."""
        data = {'params': self.params,
                'monitor_channel': self.monitor_channel,
                'higher_is_better': self.higher_is_better,
                'scores': self.scores,
                'best_params': self.best_params,
                'best_scores': self.best_scores}
        if self.save_path is not None:
            serial.save(self.save_path, data, on_overwrite='backup')

    def retrain_best_models(self, time_budget=None, parallel=False,
                            client_kwargs=None, view_flags=None):
        """
        Retrain best models using dataset from retrain_kwargs.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        parallel : bool, optional (default False)
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict, optional
            Keyword arguments for IPython.parallel.Client.
        view_flags : dict, optional
            Flags for IPython.parallel LoadBalancedView.
        """
        trainers = []

        # TrainCV templates
        # update trainer for each fold with best params
        if self.best_params.ndim == 2:
            dataset_iterator = self.retrain_kwargs['dataset_iterator']
            for params in self.best_params:
                trainer = None
                this_trainers = []
                for k, datasets in enumerate(dataset_iterator):
                    trainer = yaml_parse.load(self.template % params[k])
                    this_trainer = trainer.trainers[k]
                    this_trainer.dataset = datasets['train']
                    this_trainer.algorithm._set_monitoring_dataset(datasets)
                    this_trainers.append(this_trainer)
                trainer.trainers = this_trainers
                trainers.append(trainer)
        else:
            for params in self.best_params:
                trainer = yaml_parse.load(self.template % params)
                dataset = self.retrain_kwargs['dataset']
                if isinstance(dataset, dict):
                    trainer.dataset = dataset['train']
                    trainer.algorithm._set_monitoring_dataset(dataset)
                else:
                    trainer.dataset = dataset
                    trainer.algorithm._set_monitoring_dataset(
                        {'train': dataset})
                trainers.append(trainer)

        # this could be parallelized better when n_best > 1
        for trainer in trainers:
            if isinstance(trainer, TrainCV):
                trainer.main_loop(time_budget, parallel, client_kwargs,
                                  view_flags)
            else:
                trainer.main_loop(time_budget)


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
    save_path : str, optional
        Output filename for trained model(s). Also used (with modification)
        for individual models if template contains %(save_path)s or
        %(best_save_path)s fields.
    allow_overwrite : bool, optional (default True)
        Whether to overwrite pre-existing output file matching save_path.
    monitor_channel : str, optional
        Monitor channel to use to compare models.
    higher_is_better : bool, optional (default False)
        Whether higher monitor_channel values correspond to better models.
    n_best : int, optional
        Maximum number of models to save, ranked by monitor_channel value.
    retrain : bool, optional (default True)
        Whether to train the best model(s).
    retrain_kwargs : dict, optional
        Keyword arguments to modify the template trainer prior to
        retraining. If not provided when retrain is True, the dataset is
        extracted from the template dataset_iterator. Otherwise,
        retrain_kwargs must contain 'dataset', which can be a Dataset or
        a dict containing at least a 'train' dataset.
    """
    def __init__(self, template, param_grid, save_path=None,
                 allow_overwrite=True, monitor_channel=None,
                 higher_is_better=False, n_best=None, retrain=True,
                 retrain_kwargs=None):
        super(GridSearchCV, self).__init__(template, param_grid, save_path,
                                           allow_overwrite, monitor_channel,
                                           higher_is_better, n_best)
        self.retrain = retrain
        if retrain_kwargs is not None:
            assert 'dataset' in retrain_kwargs
            if isinstance(retrain_kwargs['dataset'], dict):
                assert 'train' in retrain_kwargs['dataset']
        self.retrain_kwargs = retrain_kwargs

    def score_grid(self, *args, **kwargs):
        """
        Get scores for each grid point and then average scores across
        cross-validation folds.

        Parameters
        ----------
        args : list
            Positional arguments.
        kwargs : dict
            Keyword arguments.
        """
        super(GridSearchCV, self).score_grid(*args, **kwargs)
        self.scores = np.mean(self.scores, axis=1)

    def retrain_best_models(self, time_budget=None, parallel=False,
                            client_kwargs=None, view_flags=None):
        """
        Train best models on full dataset.

        Parameters
        ----------
        time_budget : int, optional
            Maximum number of seconds before interrupting training.
        parallel : bool, optional (default False)
            Whether to train subtrainers in parallel using
            IPython.parallel.
        client_kwargs : dict, optional
            Keyword arguments for IPython.parallel.Client.
        view_flags : dict, optional
            Flags for IPython.parallel LoadBalancedView.
        """
        if self.retrain_kwargs is not None:
            dataset = self.retrain_kwargs['dataset']
        else:
            dataset = self.get_trainers().next().dataset_iterator.dataset
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

        if parallel:
            from IPython.parallel import Client

            if client_kwargs is None:
                client_kwargs = {}
            if view_flags is None:
                view_flags = {}
            client = Client(**client_kwargs)
            view = client.load_balanced_view()
            view.set_flags(**view_flags)
            view.map_sync(self.train, trainers, [time_budget] * len(trainers))
        else:
            for trainer in trainers:
                trainer.main_loop(time_budget)
