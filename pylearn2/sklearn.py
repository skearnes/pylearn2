"""
Train and TrainCV objects that wrap sklearn models.

This module allows sklearn models to be trained using pylearn2 datasets and
evaluated with sklearn estimator metrics, which are supported with
!!python/name tags in YAML. See tests for examples.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"
__maintainer__ = "Steven Kearnes"

from copy import deepcopy
import logging
import numpy as np
import os
import time

from pylearn2 import cross_validation, train

log = logging.getLogger(__name__)


def extract_data(dataset):
    """
    Extract data from a pylearn2 dataset.

    Parameters
    ----------
    dataset : Dataset
    """
    iterator = dataset.iterator(mode='sequential', num_batches=1,
                                data_specs=dataset.data_specs)
    data = tuple(iterator.next())
    if len(data) == 2 and data[1].ndim == 2:
        data = (data[0], np.argmax(data[1], axis=1))
    return data


class Train(train.Train):
    """
    Pylearn2-like interface for sklearn models.

    Parameters
    ----------
    dataset : Dataset
        Training dataset.
    model : sklearn Model
        Estimator with fit and predict/predict_proba/decision_function
        methods.
    save_path : str or None
        Output filename for trained models.
    save_freq : int
        Save frequency, in epochs.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    monitoring_dataset : Dataset or dict
        Datasets to monitor.
    metrics : list
        Functions used to calculate metrics on monitoring dataset(s). Each
        function should take true and predicted labels as arguments.
    predict_method : str or None
        One of 'decision_function', 'predict_proba', or 'predict'. If None,
        attempts each of those options successively.
    decision_function_index : int (default=0)
        Index of column in decision_function output to use in metrics.
    predict_proba_index : int (default=1)
        Index of positive class in predict_proba output.
    """
    def __init__(self, dataset, model, save_path=None, save_freq=0,
                 allow_overwrite=True, monitoring_dataset=None, metrics=None,
                 predict_method=None, decision_function_index=0,
                 predict_proba_index=1):
        self.dataset = dataset
        self.model = model
        self.model.monitor = Monitor()
        self.save_path = save_path
        self.save_freq = save_freq
        self.extensions = []
        self.allow_overwrite = allow_overwrite
        if monitoring_dataset is None:
            monitoring_dataset = dataset
        self.monitoring_dataset = monitoring_dataset
        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.predict_method = predict_method
        self.decision_function_index = decision_function_index
        self.predict_proba_index = predict_proba_index

        # carry-over attributes
        self.extensions = []
        self.first_save = True

    def main_loop(self, time_budget=None):
        """
        Fit the model and calculate metrics on monitoring dataset(s).

        Parameters
        ----------
        time_budget : int
            Not used.
        """
        train_data = extract_data(self.dataset)
        start = time.time()
        self.model.fit(*train_data)
        finish = time.time()
        elapsed = finish - start

        if isinstance(self.monitoring_dataset, dict):
            for name, dataset in self.monitoring_dataset.items():
                self.monitor_dataset(dataset, name)
        else:
            self.monitor_dataset(self.monitoring_dataset)
        if len(self.model.monitor.channels):
            for key in self.model.monitor.channels.keys():
                self.model.monitor.channels[key].time_record[0] = elapsed
                break
        if self.save_freq > 0:
            self.save()

    def monitor_dataset(self, dataset, name=None):
        """
        Calculate metric values for a dataset.

        Parameters
        ----------
        dataset : Dataset
            Monitoring dataset.
        name : str or None
            Dataset name to use in monitor channel labels.
        """
        data = extract_data(dataset)

        # predict
        if self.predict_method is None:
            try:
                p = self.model.decision_function(data[0])
                p = p[:, self.decision_function_index]
            except AttributeError:
                try:
                    p = self.model.predict_proba(data[0])
                    p = p[:, self.predict_proba_index]
                except AttributeError:
                    p = self.model.predict(data[0])
        else:
            if self.predict_method == 'decision_function':
                p = self.model.decision_function(data[0])
                p = p[:, self.decision_function_index]
            elif self.predict_method == 'predict_proba':
                p = self.model.predict_proba(data[0])
                p = p[:, self.predict_proba_index]
            elif self.predict_method == 'predict':
                p = self.model.predict(data[0])
            else:
                raise NotImplementedError('Unsupported predict_method ' +
                                          '"{}"'.format(self.predict_method))

        # calculate metrics
        for metric in np.atleast_1d(self.metrics):
            if name is not None:
                this_name = '{}_{}'.format(name, metric.__name__)
            else:
                this_name = metric.__name__
            value = metric(data[1], p)
            self.model.monitor.set_value(this_name, value)

    def set_monitoring_dataset(self, monitoring_dataset):
        """
        Set the monitoring dataset(s) for this trainer.

        Parameters
        ----------
        monitoring_dataset : Dataset or dict
            Dataset(s) to monitor.
        """
        self.monitoring_dataset = monitoring_dataset


class TrainCV(cross_validation.TrainCV):
    """
    Pylearn2-like cross-validation interface for sklearn models.

    Parameters
    ----------
    dataset_iterator : iterable
        Cross validation iterator providing (test, train) or (test, valid,
        train) datasets.
    model : Model or None
        Training model.
    model_iterator : iterable or None
        Training model for each fold. For example, models may have been
        pre-trained on these folds and used for building an ensemble model.
    save_path : str or None
        Output filename for trained models. Also used (with modification)
        for individual models if save_folds is True.
    save_freq : int
        Save frequency, in epochs. Only used if save_folds is True.
    allow_overwrite : bool
        Whether to overwrite pre-existing output file matching save_path.
    save_folds: bool
        Whether to write individual files for each cross-validation fold.
    metrics : list
        Functions used to calculate metrics on monitoring dataset(s). Each
        function should take true and predicted labels as arguments.
    """
    def __init__(self, dataset_iterator, model=None, model_iterator=None,
                 save_path=None, save_freq=0, allow_overwrite=True,
                 save_folds=False, metrics=None):
        self.dataset_iterator = dataset_iterator
        assert model is not None or model_iterator is not None, (
            "One of model or model_iterator must be provided.")
        assert model is None or model_iterator is None
        trainers = []
        for k, datasets in enumerate(dataset_iterator):
            if save_folds:
                path, ext = os.path.splitext(save_path)
                this_save_path = path + "-{}".format(k) + ext
                this_save_freq = save_freq
            else:
                this_save_path = None
                this_save_freq = 0

            # setup model
            if model is not None:
                this_model = deepcopy(model)
            else:
                this_model = deepcopy(model_iterator[k])

            # construct an isolated Train object
            # no shared references between trainers are allowed
            # (hence all the deepcopy operations)
            try:
                assert isinstance(datasets, dict)
                trainer = Train(datasets['train'], this_model, this_save_path,
                                this_save_freq, allow_overwrite, datasets,
                                metrics)
            except AssertionError:
                raise AssertionError("Dataset iterator must be a dict with " +
                                     "dataset names (e.g. 'train') as keys.")
            except KeyError:
                raise KeyError("Dataset iterator must yield training data.")
            trainers.append(trainer)
        self.trainers = trainers
        self.save_path = save_path
        self.allow_overwrite = allow_overwrite

        # carry-over attributes
        self.extensions = []
        self.cv_extensions = []


class Monitor(object):
    """Monitor for a sklearn model."""
    def __init__(self):
        self.channels = {}
        self._epochs_seen = 1

    def set_value(self, channel_name, value):
        """
        Set the value of a monitor channel.

        Parameters
        ----------
        channel_name : str
            Name of the monitor channel to update.
        value : object (usually int or float)
            Channel value.
        """
        if channel_name not in self.channels:
            self.channels[channel_name] = MonitorChannel(channel_name)
        self.channels[channel_name].set_value(value)


class MonitorChannel(object):
    """
    Monitor channel for a sklearn model.

    Parameters
    ----------
    name : str
        Channel name.
    """
    def __init__(self, name):
        self.name = name
        self.val_record = [None]
        self.time_record = [0]

    def set_value(self, value):
        """
        Set channel value.

        Parameters
        ----------
        value : int or float
            Channel value.
        """
        self.val_record[0] = value
        log.info('\t{}: {}'.format(self.name, value))
