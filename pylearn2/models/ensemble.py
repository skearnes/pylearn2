"""
Ensemble models.

Most of the time these will be built on MLPs, which will have an fprop
method. Do we need to consider the case of an ensemble autoencoder, for
example?

Ensemble should behave something like an MLP, except with horizontal
stacking as well as vertical stacking. We could even allow the underlying
models to be mutable as in a MLP.
"""
from theano import tensor as T

from pylearn2.models import Model


class Ensemble(Model):
    """
    Ensemble model that combines predictions from two or more models.
    Subclasses perform different combinations, possibly learning a meta-
    model.

    Parameters
    ----------
    models : list
        Pretrained models.
    kwargs : dict
        Keyword arguments for Model constructor.
    """
    def __init__(self, models, **kwargs):
        super(Ensemble, self).__init__(**kwargs)
        self.models = models

    def fprop(self, state):
        """
        Get predictions from each model in the ensemble.

        Parameters
        ----------
        state : Space
            Batch of examples to propogate through each model.
        """
        rval = [model.fprop(state) for model in self.models]
        return rval


class EnsembleAverage(Ensemble):
    """
    Ensemble model that calculates a (weighted) average of the predictions
    of two or more models.

    Parameters
    ----------
    models : list
        Pretrained models.
    weights : list or None
        Contribution of each model in the weighted average.
    normalize : bool
        Whether to normalize averaged predictions so they sum to one.
    kwargs : dict
        Keyword arguments for Model constructor.
    """
    def __init__(self, models, weights, normalize=True, **kwargs):
        super(EnsembleAverage, self).__init__(models, **kwargs)
        if weights is None:
            weights = [1. for _ in models]
        self.weights = weights
        self.normalize = normalize

    def fprop(self, state):
        """
        Average model predictions.

        Parameters
        ----------
        state : Space
            Batch of examples to propogate through each model.
        """
        rval = super(EnsembleAverage, self).fprop(state)
        rval = T.mean(rval, axis=0)
        if self.normalize:
            rval /= rval.sum(axis=1).dimshuffle(0, 'x')
        return rval
