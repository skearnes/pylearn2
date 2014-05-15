"""
Ensemble layers for MLPs.

If multiple models have been trained to predict the same targets, it is
often useful to combine their output to improve predictions, possibly by
learning a meta-estimator. The classes in this module are MLP layers that
wrap CompositeLayer. While CompositeLayers can be used in MLPs without any
wrapper (by flattening their output), these classes explicitly acknowledge
and take advantage of the relationships between the predictions of the
parallel layers.
"""
from theano import tensor as T

from pylearn2.models.mlp import Layer

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"


class Ensemble(Layer):
    """
    Ensemble layer that combines output from each layer in a
    CompositeLayer.

    Parameters
    ----------
    composite_layer : CompositeLayer
        Composite layer containing layers to run in parallel.
    layer_name : str
        Name of this layer.
    kwargs : dict
        Keyword arguments for `Layer` constructor.
    """
    def __init__(self, composite_layer, layer_name, **kwargs):
        super(Ensemble, self).__init__(**kwargs)
        self.composite_layer = composite_layer
        self.layer_name = layer_name

    def fprop(self, state_below):
        """
        Get predictions from each model in the ensemble.

        Parameters
        ----------
        state_below : Space
            Batch of examples to propogate through the composite layer.
        """
        raise NotImplementedError('fprop')


class EnsembleAverage(Ensemble):
    """
    Ensemble layer that calculates a (weighted) average of the output from
    the composite layer.

    Parameters
    ----------
    composite_layer : CompositeLayer
        Composite layer containing layers to run in parallel.
    layer_name : str
        Name of this layer.
    weights : list or None
        Contribution of each model in the weighted average.
    normalize : bool
        Whether to normalize averaged predictions so they sum to one.
    kwargs : dict
        Keyword arguments for `Layer` constructor.
    """
    def __init__(self, composite_layer, layer_name, weights=None,
                 normalize=True, **kwargs):
        super(EnsembleAverage, self).__init__(composite_layer, layer_name,
                                              **kwargs)
        if weights is None:
            weights = [1.0 for _ in composite_layer.layers]
        self.weights = weights
        self.normalize = normalize

    def fprop(self, state_below):
        """
        Average outputs from composite layer.

        Parameters
        ----------
        state_below : Space
            Batch of examples to propogate through each model.
        """
        rval = self.composite_layer.fprop(state_below)
        rval = T.mean(rval, axis=0)
        if self.normalize:
            rval /= rval.sum(axis=1).dimshuffle(0, 'x')
        return rval
