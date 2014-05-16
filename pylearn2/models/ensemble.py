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

from pylearn2.models.mlp import CompositeLayer
from pylearn2.utils import safe_zip

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"


def get_z_from_p(p):
    """
    Back out softmax inputs from probabilities. This is required because
    the Softmax cost function checks that Y_hat is coming from softmax and
    uses the raw inputs when calculating log probabilities.
    """

class Ensemble(CompositeLayer):
    """
    Subclass of CompositeLayer that does special handling of output,
    taking advantage of the assumed relationships between the parallel
    layers.

    Note that the output spaces of each component layer should match.

    Parameters
    ----------
    layer_name : str
        Name of this layer.
    layers : tuple or list
        The component layers to run in parallel.
    probabilities : bool
        Assume that component layer outputs are class probabilities.
    inputs_to_layers : dict or None
        Mapping for inputs to component layers.
    """
    def __init__(self, layer_name, layers, probabilities=True,
                 inputs_to_layers=None):
        super(Ensemble, self).__init__(layer_name, layers, inputs_to_layers)
        self.probabilities = probabilities

    def set_input_space(self, space):
        """
        Set the input space of this layer. CompositeLayer also uses this
        method to set the output space. Here we check that all of the
        layers have the same output space and then call set_output_space to
        set the output space of this layer.

        Parameters
        ----------
        space : Space
            Input space.
        """
        super(Ensemble, self).set_input_space(space)
        output_space = self.layers[0].get_output_space()
        for layer in self.layers:
            assert layer.get_output_space() == output_space
        self.set_output_space(output_space)

    def set_output_space(self, space):
        """
        Set the output space of this layer. This method is provided for
        subclasses that might not populate the same output space as their
        component layers.

        Parameters
        ----------
        space : Space
            Output space.
        """
        self.output_space = space

    def cost(self, Y, Y_hat):
        """
        Calculate the cost of predicting Y_hat when the true value is Y.

        Parameters
        ----------
        Y : theano.gof.Variable
            Target value(s).
        Y_hat : theano.gof.Variable
            Predicted value(s).
        """
        raise NotImplementedError('cost')


class EnsembleAverage(Ensemble):
    """
    Ensemble layer that calculates a (weighted) average of the output from
    the composite layer.

    Parameters
    ----------
    layer_name : str
        Name of this layer.
    layers : tuple or list
        The component layers to run in parallel.
    weights : list or None
        Contribution of each model in the weighted average.
    probabilities : bool
        Assume that component layer outputs are class probabilities.
    inputs_to_layers : dict or None
        Mapping for inputs to component layers.
    """
    def __init__(self, layer_name, layers, weights=None, probabilities=True,
                 inputs_to_layers=None):
        super(EnsembleAverage, self).__init__(layer_name, layers,
                                              probabilities, inputs_to_layers)
        if weights is None:
            weights = [1.0 for _ in layers]
        assert len(weights) == len(layers)
        self.weights = weights

    def fprop(self, state_below):
        """
        Average outputs from composite layer.

        Parameters
        ----------
        state_below : Space
            Batch of examples to propogate through each model.
        """
        components = super(EnsembleAverage, self).fprop(state_below)
        components = tuple(w * c
                           for w, c in safe_zip(self.weights, components))
        rval = T.sum(components, axis=0) / T.sum(self.weights)
        if self.probabilities:
            rval /= rval.sum(axis=1).dimshuffle(0, 'x')
        return rval

    def cost(self, Y, Y_hat):
        """
        Average of component layer costs.

        Parameters
        ----------
        Y : theano.gof.Variable
            Target value(s).
        Y_hat : theano.gof.Variable
            Predicted value(s).
        """
        rval = tuple(w * layer.cost(Y, Y_hat)
                     for w, layer in safe_zip(self.weights, self.layers))
        return rval
