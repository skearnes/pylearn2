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

from pylearn2.models.mlp import (CompositeLayer, geometric_mean_prediction,
                                 MLP, Softmax)

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"


class Ensemble(CompositeLayer):
    """
    Subclass of CompositeLayer that does special handling of output,
    taking advantage of the assumed relationships between the parallel
    layers. This implies that the output classes and output spaces of each
    component layer should match.

    Parameters
    ----------
    layer_name : str
        Name of this layer.
    layers : tuple or list
        The component layers to run in parallel.
    inputs_to_layers : dict or None
        Mapping for inputs to component layers.
    """
    def __init__(self, layer_name, layers, inputs_to_layers=None):
        super(Ensemble, self).__init__(layer_name, layers, inputs_to_layers)

        # check that component layer output classes match
        if isinstance(layers[0], MLP):
            output_type = type(layers[0].layers[-1])
            for layer in layers[1:]:
                assert isinstance(layer.layers[-1], output_type)
        else:
            raise NotImplementedError('Ensemble layers must be MLPs.')
        self.output_type = output_type

        # check that component layer output spaces match
        output_space = layers[0].get_output_space()
        for layer in layers[1:]:
            assert layer.get_output_space() == output_space

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

    def fprop(self, state_below):
        """
        Transform input.

        Parameters
        ----------
        state_below : Space
            Batch of examples to propogate through each model.
        """
        raise NotImplementedError('fprop')

    def cost(self, Y, Y_hat):
        """
        Calculate the cost of predicting Y_hat when the true value is Y.
        Since the component layers are expected to have the same output
        class, the cost for the Ensemble is the same as the cost for one of
        the component layers.

        Parameters
        ----------
        Y : theano.gof.Variable
            Target value(s).
        Y_hat : theano.gof.Variable
            Predicted value(s).
        """
        rval = self.layers[0].cost(Y, Y_hat)
        return rval


class GeometricMean(Ensemble):
    """
    Ensemble layer that outputs the geometric mean of the outputs from the
    composite layer.

    Parameters
    ----------
    layer_name : str
        Name of this layer.
    layers : tuple or list
        The component layers to run in parallel.
    inputs_to_layers : dict or None
        Mapping for inputs to component layers.
    """
    def fprop(self, state_below):
        """
        Geometric mean of outputs from composite layer.

        Parameters
        ----------
        state_below : Space
            Batch of examples to propogate through each model.
        """
        components = super(GeometricMean, self).fprop(state_below)
        if issubclass(self.output_type, Softmax):
            rval = geometric_mean_prediction(components)
        else:
            rval = T.exp(T.mean(T.log(components), axis=0))
        return rval
