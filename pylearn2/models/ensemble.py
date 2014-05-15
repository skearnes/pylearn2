"""
Ensemble models.

Most of the time these will be built on MLPs, which will have an fprop
method. Do we need to consider the case of an ensemble autoencoder, for
example?
"""
from pylearn2.models import Model


class Ensemble(Model):
    """
    Ensemble model that combines predictions from two or more models.
    Subclasses perform different combinations, possibly learning a meta-
    model.
    """
    def __init__(self, models, **kwargs):
        super(Ensemble, self).__init__(**kwargs)
        self.models = models

    def fprop(self, state):
        rval = [model.fprop(state) for model in self.models]
        return rval
