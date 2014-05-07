"""
HDF5 dataset tests.
"""
import numpy as np
import os
import tempfile

from pylearn2.config import yaml_parse
from pylearn2.testing.datasets import random_one_hot_dense_design_matrix
from pylearn2.testing.skip import skip_if_no_h5py


def test_hdf5_dataset():
    """Trains the model described in scripts/papers/maxout/mnist_pi.yaml
    using HDF5 datasets and a max_epochs termination criterion."""
    skip_if_no_h5py()
    import h5py

    # save random data to HDF5
    handle, filename = tempfile.mkstemp()
    dataset = random_one_hot_dense_design_matrix(np.random.RandomState(1),
                                                 num_examples=1000, dim=50,
                                                 num_classes=3)
    with h5py.File(filename, 'w') as f:
        f.create_dataset('X', data=dataset.get_design_matrix())
        f.create_dataset('y', data=dataset.get_targets())

    # instantiate Train object
    trainer = yaml_parse.load(trainer_yaml % {'filename': filename})
    trainer.main_loop()

    # cleanup
    os.remove(filename)

# trainer is a modified version of scripts/papers/maxout/mnist_pi.yaml
trainer_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.hdf5.HDF5Dataset {
        filename: %(filename)s,
        X: 'X',
        y: 'y',
    },
    model: !obj:pylearn2.models.mlp.MLP {
        layers: [
                 !obj:pylearn2.models.mlp.Sigmoid {
                     layer_name: 'h0',
                     dim: 25,
                     irange: .005,
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     layer_name: 'y',
                     n_classes: 3,
                     irange: 0.
                 }
                ],
        nvis: 50,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size: 100,
        learning_rate: .1,
        learning_rule:
            !obj:pylearn2.training_algorithms.learning_rule.Momentum {
                init_momentum: .5,
            },
        monitoring_dataset:
            {
                'train' : *train,
            },
        cost: !obj:pylearn2.costs.mlp.dropout.Dropout {
            input_include_probs: { 'h0' : .8 },
            input_scales: { 'h0': 1. }
        },
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
            },
    },
}
"""
