"""
Tests for ensemble models.
"""
from pylearn2.config import yaml_parse


def test_ensemble_average():
    """Test EnsembleAverage."""
    trainer = yaml_parse.load(test_ensemble_average_yaml)
    trainer.main_loop()

test_ensemble_average_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train
        !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: 1 },
            num_examples: 100,
            dim: 10,
            num_classes: 3,
        },
    model: !obj:pylearn2.models.mlp.MLP {
        nvis: 10,
        layers: [
            !obj:pylearn2.models.ensemble.EnsembleAverage {
                layer_name: ensemble,
                layers: [
                    !obj:pylearn2.models.mlp.MLP {
                        nvis: 10,
                        layers: [
                            !obj:pylearn2.models.mlp.Sigmoid {
                                layer_name: h0,
                                dim: 20,
                                irange: 0.05,
                            },
                            !obj:pylearn2.models.mlp.Softmax {
                                layer_name: y,
                                n_classes: 3,
                                irange: 0.0,
                            },
                        ],
                    },
                    !obj:pylearn2.models.mlp.MLP {
                        nvis: 10,
                        layers: [
                            !obj:pylearn2.models.mlp.Linear {
                                layer_name: h0,
                                dim: 20,
                                irange: 0.05,
                            },
                            !obj:pylearn2.models.mlp.Softmax {
                                layer_name: y,
                                n_classes: 3,
                                irange: 0.0,
                            },
                        ],
                    },
                ],
            },
            #!obj:pylearn2.models.mlp.Softmax {
            #    layer_name: y,
            #    n_classes: 3,
            #    irange: 0.0,
            #},
        ],
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 10,
        line_search_mode: exhaustive,
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        monitoring_dataset: *train,
    },
}
"""
