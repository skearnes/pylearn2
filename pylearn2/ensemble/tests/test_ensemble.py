"""
Tests for ensemble models.
"""
from pylearn2.config import yaml_parse


def test_train_ensemble_grid_search():
    """Test TrainEnsembleGridSearch."""
    trainer = yaml_parse.load(test_train_ensemble_grid_search_yaml)
    trainer.main_loop()


#def test_train_ensemble_grid_search_with_cv():
#    """Test TrainEnsembleGridSearch with cross-validation."""
#    trainer = yaml_parse.load(test_train_ensemble_grid_search_with_cv_yaml)
#    trainer.main_loop()


test_train_ensemble_grid_search_yaml = """
!obj:pylearn2.ensemble.GridSearchEnsemble {
    dataset: &train
        !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: 1 },
            num_examples: 10,
            dim: 10,
            num_classes: 2,
        },
    grid_search: !obj:pylearn2.grid_search.GridSearch {
      template: "
        !obj:pylearn2.train.Train {
          dataset: &train
          !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
            rng: !obj:numpy.random.RandomState { seed: 1 },
            num_examples: 10,
            dim: 10,
            num_classes: 2,
          },
          model: !obj:pylearn2.models.mlp.MLP {
            nvis: 10,
            layers: [
              !obj:pylearn2.models.mlp.Sigmoid {
                dim: %(dim)s,
                layer_name: h0,
                irange: 0.05,
              },
              !obj:pylearn2.models.mlp.Softmax {
                n_classes: 2,
                layer_name: y,
                irange: 0.0,
              },
            ],
          },
          algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
            batch_size: 5,
            line_search_mode: exhaustive,
            conjugate: 1,
            termination_criterion:
              !obj:pylearn2.termination_criteria.EpochCounter {
                max_epochs: 1,
              },
            monitoring_dataset: {
              train: *train,
            },
          },
        }",
      param_grid: {
        dim: [2, 4, 8]
      },
      monitor_channel: train_objective,
      n_best: 2,
    },
    ensemble: 'average',
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
        batch_size: 5,
        line_search_mode: exhaustive,
        conjugate: 1,
        termination_criterion:
            !obj:pylearn2.termination_criteria.EpochCounter {
                    max_epochs: 1,
        },
        monitoring_dataset: {
          train: *train,
        },
    },
}
"""
