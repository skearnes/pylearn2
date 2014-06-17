"""
Test hyperparameter grid search.
"""
from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


def test_unique_parameter_sampler():
    """Test UniqueParameterSampler."""
    skip_if_no_sklearn()
    from pylearn2.grid_search.random_grid_search import UniqueParameterSampler
    grid = {'dim': [2, 4, 8], 'seed': [1, 2, 3]}
    sampler = UniqueParameterSampler(grid, 3, random_state=1)
    samples = list(sampler)
    assert len(samples) == 3
    assert samples[0] != samples[1] and samples[0] != samples[2]


def test_random_grid_search():
    """Test RandomGridSearch."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_random_grid_search_yaml)
    trainer.main_loop()
    print trainer.params
    print trainer.scores
    print trainer.best_params
    print trainer.best_scores


def test_random_grid_search_cv():
    """Test RandomGridSearchCV."""
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_random_grid_search_cv_yaml)
    trainer.main_loop()
    print trainer.params
    print trainer.scores
    print trainer.best_params
    print trainer.best_scores

test_random_grid_search_yaml = """
!obj:pylearn2.grid_search.random_grid_search.RandomGridSearch {
  n_iter: 3,
  random_state: 1,
  template: "
    !obj:pylearn2.train.Train {
      dataset: &train
      !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
          rng: !obj:numpy.random.RandomState { seed: %(seed)s },
          num_examples: 100,
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
        batch_size: 10,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
          !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 1,
          },
        monitoring_dataset: {
          train: *train,
        },
      },
      extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
          channel_name: train_objective,
          store_best_model: 1,
        },
      ],
    }",
  param_grid: {
    seed: [1, 2, 3],
    dim: [2, 4, 1],
  },
  monitor_channel: train_objective,
  n_best: 1,
}
"""

test_random_grid_search_cv_yaml = """
!obj:pylearn2.grid_search.random_grid_search.RandomGridSearchCV {
  n_iter: 3,
  random_state: 1,
  template: "
    !obj:pylearn2.cross_validation.TrainCV {
      dataset_iterator:
        !obj:pylearn2.cross_validation.dataset_iterators.DatasetKFold {
          dataset:
            !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
              rng: !obj:numpy.random.RandomState { seed: %(seed)s },
              num_examples: 100,
              dim: 10,
              num_classes: 2,
            },
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
        batch_size: 10,
        line_search_mode: 'exhaustive',
        conjugate: 1,
        termination_criterion:
          !obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs: 1,
          },
      },
    }",
  param_grid: {
    seed: [1, 2, 3],
    dim: [2, 4, 8],
  },
  monitor_channel: train_objective,
  n_best: 1,
  retrain: 1,
}
"""
