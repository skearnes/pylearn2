"""
Tests for training sklearn models in pylearn2.
"""
from pylearn2.config import yaml_parse
from pylearn2.testing.skip import skip_if_no_sklearn


#def test_train():
#    skip_if_no_sklearn()
#    trainer = yaml_parse.load(test_train_yaml)
#    trainer.main_loop()

def test_sklearn_model():
    skip_if_no_sklearn()
    trainer = yaml_parse.load(test_sklearn_model_yaml)
    trainer.main_loop()

test_train_yaml = """
!obj:pylearn2.sklearn.Train {
  dataset: &train
    !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
      rng: !obj:numpy.random.RandomState { seed: 1 },
      num_examples: 10,
      dim: 10,
      num_classes: 2,
    },
  model: !obj:sklearn.ensemble.RandomForestClassifier {
    n_estimators: 10,
  },
  metrics: [
    !!python/name:sklearn.metrics.roc_auc_score ,
  ],
  monitoring_dataset: { 'train': *train }
}
"""

test_sklearn_model_yaml = """
!obj:pylearn2.train.Train {
  dataset: &train
    !obj:pylearn2.testing.datasets.random_one_hot_dense_design_matrix {
      rng: !obj:numpy.random.RandomState { seed: 1 },
      num_examples: 10,
      dim: 5,
      num_classes: 2,
    },
  model: !obj:pylearn2.sklearn.SKLearnModel {
    model: !obj:sklearn.ensemble.RandomForestClassifier {
      n_estimators: 10,
    },
    input_dim: 5,
    output_dim: 2,
  },
  algorithm:
    !obj:pylearn2.sklearn.SKLearnTrainingAlgorithm {
      monitoring_dataset: *train,
    },
  extensions: [
    !obj:pylearn2.train_extensions.roc_auc.RocAucChannel {},
  ],
}
"""
