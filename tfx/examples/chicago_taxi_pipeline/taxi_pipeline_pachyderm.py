# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ML example using Pachyderm TFX pipeline"""

import datetime
import os
from typing import List, Text
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.pachyderm.pachyderm_dag_runner import PachydermDagRunner
from tfx.orchestration.pachyderm.pachyderm_dag_runner import PachydermRunnerConfig
from tfx.orchestration.pachyderm.utils import pfs_external_input
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.version import __version__

import python_pachyderm


_pipeline_name = "ChicagoTaxiPachyderm"

_pipeline_spec_dir = os.path.join(os.getcwd(), "specs")

_local_test_root = os.path.join(os.getcwd(), "test")

_target_docker_image = "tfxpachyderm/chicago-taxi-example:{}".format(__version__)

_base_docker_image = "tensorflow/tfx:0.23.0"

_input_repo = python_pachyderm.PFSInput(repo="ChicagoTaxiPachyderm", branch="master")

_taxi_root = os.path.join(os.environ["HOME"], "taxi")

_module_file = os.path.join(os.sep, "src", "taxi_utils.py")

_serving_model_dir = os.path.join(_taxi_root, "serving_model", _pipeline_name)

# Pipeline arguments for Beam powered Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

def _create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    module_file: Text,
    serving_model_dir: Text,
    beam_pipeline_args: List[Text],
) -> pipeline.Pipeline:
  """Implements the chicago taxi pipeline with TFX."""
  # Pachyderm repo holding input data to be used
  input_repo = pfs_external_input(_input_repo)

  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = CsvExampleGen(input=input_repo)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])

  # Performs transformations and feature engineering in training and serving.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)

  # Uses user-provided Python function that implements a model using TF-Learn.
  trainer = Trainer(
      module_file=module_file,
      transformed_examples=transform.outputs['transformed_examples'],
      schema=schema_gen.outputs['schema'],
      transform_graph=transform.outputs['transform_graph'],
      train_args=trainer_pb2.TrainArgs(num_steps=10000),
      eval_args=trainer_pb2.EvalArgs(num_steps=5000))

  # Get the latest blessed model for model validation.
  # model_resolver = ResolverNode(
  #     instance_name='latest_blessed_model_resolver',
  #     resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
  #     model=Channel(type=Model, producer_component_id=trainer.id),
  #     model_blessing=Channel(type=ModelBlessing, producer_component_id=Evaluator.get_id()))

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(signature_name='eval')],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['trip_start_hour'])
      ],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'binary_accuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.6}),
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ])

  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=trainer.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=serving_model_dir)))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen, example_validator, transform,
          trainer, evaluator, pusher
      ],
      enable_cache=False,
      metadata_connection_config=metadata.mysql_uri_metadata_connection_config(
        uri=os.getenv("ML_METADATA_MYSQL_URI")
      ),
      beam_pipeline_args=beam_pipeline_args,
      additional_pipeline_args={
        "target_docker_image": _target_docker_image,
        "base_docker_image": _base_docker_image,
        "local_test_root": _local_test_root,
        },
      )

if __name__ == '__main__':
  container_secrets = [
    python_pachyderm.SecretMount(
      name="tfx-mysql", key="uri", env_var="ML_METADATA_MYSQL_URI"
    ),
  ]

  config = PachydermRunnerConfig(container_secrets=container_secrets)

  PachydermDagRunner(output_dir=_pipeline_spec_dir, config=config).run(
      _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_local_test_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        beam_pipeline_args=_beam_pipeline_args,
      )
    )
