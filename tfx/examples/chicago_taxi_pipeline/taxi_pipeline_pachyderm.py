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
from typing import Text
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import ModelValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.orchestration.pachyderm.pachyderm_dag_runner import PachydermDagRunner
from tfx.orchestration.pachyderm.pachyderm_dag_runner import PachydermRunnerConfig
from tfx.proto import evaluator_pb2
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2

import python_pachyderm

from tfx_pachyderm.utils import pfs_input

_pipeline_name = "ChicagoTaxiPachyderm"

_pipeline_spec_dir = os.path.join(os.getcwd(), "specs")

_local_test_root = os.path.join(os.getcwd(), "test")

_tfx_image = "tfxpachyderm/chicago-taxi-example:{}".format(_version.__version__)

_input_repo = python_pachyderm.PFSInput(repo="ChicagoTaxiPachyderm", branch="master")

_taxi_root = os.path.join(os.environ["HOME"], "taxi")

_module_file = os.path.join(os.sep, "src", "taxi_utils.py")

_serving_model_dir = os.path.join(_taxi_root, "serving_model", _pipeline_name)


def _create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    module_file: Text,
    serving_model_dir: Text,
) -> pipeline.Pipeline:
    """Implements the chicago taxi pipeline with TFX."""
    input_repo = pfs_input(_input_repo)

    # Brings data into the pipeline or otherwise joins/converts training data.
    example_gen = CsvExampleGen(input_base=input_repo)

    # Computes statistics over data for visualization and example validation.
    statistics_gen = StatisticsGen(input_data=example_gen.outputs.examples)

    # Generates schema based on statistics files.
    infer_schema = SchemaGen(stats=statistics_gen.outputs.output)

    # Performs anomaly detection based on statistics and data schema.
    validate_stats = ExampleValidator(
        stats=statistics_gen.outputs.output, schema=infer_schema.outputs.output
    )

    # Performs transformations and feature engineering in training and serving.
    transform = Transform(
        input_data=example_gen.outputs.examples,
        schema=infer_schema.outputs.output,
        module_file=module_file,
    )

    # Uses user-provided Python function that implements a model using TF-Learn.
    trainer = Trainer(
        module_file=module_file,
        transformed_examples=transform.outputs.transformed_examples,
        schema=infer_schema.outputs.output,
        transform_output=transform.outputs.transform_output,
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000),
    )

    # Uses TFMA to compute a evaluation statistics over features of a model.
    model_analyzer = Evaluator(
        examples=example_gen.outputs.examples,
        model_exports=trainer.outputs.output,
        feature_slicing_spec=evaluator_pb2.FeatureSlicingSpec(
            specs=[
                evaluator_pb2.SingleSlicingSpec(
                    column_for_slicing=["trip_start_hour"]
                )
            ]
        ),
    )

    # Performs quality validation of a candidate model (compared to a baseline).
    model_validator = ModelValidator(
        examples=example_gen.outputs.examples, model=trainer.outputs.output
    )

    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model_export=trainer.outputs.output,
        model_blessing=model_validator.outputs.blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            infer_schema,
            validate_stats,
            transform,
            trainer,
            model_analyzer,
            model_validator,
            pusher,
        ],
        enable_cache=False,
        metadata_connection_config=metadata.mysql_metadata_connection_config(
            host=os.getenv("ML_METADATA_MYSQL_HOST"),
            port=3306,
            username=os.getenv("ML_METADATA_MYSQL_USER"),
            password=os.getenv("ML_METADATA_MYSQL_PASSWORD"),
            database=os.getenv("ML_METADATA_MYSQL_DATABASE")
        ),
        additional_pipeline_args={
            "tfx_image": _tfx_image,
            "local_test_root": _local_test_root,
        }
    )

if __name__ == '__main__':
    container_secrets = [
        python_pachyderm.Secret(
            name="tfx-mysql", key="host", env_var="ML_METADATA_MYSQL_HOST"
        ),
        python_pachyderm.Secret(
            name="tfx-mysql", key="user", env_var="ML_METADATA_MYSQL_USER"
        ),
        python_pachyderm.Secret(
            name="tfx-mysql", key="password", env_var="ML_METADATA_MYSQL_PASSWORD"
        ),
        python_pachyderm.Secret(
            name="tfx-mysql", key="database", env_var="ML_METADATA_MYSQL_DATABASE"
        ),
    ]

    config = PachydermRunnerConfig(container_secrets=container_secrets)

    PachydermDagRunner(output_dir=_pipeline_spec_dir, config=config).run(
        _create_pipeline(
            pipeline_name=_pipeline_name,
            pipeline_root=_local_test_root,
            module_file=_module_file,
            serving_model_dir=_serving_model_dir,
        )
    )
