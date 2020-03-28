# Lint as: python2, python3
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
"""Handler for Pachyderm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional, Text, Tuple

import click
import python_pachyderm
from tabulate import tabulate
import tensorflow as tf

from tfx.tools.cli import labels
from tfx.tools.cli.container_builder import builder
from tfx.tools.cli.container_builder import labels as container_builder_labels
from tfx.tools.cli.handler import base_handler
from tfx.utils import io_utils


class PachydermHandler(base_handler.BaseHandler):
  """Helper methods for Pachyderm Handler."""

  def __init__(self, flags_dict: Dict[Text, Any]):
    """Initialize Pachyderm handler.

    Args:
      flags_dict: A dictionary with flags provided in a command.
    """
    super(PachydermHandler, self).__init__(flags_dict)
    if labels.PACHD_ADDRESS in self.flags_dict:
      self._client = python_pachyderm.Client.new_from_pachd_address(
            pachd_address=self.flags_dict[labels.PACHD_ADDRESS])
    else:
      self._client = None

  def create_pipeline(self, update: bool = False) -> None:
    """Creates or updates a pipeline in Pachyderm.

    Args:
      update: set as true to update pipeline.
    """
    # Build pipeline container image.
    try:
      target_image = self.flags_dict.get(labels.TARGET_IMAGE)
      skaffold_cmd = self.flags_dict.get(labels.SKAFFOLD_CMD)
      if target_image is not None or os.path.exists(
          container_builder_labels.BUILD_SPEC_FILENAME):
        base_image = self.flags_dict.get(labels.BASE_IMAGE)
        target_image = self._build_pipeline_image(target_image, base_image,
                                                  skaffold_cmd)
    except (ValueError, subprocess.CalledProcessError, RuntimeError):
      click.echo('No container image is built.')
      raise
    else:
      click.echo('New container image is built. Target image is available in '
                 'the build spec file.')

    # Compile pipeline to check if pipeline_args are extracted successfully.
    pipeline_args = self.compile_pipeline()

    pipeline_name = pipeline_args[labels.PIPELINE_NAME]

    # self._save_pipeline(pipeline_args, update=update)

    if update:
      click.echo('Pipeline "{}" updated successfully.'.format(pipeline_name))
    else:
      click.echo('Pipeline "{}" created successfully.'.format(pipeline_name))

  def update_pipeline(self) -> None:
    """Updates pipeline in Pachyderm."""
    self.create_pipeline(update=True)

  def list_pipelines(self) -> None:
    """List all the pipelines in the environment."""
    if self._client is None:
        click.echo('Pachyderm client not connected')
        return

    pipelines = self._client.list_pipeline()
    tfx_stages = [ p.pipeline.name for p in pipelines.pipeline_info if 'TFX' in p.metadata.annotations ]
    tfx_pipelines = { p.split('_')[0]: [] for p in tfx_stages }
    for stage in tfx_stages:
      tfx_pipelines[stage.split('_')[0]].append(stage)

    if len(tfx_pipelines):
      for pipeline, stages in tfx_pipelines.items():
        click.echo(pipeline)
        for stage in stages: click.echo(f"-- {stage.split('_')[-1]}")
    else:
      click.echo('No pipelines to display.')

  def delete_pipeline(self) -> None:
    """Delete pipeline in Pachyderm."""

    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    # Check if pipeline exists on server.
    pipeline_id, experiment_id = self._get_pipeline_id_and_experiment_id(
        pipeline_name)
    self._client._pipelines_api.get_pipeline(pipeline_id)  # pylint: disable=protected-access

    # Delete pipeline for kfp server.
    self._client._pipelines_api.delete_pipeline(id=pipeline_id)  # pylint: disable=protected-access

    # Delete experiment from server.
    self._client._experiment_api.delete_experiment(experiment_id)  # pylint: disable=protected-access

    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name,
                                         '')

    # Delete pipeline for home directory.
    io_utils.delete_dir(handler_pipeline_path)

    click.echo('Pipeline ' + pipeline_name + ' deleted successfully.')

  def compile_pipeline(self) -> Dict[Text, Any]:
    """Compiles pipeline in Pachyderm.

    Returns:
      pipeline_args: python dictionary with pipeline details extracted from DSL.
    """
    self._check_pipeline_dsl_path()
    self._check_dsl_runner()
    pipeline_args = self._extract_pipeline_args()
    if not pipeline_args:
      sys.exit('Unable to compile pipeline. Check your pipeline dsl.')
    click.echo('Pipeline compiled successfully.')
    return pipeline_args

  def create_run(self) -> None:
    """Runs a pipeline in Pachyderm."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]
    experiment_name = pipeline_name

    pipeline_id, experiment_id = self._get_pipeline_id_and_experiment_id(
        pipeline_name)

    # Run pipeline.
    run = self._client.run_pipeline(
        experiment_id=experiment_id,
        job_name=experiment_name,
        pipeline_id=pipeline_id)

    click.echo('Run created for pipeline: ' + pipeline_name)
    self._print_runs([run])

  def delete_run(self) -> None:
    """Deletes a run."""
    self._client._run_api.delete_run(self.flags_dict[labels.RUN_ID])  # pylint: disable=protected-access

    click.echo('Run deleted.')

  def terminate_run(self) -> None:
    """Stops a run."""
    self._client._run_api.terminate_run(self.flags_dict[labels.RUN_ID])  # pylint: disable=protected-access
    click.echo('Run terminated.')

  def list_runs(self) -> None:
    """Lists all runs of a pipeline."""
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    pipeline_id, experiment_id = self._get_pipeline_id_and_experiment_id(
        pipeline_name)
    # Check if pipeline exists.
    self._client._pipelines_api.get_pipeline(pipeline_id)  # pylint: disable=protected-access

    # List runs.
    # TODO(jyzhao): use metadata context to get the run info.
    response = self._client.list_runs(experiment_id=experiment_id)

    if response and response.runs:
      self._print_runs(response.runs)
    else:
      click.echo('No runs found.')

  def get_run(self) -> None:
    """Checks run status."""

    run = self._client.get_run(self.flags_dict[labels.RUN_ID]).run
    self._print_runs([run])

  def _get_pipeline_specs_path(self) -> Text:
    """Gets pipeline spec path from either input flag or in specs/ where
      pipeline dsl is located
    """
    if self.flags_dict[labels.PIPELINE_SPEC_PATH]:
        return self.flags_dict[labels.PIPELINE_SPEC_PATH]

    return os.path.join(self.flags_dict[labels.PIPELINE_DSL_PATH], 'specs')

  def _save_pipeline(self,
                     pipeline_args: Dict[Text, Any],
                     update: bool = False) -> None:
    """Creates/updates pipeline folder in the handler directory."""
    pipeline_name = pipeline_args[labels.PIPELINE_NAME]

    if update:
      version_name = '{}_{}'.format(pipeline_name,
                                    time.strftime('%Y%m%d%H%M%S'))
      upload_response = self._client.pipeline_uploads.upload_pipeline_version(
          uploadfile=pipeline_package_path,
          name=version_name,
          pipelineid=pipeline_id)
    else:  # creating a new pipeline.
      upload_response = self._client.upload_pipeline(
          pipeline_package_path=pipeline_package_path,
          pipeline_name=pipeline_name)
      pipeline_id = upload_response.id

    # Display the link to the pipeline detail page in KFP UI.
    click.echo(upload_response)

    # Add pipeline details to pipeline_args.
    pipeline_args[labels.PIPELINE_NAME] = pipeline_name

    # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(handler_pipeline_path,
                                      'pipeline_args.json')

    # Copy pipeline_args to pipeline folder.
    tf.io.gfile.makedirs(handler_pipeline_path)
    with open(pipeline_args_path, 'w') as f:
      json.dump(pipeline_args, f)

  def _check_pipeline_package_path(self, pipeline_name: Text) -> None:

    # When unset, search for the workflow file in the current dir.
    if not self.flags_dict[labels.PIPELINE_PACKAGE_PATH]:
      self.flags_dict[labels.PIPELINE_PACKAGE_PATH] = os.path.join(
          os.getcwd(), '{}.tar.gz'.format(pipeline_name))

    pipeline_package_path = self.flags_dict[labels.PIPELINE_PACKAGE_PATH]
    if not tf.io.gfile.exists(pipeline_package_path):
      sys.exit(
          'Pipeline package not found at {}. When --package_path is unset, it will try to find the workflow file, "<pipeline_name>.tar.gz" in the current directory.'
          .format(pipeline_package_path))

  def _build_pipeline_image(self,
                            target_image: Optional[Text] = None,
                            base_image: Optional[Text] = None,
                            skaffold_cmd: Optional[Text] = None) -> None:
    return builder.ContainerBuilder(
        target_image=target_image,
        base_image=base_image,
        skaffold_cmd=skaffold_cmd).build()

  def _get_pipeline_id_and_experiment_id(
      self, pipeline_name: Text) -> Tuple[Text, Text]:
    # Path to pipeline folder.
    handler_pipeline_path = os.path.join(self._handler_home_dir, pipeline_name)

    # Check if pipeline exists.
    # self._check_pipeline_existence(pipeline_name)

    # Path to pipeline_args.json .
    pipeline_args_path = os.path.join(handler_pipeline_path,
                                      'pipeline_args.json')
    # Get pipeline_id/experiment_id from pipeline_args.json
    with open(pipeline_args_path, 'r') as f:
      pipeline_args = json.load(f)
    return (pipeline_args[labels.PIPELINE_ID],
            pipeline_args[labels.EXPERIMENT_ID])

  def _print_runs(self, runs):
    """Prints runs in a tabular format with headers mentioned below."""
    headers = ['pipeline_name', 'run_id', 'status', 'created_at', 'link']
    pipeline_name = self.flags_dict[labels.PIPELINE_NAME]

    def _get_run_details(run_id):
      """Return the link to the run detail page."""
      return '{prefix}/#/runs/details/{run_id}'.format(
          prefix=self._client._get_url_prefix(), run_id=run_id)  # pylint: disable=protected-access

    data = [
        [  # pylint: disable=g-complex-comprehension
            pipeline_name, run.id, run.status,
            run.created_at.isoformat(),
            _get_run_details(run.id)
        ] for run in runs
    ]
    click.echo(tabulate(data, headers=headers, tablefmt='grid'))
