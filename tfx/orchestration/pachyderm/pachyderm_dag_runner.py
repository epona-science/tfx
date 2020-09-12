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
"""TFX runner for Pachyderm."""

import os
from typing import Any, List, Optional, Text

from python_pachyderm import Input
from python_pachyderm import PFSInput
from python_pachyderm import Secret

from tfx.orchestration import pipeline
from tfx.orchestration import tfx_runner
from tfx.orchestration import data_types
from tfx.orchestration.pachyderm import pachyderm_component
from tfx.orchestration.pachyderm import pachyderm_component_launcher
from tfx.types import Artifact


class PachydermRunnerConfig:
    """Runtime configuration parameters specific to execution on Pachyderm."""

    def __init__(self, container_secrets: Optional[List[Secret]] = None):
        """Creates a PachydermRunnerConfig object.

        Args:
          container_secrets: Kubernetes secrets to add to all component
            pachyderm specifications
        """
        self.container_secrets = container_secrets


def _create_pfs_input(artifact: Artifact) -> Input:
    """Constructs a PFSInput object from ExternalPath artifact

    Args:
        artifact: Artifact to extract from

    Returns:
        corresponding Input
    """
    try:
        artifact.get_string_custom_property('repo')
    except:
        raise ValueError(
            "Can't create PFSInput, artifact missing custom property 'repo'"
        )

    repo = artifact.get_string_custom_property("repo")
    branch = artifact.get_string_custom_property("branch")

    pfs_input = PFSInput(repo=repo, branch=branch or "master", glob="/")

    return Input(pfs=pfs_input)


class PachydermDagRunner(tfx_runner.TfxRunner):
    """Pachyderm pipline runner.

    Constructs a pipeline definition YAML file based on the TFX logical pipeline.
    """

    def __init__(
        self,
        output_dir: Optional[Text] = None,
        config: Optional[PachydermRunnerConfig] = None,
    ):
        """Initializes PachydermRunner for creating Pachyderm pipelines.

        Args:
          output_dir: An optional output directory into which to output the pipeline
            definition files. Defaults to the current working directory.

          config: An optional PachydermRunnerConfig object to specify runtime
            configuration when running the pipeline under Pachyderm.
        """
        self._output_dir = output_dir or os.getcwd()
        self._config = config or PachydermRunnerConfig()

    def _build_stages(self, pipeline: pipeline.Pipeline):
        """Constructs a Pachyderm Pipeline graph.

        Args:
          pipeline: The logical TFX pipeline to base the construction on.
        """
        docker_image = pipeline.additional_pipeline_args.get("target_docker_image")

        pipeline_properties = pachyderm_component.PipelineProperties(
            name=pipeline.pipeline_info.pipeline_name,
            docker_image=docker_image,
            spec_output_dir=self._output_dir,
            beam_pipeline_args=pipeline.beam_pipeline_args,
            container_secrets=self._config.container_secrets,
        )

        channel_pps_uris = {}
        stages = []

        for component in pipeline.components:
            stage = pachyderm_component.PachydermComponent(
                component, pipeline_properties
            )
            stages.append(stage)

            for _, channel in stage.output_channels():
                channel_pps_uris[channel] = stage.output_repo_input_spec

        for stage in stages:
            input_repos = []

            for input_name, input_channel in stage.input_channels():

                if input_channel in channel_pps_uris:
                    input_repos.append(channel_pps_uris[input_channel])

                elif input_channel.type_name == "ExternalArtifact":
                    artifacts = input_channel.get()
                    pfs_inputs = [_create_pfs_input(x) for x in artifacts]
                    input_repos += pfs_inputs

                else:
                    raise ValueError(
                        "Input channel {} provided to component {} is not produced"\
                        " by any prior stage nor is it an 'ExternalArtifact'".format(
                          input_name, stage.name)
                    )

            stage.pps_inputs = input_repos
            stage.write_specification()

    def run(self, pipeline: pipeline.Pipeline) -> Optional[Any]:
        """Compiles and outputs a Pachyderm Pipeline YAML definition file.

        Args:
          pipeline: The logical TFX pipeline to use when building the Pachyderm
            pipeline.
        """
        mode = os.getenv("TFX_PACH_MODE", "BUILD")

        if mode == "BUILD":

            self._build_stages(pipeline)

        elif mode == "RUN":

            component_id = os.getenv("TFX_PACH_COMPONENT_ID")
            components = list(
                filter(
                    lambda component: component.id == component_id,
                    pipeline.components,
                )
            )

            if not components:
                raise ValueError(
                    "No pipeline component for TFX_PACH_COMPONENT_ID of {}".format(
                        component_id
                    )
                )
            elif len(components) > 1:
                raise ValueError(
                    "Found {} components with TFX_PACH_COMPONENT_ID of {}".format(
                        len(components), component_id
                    )
                )

            component = components[0]

            pipeline_info = pipeline.pipeline_info
            pipeline_info.run_id = os.getenv("PACH_OUTPUT_COMMIT_ID")
            launcher = pachyderm_component_launcher.PachydermComponentLauncher(
                component,
                pipeline_info,
                data_types.DriverArgs(pipeline.enable_cache),
                pipeline.metadata_connection_config,
                pipeline.pipeline_args,
            )

            launcher.launch()
        else:

            raise ValueError(
                "Invalid TFX_PACH_MODE '{}'; expecting 'BUILD' or 'RUN'".format(
                    mode
                )
            )
