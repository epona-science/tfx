# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pachyderm implementation of TFX components.
"""

import os
from typing import Iterable, List, Optional, Text, Tuple

from google.protobuf.json_format import MessageToJson
import python_pachyderm as pachyderm
from tfx.components.base import base_component
from tfx.types import Channel


_BASE_IMAGE = "tfxpachyderm/tfx-pachyderm"
_COMMAND = ["python", "/src/pipeline_pachyderm.py"]


class PipelineProperties:
    """Holds pipeline level execution properties that apply to all component."""

    def __init__(
        self,
        name: Text,
        docker_image: Optional[Text] = None,
        spec_output_dir: Optional[Text] = None,
        beam_pipeline_args: Optional[Text] = None,
        container_secrets: Optional[List[pachyderm.Secret]] = None,
    ):
        self.name = name
        self.image = docker_image or _BASE_IMAGE
        self.spec_output_dir = spec_output_dir or os.getcwd()
        self.exec_properties = {}
        self.container_secrets = container_secrets

        os.makedirs(self.spec_output_dir, exist_ok=True)

        if beam_pipeline_args:
            self.exec_properties["beam_pipeline_args"] = beam_pipeline_args


class PachydermComponent:
    """Base component for all Pachyderm pipelines TFX components.
    """

    def __init__(
        self,
        component: base_component.BaseComponent,
        pipeline_properties: PipelineProperties,
    ):
        """Creates a new component.

        Args:
          component: TFX component name.
          input_dict: Dictionary of input names to TFX types, or
            kfp.dsl.PipelineParam representing input parameters.
          output_dict: Dictionary of output names to List of TFX types.
          exec_properties: Execution properties.
          executor_class_path: <module>.<class> for Python class of executor.
          pipeline_properties: Pipeline level properties shared by all components.

        Returns:
          Newly constructed TFX Pachyderm component instance.
        """
        self.component = component
        self.pipeline_properties = pipeline_properties
        self._pps_inputs = []

        self.name = "{}_{}".format(
            self.pipeline_properties.name, self.component.id
        )

        self.description = "{} component for {} TFX pipeline".format(
            self.component.id, self.pipeline_properties.name
        )

        self._pipeline_spec = self._build_pipeline_spec()
        self.output_repo_input_spec = self._pps_output_repo_input_spec()

    def _build_pipeline_spec(self) -> pachyderm.CreatePipelineRequest:
        """Generates the pachyderm pipeline specification for this component

        Returns:
            pachyderm pipeline specification
        """
        input_spec = None
        if len(self._pps_inputs) == 1:
            input_spec = self._pps_inputs[0]
        elif len(self._pps_inputs) > 1:
            input_spec = pachyderm.Input(cross=self._pps_inputs)

        spec = pachyderm.CreatePipelineRequest(
            pipeline=pachyderm.Pipeline(name=self.name),
            description=self.description,
            transform=pachyderm.Transform(
                cmd=_COMMAND,
                image=self.pipeline_properties.image,
                env={
                    "TFX_PACH_MODE": "RUN",
                    "TFX_PACH_COMPONENT_ID": self.component.id,
                },
                secrets=self.pipeline_properties.container_secrets
            ),
            input=input_spec,
        )

        return spec

    def _pps_output_repo_input_spec(self) -> pachyderm.Input:
        """Generates the pachyderm Input specification of this components
        output repo so that proceeding stages can attach to it.

        Returns:
            output repo Input spec
        """
        pfs_input = pachyderm.PFSInput(
            repo=self.name, branch="master", glob="/"
        )

        return pachyderm.Input(pfs=pfs_input)

    @property
    def pps_inputs(self) -> List[pachyderm.Input]:
        return self._pps_inputs

    @pps_inputs.setter
    def pps_inputs(self, value: List[pachyderm.Input]):
        if len(value) == 0:
            raise ValueError("PPS Inputs cannot be empty")

        self._pps_inputs = value
        self._pipeline_spec = self._build_pipeline_spec()

    def input_channels(self) -> Iterable[Tuple[Text, Channel]]:
        """Fetches iterable of tuple pairs containing input name
        and channel for the wrapped component

        Returns:
            Iterable of name/channel pairs
        """
        return self.component.inputs.get_all().items()

    def output_channels(self) -> Iterable[Tuple[Text, Channel]]:
        """Fetches iterable of tuple pairs containing output name
        and channel for the wrapped component

        Returns:
            Iterable of name/channel pairs
        """
        return self.component.outputs.get_all().items()

    def write_specification(self):
        jsonSpec = MessageToJson(self._pipeline_spec)
        output_file = os.path.join(
            self.pipeline_properties.spec_output_dir, "{}.json".format(self.name)
        )

        with open(output_file, "w") as spec:
            spec.write(jsonSpec)
