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
"""For component execution, includes driver, executor and publisher."""

from typing import Any, Dict, Text

from ml_metadata.proto import metadata_store_pb2
from tfx.components.base import base_component
from tfx.components.base import base_driver
from tfx.orchestration.launcher import base_component_launcher
from tfx.orchestration.pachyderm.pachyderm_driver import PachydermDriver
from tfx.orchestration import data_types


class PachydermComponentLauncher(base_component_launcher.BaseComponentLauncher):
    """Responsible for launching driver, executor and publisher of component."""

    def __init__(
        self,
        component: base_component.BaseComponent,
        pipeline_info: data_types.PipelineInfo,
        driver_args: data_types.DriverArgs,
        metadata_connection_config: metadata_store_pb2.ConnectionConfig,
        additional_pipeline_args: Dict[Text, Any],
    ):
        """Initialize a ComponentLauncher.

        Args:
          component: Component that to be executed.
          pipeline_info: An instance of data_types.PipelineInfo that holds pipeline
            properties.
          driver_args: An instance of data_types.DriverArgs that holds component
            specific driver args.
          metadata_connection_config: ML metadata connection config.
          additional_pipeline_args: Additional pipeline args, includes,
            - beam_pipeline_args: Beam pipeline args for beam jobs within executor.
              Executor will use beam DirectRunner as Default.
        """
        super().__init__(
            component=component,
            pipeline_info=pipeline_info,
            driver_args=driver_args,
            metadata_connection_config=metadata_connection_config,
            additional_pipeline_args=additional_pipeline_args,
        )

        if component.driver_class == base_driver.BaseDriver:
            self._driver_class = PachydermDriver
        else:
            self._driver_class = type(
                "Driver", (component.driver_class, PachydermDriver), {}
            )
