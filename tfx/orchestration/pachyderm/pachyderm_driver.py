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
""""Pachyderm TFX driver base class"""

import os
import tensorflow as tf

from typing import Any, Dict, List, Text

from tfx.components.base import base_driver
from tfx.orchestration import data_types
from tfx.types import Artifact, Channel, channel_utils


def _generate_output_uri(
    artifact: Artifact, base_output_dir: Text
) -> Text:
    """Generate uri for output artifact."""
    uri = os.path.join(base_output_dir, artifact.split, "")

    tf.logging.info("Creating output artifact uri %s as directory", uri)
    tf.gfile.MakeDirs(uri)

    return uri


class PachydermDriver(base_driver.BaseDriver):
    """PachydermDriver is the base class of all custom drivers.

    This can also be used as the default driver of a component if no custom logic
    is needed.

    Attributes:
        _metadata_handler: An instance of Metadata.
    """

    def resolve_input_artifacts(
        self,
        input_dict: Dict[Text, Channel],
        exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
        pipeline_info: data_types.PipelineInfo,
    ) -> Dict[Text, List[Artifact]]:
        """Resolve input artifacts from metadata.

        Subclasses might override this function for customized artifact properties
        resolution logic. However please note that this function is supposed to be
        called in normal cases (except head of the pipeline) since it handles
        artifact info passing from upstream components.

        Args:
          input_dict: key -> Channel mapping for inputs generated in logical
            pipeline.
          exec_properties: Dict of other execution properties, e.g., configs.
          pipeline_info: An instance of data_types.PipelineInfo, holding pipeline
            related properties including component_type and component_id.

        Returns:
          Final execution properties that will be used in execution.

        Raises:
          RuntimeError: for Channels that do not contain any artifact. This will be
          reverted once we support Channel-based input resolution.
        """
        result = {}
        for name, input_channel in input_dict.items():
            artifacts = list(input_channel.get())
            if not artifacts:
                raise RuntimeError(
                    "Channel-based input resolution is not supported."
                )

            producer_commit = os.getenv("{}_{}_COMMIT".format(
                pipeline_info.pipeline_name, artifacts[0].producer_component))

            mat_artifacts = self._metadata_handler.search_artifacts(
                artifacts[0].name,
                pipeline_info.pipeline_name,
                producer_commit,
                artifacts[0].producer_component,
            )

            for artifact in mat_artifacts:
                pfs_repo = "{}_{}".format(
                        pipeline_info.pipeline_name, artifact.producer_component
                )
                uri = os.path.join(os.sep, "pfs", pfs_repo, artifact.split, "")
                artifact.uri = uri

            result[name] = mat_artifacts

        return result

    def _prepare_output_artifacts(
        self,
        output_dict: Dict[Text, Channel],
        execution_id: int,
        pipeline_info: data_types.PipelineInfo,
        component_info: data_types.ComponentInfo,
    ) -> Dict[Text, List[Artifact]]:
        """Prepare output artifacts by assigning uris to each artifact."""

        result = channel_utils.unwrap_channel_dict(output_dict)
        base_output_dir = os.path.join(os.sep, "pfs", "out")

        for _, output_list in result.items():
            for artifact in output_list:
                artifact.uri = _generate_output_uri(artifact, base_output_dir)

        return result
