import os

from python_pachyderm import PFSInput
from tfx.types import Artifact, Channel, channel_utils

PFS_REPO = "repo"
PFS_BRANCH = "branch"


def pfs_input(input_spec: PFSInput) -> Channel:
    """Creates an ExternalPath artifact based off a single
    PFSInput declaration.

    Args:
        input_spec: PFSInput message describing the input
            repo the pipeline starts from.

    Returns:
        input channel
    """
    artifact = Artifact(type_name="ExternalPath")

    artifact.set_string_custom_property(PFS_REPO, input_spec.repo)
    artifact.set_string_custom_property(PFS_BRANCH, input_spec.branch)

    artifact.uri = os.path.join(os.sep, "pfs", input_spec.repo)

    return channel_utils.as_channel([artifact])
