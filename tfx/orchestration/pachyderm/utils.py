import os

from python_pachyderm import PFSInput

from tfx import types
from tfx.types import channel_utils
from tfx.types.artifact import Artifact
from tfx.types.artifact import Property
from tfx.types.artifact import PropertyType
from tfx.types.standard_artifacts import ExternalArtifact

# PFS repo property
PFS_REPO_PROPERTY = Property(type=PropertyType.STRING)

# PFS branch property
PFS_BRANCH_PROPERTY = Property(type=PropertyType.STRING)

PFS_REPO = 'repo'
PFS_BRANCH = 'branch'

class PFSArtifact(Artifact):
    TYPE_NAME = 'PFSArtifact'
    PROPERTIES = {
        PFS_REPO: PFS_REPO_PROPERTY,
        PFS_BRANCH: PFS_BRANCH_PROPERTY
    }


def pfs_input(input_spec: PFSInput) -> types.Channel:
    """Creates an ExternalPath artifact based off a single
    PFSInput declaration.

    Args:
        input_spec: PFSInput message describing the input
            repo the pipeline starts from.

    Returns:
        input channel
    """
    artifact = PFSArtifact()
    artifact.repo = input_spec.repo
    artifact.branch = input_spec.branch

    artifact.uri = os.path.join(os.sep, "pfs", input_spec.repo)

    return channel_utils.as_channel([artifact])

def pfs_external_input(input_spec: PFSInput) -> types.Channel:
    """Creates an ExternalPath artifact based off a single
    PFSInput declaration.

    Args:
        input_spec: PFSInput message describing the input
            repo the pipeline starts from.

    Returns:
        input channel
    """
    artifact = ExternalArtifact()
    artifact.set_string_custom_property(PFS_REPO, input_spec.repo)
    artifact.set_string_custom_property(PFS_BRANCH, input_spec.branch)

    artifact.uri = os.path.join(os.sep, "pfs", input_spec.repo)

    return channel_utils.as_channel([artifact])
