"""Define author-facing file artifacts and transport metadata helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_ARTIFACT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_ARTIFACT_TOTAL_BYTES = 1024 * 1024 * 1024
DEFAULT_ARTIFACT_TIMEOUT = 300.0
DEFAULT_ARTIFACT_RETRIES = 2


@dataclass(frozen=True)
class ArtifactRef:
    """Reference one computation artifact without embedding its bytes."""

    name: str
    path: str
    media_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Reject names that could influence runtime-controlled paths."""
        if not isinstance(self.name, str) or not _ARTIFACT_NAME_PATTERN.fullmatch(
            self.name
        ):
            raise ValueError(
                "Artifact name must contain only letters, numbers, '.', '_', or '-' "
                "and cannot contain path separators"
            )
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("Artifact path must be a non-empty string")
        if self.media_type is not None and (
            not isinstance(self.media_type, str) or not self.media_type.strip()
        ):
            raise ValueError("Artifact media_type must be a non-empty string or None")


def artifact(name: str, path: str, media_type: str = None) -> ArtifactRef:
    """Declare a file artifact produced by a computation step."""
    return ArtifactRef(name=name, path=path, media_type=media_type)


def validate_artifact_name(name: str) -> None:
    """Validate a logical artifact name supplied over the transport boundary."""
    ArtifactRef(name=name, path="transport-placeholder")
