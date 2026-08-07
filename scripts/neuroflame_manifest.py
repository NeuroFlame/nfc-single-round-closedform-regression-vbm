"""Load and validate the repository-level NeuroFLAME computation manifest."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

MANIFEST_FILE = ".neuroflame.json"
MANIFEST_VERSION = 1
SEMVER = re.compile(r"\d+\.\d+\.\d+")
REQUIRED_IMAGE_KEYS = ("title", "repository", "floatingTag", "tagPrefix", "source")


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    """Return a manifest mapping or raise a field-specific validation error."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _require_semver(value: object, field: str) -> str:
    """Return a strict semantic version string from a manifest field."""
    if not isinstance(value, str) or not SEMVER.fullmatch(value):
        raise ValueError(f"{field} must use MAJOR.MINOR.PATCH")
    return value


def validate_manifest(raw: object) -> dict[str, Any]:
    """Validate and return a NeuroFLAME manifest."""
    manifest = _require_mapping(raw, "manifest")
    manifest_version = manifest.get("manifestVersion")
    if (
        not isinstance(manifest_version, int)
        or isinstance(manifest_version, bool)
        or manifest_version != MANIFEST_VERSION
    ):
        raise ValueError(f"manifestVersion must be {MANIFEST_VERSION}")

    computation = _require_mapping(manifest.get("computation"), "computation")
    _require_semver(computation.get("version"), "computation.version")

    compatibility = _require_mapping(manifest.get("compatibility"), "compatibility")
    _require_semver(
        compatibility.get("computationApiVersion"),
        "compatibility.computationApiVersion",
    )
    _require_semver(
        compatibility.get("boilerplateVersion"),
        "compatibility.boilerplateVersion",
    )

    image = _require_mapping(manifest.get("image"), "image")
    missing = [
        key for key in REQUIRED_IMAGE_KEYS if not isinstance(image.get(key), str)
    ]
    if missing:
        raise ValueError(f"Missing image configuration keys: {', '.join(missing)}")
    if any(not image[key].strip() for key in REQUIRED_IMAGE_KEYS if key != "tagPrefix"):
        raise ValueError("Image configuration values must not be empty")

    return manifest


def load_manifest(repository: Path) -> dict[str, Any]:
    """Load and validate a repository's NeuroFLAME manifest."""
    manifest_path = repository / MANIFEST_FILE
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Missing NeuroFLAME manifest: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {manifest_path}: {error}") from error
    return validate_manifest(raw)


def write_manifest(repository: Path, manifest: dict[str, Any]) -> None:
    """Validate and write a canonical NeuroFLAME manifest."""
    validated = validate_manifest(manifest)
    (repository / MANIFEST_FILE).write_text(
        json.dumps(validated, indent=2) + "\n",
        encoding="utf-8",
    )


def merge_compatibility(
    source_manifest: dict[str, Any], target_manifest: dict[str, Any]
) -> dict[str, Any]:
    """Apply framework-owned manifest fields while preserving author metadata."""
    source = validate_manifest(source_manifest)
    merged = deepcopy(validate_manifest(target_manifest))
    merged["manifestVersion"] = source["manifestVersion"]
    merged["compatibility"] = deepcopy(source["compatibility"])
    return merged
