"""Record terminal computation failures for container-level reporting."""

import json
import os
import traceback
from typing import Any, Dict, List

TERMINAL_ERROR_FILE_NAME = ".neuroflame_error.json"
TERMINAL_ERROR_MAX_BYTES = 64 * 1024
TERMINAL_ERROR_MAX_STAGE_LENGTH = 128
TERMINAL_ERROR_MAX_SCOPE_LENGTH = 512
TERMINAL_ERROR_MAX_TYPE_LENGTH = 256
TERMINAL_ERROR_MAX_MESSAGE_LENGTH = 4000
TERMINAL_ERROR_MAX_TRACEBACK_LENGTH = 12000
SHARED_ERROR_PREFIX = "NEUROFLAME_SHARED_ERROR:"
SHARED_ERROR_SCHEMA_VERSION = 1
SHARED_ERROR_CODE_SITE = "participant_computation_failed"
SHARED_ERROR_CODE_CENTRAL = "central_computation_failed"
_SHARED_ERROR_STAGES = {"startup", "execution", "aggregation", "transfer"}
_PUBLIC_STAGE_MAP = {
    "controller_startup": "startup",
    "input_validation": "startup",
    "controller_execution": "execution",
    "task_execution": "execution",
    "site_result": "transfer",
    "aggregation": "aggregation",
}
ERROR_ENVELOPE_KEY = "__neuroflame_error__"
ERROR_SCHEMA_VERSION = 1
ERROR_ORIGIN_SITE = "site"
ERROR_ORIGIN_CENTRAL = "central"
_VALID_ERROR_ORIGINS = {ERROR_ORIGIN_SITE, ERROR_ORIGIN_CENTRAL}


def _bounded_terminal_error_text(value: Any, maximum_length: int) -> str:
    """Return trimmed marker text within the version-one field contract."""
    return str(value).strip()[:maximum_length]


def _serialize_terminal_error(
    *, origin: str, stage: str, scope: str, error: Exception
) -> str:
    """Build a version-one marker that remains within the consumer byte limit."""
    error_type = _bounded_terminal_error_text(
        type(error).__name__, TERMINAL_ERROR_MAX_TYPE_LENGTH
    )
    message = _bounded_terminal_error_text(error, TERMINAL_ERROR_MAX_MESSAGE_LENGTH)
    marker = {
        "schema_version": ERROR_SCHEMA_VERSION,
        "origin": origin,
        "stage": _bounded_terminal_error_text(stage, TERMINAL_ERROR_MAX_STAGE_LENGTH),
        "scope": _bounded_terminal_error_text(scope, TERMINAL_ERROR_MAX_SCOPE_LENGTH),
        "error_type": error_type,
        "message": message or error_type,
        "traceback": _bounded_terminal_error_text(
            traceback.format_exc(), TERMINAL_ERROR_MAX_TRACEBACK_LENGTH
        ),
    }

    def serialize() -> str:
        return json.dumps(marker, indent=2, ensure_ascii=False)

    serialized = serialize()
    if len(serialized.encode("utf-8")) <= TERMINAL_ERROR_MAX_BYTES:
        return serialized

    traceback_text = marker["traceback"]
    low = 0
    high = len(traceback_text)
    while low < high:
        midpoint = (low + high + 1) // 2
        marker["traceback"] = traceback_text[:midpoint]
        candidate = serialize()
        if len(candidate.encode("utf-8")) <= TERMINAL_ERROR_MAX_BYTES:
            low = midpoint
        else:
            high = midpoint - 1
    marker["traceback"] = traceback_text[:low]
    return serialize()


def clear_terminal_error(output_dir: str) -> None:
    """Remove a terminal-error marker left by an earlier run."""
    error_path = os.path.join(output_dir, TERMINAL_ERROR_FILE_NAME)
    try:
        os.remove(error_path)
    except FileNotFoundError:
        pass


def build_error_envelope(origin: str, stage: str, scope: str) -> Dict[str, Any]:
    """Build safe failure provenance for transport through an NVFlare Shareable."""
    if origin not in _VALID_ERROR_ORIGINS:
        raise ValueError(f"Invalid error origin: {origin!r}")
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "origin": origin,
        "stage": stage,
        "scope": scope,
    }


def parse_error_envelope(value: Any) -> Dict[str, Any] | None:
    """Validate a transported NeuroFLAME error envelope."""
    if not isinstance(value, dict):
        return None
    if value.get("schema_version") != ERROR_SCHEMA_VERSION:
        return None
    if value.get("origin") not in _VALID_ERROR_ORIGINS:
        return None
    if not isinstance(value.get("stage"), str) or not value["stage"]:
        return None
    if not isinstance(value.get("scope"), str) or not value["scope"]:
        return None
    return {
        "schema_version": ERROR_SCHEMA_VERSION,
        "origin": value["origin"],
        "stage": value["stage"],
        "scope": value["scope"],
    }


def record_terminal_error(
    output_dir: str,
    scope: str,
    error: Exception,
    *,
    origin: str,
    stage: str,
) -> None:
    """Persist an exception and traceback without replacing the original error."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        error_path = os.path.join(output_dir, TERMINAL_ERROR_FILE_NAME)
        marker = _serialize_terminal_error(
            origin=origin,
            stage=stage,
            scope=scope,
            error=error,
        )
        with open(error_path, "w", encoding="utf-8") as error_file:
            error_file.write(marker)
    except Exception:
        # Error reporting must not replace the computation exception.
        pass


def find_terminal_errors(root_dir: str) -> List[Dict[str, Any]]:
    """Find and parse every terminal-error marker below a directory."""
    errors = []
    if not os.path.isdir(root_dir):
        return errors

    for directory, _subdirectories, file_names in os.walk(root_dir):
        if TERMINAL_ERROR_FILE_NAME not in file_names:
            continue
        error_path = os.path.join(directory, TERMINAL_ERROR_FILE_NAME)
        try:
            with open(error_path, encoding="utf-8") as error_file:
                error = json.load(error_file)
            if not isinstance(error, dict):
                raise TypeError("Terminal error marker must contain a JSON object")
        except Exception as read_error:
            error = {
                "scope": os.path.relpath(directory, root_dir),
                "error_type": type(read_error).__name__,
                "message": f"Could not read terminal error marker: {read_error}",
                "traceback": "",
            }
        error["path"] = error_path
        errors.append(error)

    return sorted(errors, key=lambda error: error["path"])


def raise_for_terminal_errors(root_dir: str) -> None:
    """Raise one combined error when terminal markers exist below a directory."""
    errors = find_terminal_errors(root_dir)
    if not errors:
        return

    details = []
    for error in errors:
        summary = (
            f"[{error.get('scope', 'computation')}] "
            f"{error.get('error_type', 'Error')}: {error.get('message', '')}"
        )
        error_traceback = error.get("traceback")
        details.append(f"{summary}\n{error_traceback}" if error_traceback else summary)
    raise RuntimeError("Terminal computation failure:\n" + "\n".join(details))


def build_shared_error_summary(origin: str, stage: str) -> Dict[str, Any]:
    """Build the only fixed-schema failure data accepted by orchestration."""
    if origin not in _VALID_ERROR_ORIGINS:
        raise ValueError(f"Invalid shared error origin: {origin!r}")
    if stage not in _SHARED_ERROR_STAGES:
        raise ValueError(f"Invalid shared error stage: {stage!r}")
    code = (
        SHARED_ERROR_CODE_SITE
        if origin == ERROR_ORIGIN_SITE
        else SHARED_ERROR_CODE_CENTRAL
    )
    return {
        "schema_version": SHARED_ERROR_SCHEMA_VERSION,
        "origin": origin,
        "stage": stage,
        "code": code,
    }


def emit_shared_error_summary(
    root_dir: str, *, fallback_origin: str, fallback_stage: str
) -> None:
    """Print an allowlisted envelope containing no computation-authored text."""
    errors = find_terminal_errors(root_dir)
    if not errors:
        origin = fallback_origin
        stage = fallback_stage
    else:
        error = errors[0]
        candidate_origin = error.get("origin")
        origin = (
            candidate_origin
            if candidate_origin in _VALID_ERROR_ORIGINS
            else fallback_origin
        )
        stage = _PUBLIC_STAGE_MAP.get(error.get("stage"), fallback_stage)
    summary = build_shared_error_summary(origin, stage)
    print(SHARED_ERROR_PREFIX + json.dumps(summary, sort_keys=True), flush=True)
