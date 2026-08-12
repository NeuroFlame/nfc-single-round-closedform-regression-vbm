"""Create and close computation-scoped file loggers."""

import logging
import os
from typing import Any, Dict

_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def create_computation_logger(
    output_dir: str,
    file_name: str,
    parameters: Dict[str, Any],
) -> logging.Logger:
    """Create a non-propagating logger that writes inside the output directory."""
    if not isinstance(file_name, str) or not file_name:
        raise TypeError("Computation log filename must be a non-empty string")
    if os.path.isabs(file_name) or os.path.basename(file_name) != file_name:
        raise ValueError(f"Computation log filename must be relative: {file_name!r}")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, file_name)
    level_name = parameters.get("log_level", os.getenv("LOG_LEVEL", "info"))
    if not isinstance(level_name, str):
        level_name = "info"
    level = _LOG_LEVELS.get(level_name.lower(), logging.INFO)

    logger = logging.Logger(f"neuroflame.computation.{file_name}", level=level)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def close_computation_logger(logger: logging.Logger) -> None:
    """Close and detach every handler owned by a computation logger."""
    for handler in tuple(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
