"""Write standard computation output types to site output directories."""

import json
import os
from collections.abc import Mapping
from typing import Any

from .serialization import serialize_value
from .types import RuntimeContext

_TEXT_EXTENSIONS = {".htm", ".html", ".md", ".txt"}


def write_standard_outputs(outputs: Mapping, runtime: RuntimeContext) -> None:
    """Write a filename-to-value mapping using the standard output formats."""
    if outputs is None:
        return
    if not isinstance(outputs, Mapping):
        raise TypeError(
            "site_output_step must return a filename-to-value mapping or None"
        )

    os.makedirs(runtime.output_dir, exist_ok=True)

    for file_name, value in outputs.items():
        output_path = _resolve_output_path(runtime.output_dir, file_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        extension = os.path.splitext(file_name)[1].lower()

        if extension == ".json":
            with open(output_path, "w", encoding="utf-8") as output_file:
                json.dump(
                    serialize_value(
                        value,
                        max_inline_array_bytes=runtime.max_inline_array_bytes,
                    ),
                    output_file,
                    indent=4,
                )
            continue

        if extension in (".csv", ".tsv"):
            to_csv = getattr(value, "to_csv", None)
            if not callable(to_csv):
                raise TypeError(
                    f"Output '{file_name}' requires a pandas DataFrame or another "
                    "value with a to_csv() method"
                )
            options = {"sep": "\t"} if extension == ".tsv" else {}
            to_csv(output_path, **options)
            continue

        if extension in _TEXT_EXTENSIONS:
            if not isinstance(value, str):
                raise TypeError(f"Output '{file_name}' requires text content")
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(value)
            continue

        raise ValueError(
            f"Output '{file_name}' has unsupported extension '{extension or '<none>'}'. "
            "Write specialized files directly from the site_output_step using output_dir."
        )


def _resolve_output_path(output_dir: str, file_name: Any) -> str:
    if not isinstance(file_name, str) or not file_name:
        raise TypeError("Output filenames must be non-empty strings")
    if os.path.isabs(file_name):
        raise ValueError(
            f"Output filename must be relative to output_dir: {file_name!r}"
        )

    output_root = os.path.abspath(output_dir)
    output_path = os.path.abspath(os.path.join(output_root, file_name))
    if os.path.commonpath((output_root, output_path)) != output_root:
        raise ValueError(f"Output filename escapes output_dir: {file_name!r}")
    return output_path
