"""Serialize computation values for JSON-safe NVFlare transport."""

import base64
import binascii
import importlib
import io
import math
import os
import types
import typing
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any, Dict, Union, get_args, get_origin, get_type_hints

from .artifacts import ArtifactRef

DEFAULT_MAX_INLINE_ARRAY_BYTES = 8 * 1024 * 1024

_SERIALIZED_TYPE_KEY = "__neuroflame_type__"
_SERIALIZED_VALUE_KEY = "value"
_DATAFRAME_TAG = "pandas.DataFrame"
_NUMPY_ARRAY_TAG = "numpy.ndarray"
_NOT_TAGGED = object()
_NONE_TYPE = type(None)


class DataFrameSplitJsonCodec:
    """Encode pandas DataFrames with the split JSON orientation."""

    @staticmethod
    def encode(value):
        """Convert a DataFrame into split-orientation JSON data."""
        return value.to_dict(orient="split")

    @staticmethod
    def decode(value):
        """Reconstruct a DataFrame from split-orientation JSON data."""
        pandas = _require_module("pandas", "DataFrame deserialization")
        if not isinstance(value, dict) or not {"index", "columns", "data"}.issubset(
            value
        ):
            raise TypeError("Invalid split-format pandas DataFrame payload")
        return pandas.DataFrame(
            data=value["data"],
            index=value["index"],
            columns=value["columns"],
        )


class NumpyArrayCodec:
    """Encode bounded NumPy arrays with dtype and shape metadata."""

    @staticmethod
    def encode(value, max_inline_array_bytes=DEFAULT_MAX_INLINE_ARRAY_BYTES):
        """Convert a safe NumPy array into an inline base64 payload."""
        numpy = _require_module("numpy", "NumPy array serialization")
        _validate_inline_limit(max_inline_array_bytes)
        if not isinstance(value, numpy.ndarray):
            raise TypeError(f"Expected numpy.ndarray, received {type(value)!r}")
        _validate_numpy_dtype(value.dtype)
        _check_array_size(value.nbytes, max_inline_array_bytes)
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        }

    @staticmethod
    def decode(value, max_inline_array_bytes=DEFAULT_MAX_INLINE_ARRAY_BYTES):
        """Reconstruct a safe NumPy array from an inline base64 payload."""
        numpy = _require_module("numpy", "NumPy array deserialization")
        _validate_inline_limit(max_inline_array_bytes)
        if not isinstance(value, dict) or not {"dtype", "shape", "data"}.issubset(
            value
        ):
            raise TypeError("Invalid NumPy array payload")

        try:
            dtype = numpy.dtype(value["dtype"])
        except (TypeError, ValueError) as error:
            raise TypeError(f"Invalid NumPy dtype {value.get('dtype')!r}") from error
        _validate_numpy_dtype(dtype)

        shape = value["shape"]
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in shape
        ):
            raise TypeError(f"Invalid NumPy array shape {shape!r}")

        expected_bytes = math.prod(shape) * dtype.itemsize
        _check_array_size(expected_bytes, max_inline_array_bytes)
        encoded_data = value["data"]
        if not isinstance(encoded_data, str):
            raise TypeError("Invalid NumPy array data: expected a base64 string")
        try:
            raw_data = base64.b64decode(encoded_data.encode("ascii"), validate=True)
        except (UnicodeEncodeError, ValueError, binascii.Error) as error:
            raise TypeError("Invalid base64 data in NumPy array payload") from error
        if len(raw_data) != expected_bytes:
            raise ValueError(
                f"NumPy array payload contains {len(raw_data)} bytes; "
                f"shape and dtype require {expected_bytes}"
            )

        return numpy.frombuffer(raw_data, dtype=dtype).copy().reshape(tuple(shape))


def serialize_value(
    value: Any,
    codecs: Dict[type, Any] = None,
    *,
    max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES,
) -> Any:
    """Convert a supported computation value into JSON-safe transport data."""
    return _Serializer(codecs or {}, max_inline_array_bytes).serialize(value)


def deserialize_value(
    value: Any,
    expected_type: Any = None,
    codecs: Dict[type, Any] = None,
    *,
    max_inline_array_bytes: int = DEFAULT_MAX_INLINE_ARRAY_BYTES,
) -> Any:
    """Reconstruct a computation value from JSON-safe transport data."""
    return _Serializer(codecs or {}, max_inline_array_bytes).deserialize(
        value, expected_type
    )


class _Serializer:
    def __init__(self, codecs: Dict[type, Any], max_inline_array_bytes: int):
        _validate_inline_limit(max_inline_array_bytes)
        self.codecs = codecs
        self.max_inline_array_bytes = max_inline_array_bytes

    def serialize(self, value: Any, codec: Any = None) -> Any:
        if value is None:
            return None
        if isinstance(value, ArtifactRef):
            raise TypeError(
                "ArtifactRef values must be returned from a computation step so the "
                "framework can stage them before serialization"
            )
        if isinstance(value, (os.PathLike, io.IOBase)):
            raise TypeError(
                f"Files and paths ({type(value).__name__}) are not inline computation data. "
                "Return an ArtifactRef created with framework.artifact() instead."
            )
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Raw binary values are not inline computation data. Use a bounded NumPy "
                "array or ArtifactRef/file transfer support."
            )
        if codec is not None:
            return self.serialize(codec.encode(value))
        for registered_type, registered_codec in self.codecs.items():
            if isinstance(value, registered_type):
                return self.serialize(registered_codec.encode(value))
        if isinstance(value, (str, int, float, bool)):
            return value
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: self.serialize(
                    getattr(value, field.name),
                    field.metadata.get("codec"),
                )
                for field in fields(value)
            }
        if isinstance(value, dict):
            return {
                self.serialize(key): self.serialize(item) for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.serialize(item) for item in value]
        if _is_dataframe_value(value):
            return self._tagged(
                _DATAFRAME_TAG,
                DataFrameSplitJsonCodec.encode(value),
            )
        if _is_numpy_array_value(value):
            return self._tagged(
                _NUMPY_ARRAY_TAG,
                NumpyArrayCodec.encode(value, self.max_inline_array_bytes),
            )

        numpy_scalar = _as_numpy_scalar(value)
        if numpy_scalar is not _NOT_TAGGED:
            return numpy_scalar

        raise TypeError(
            f"Value of type {type(value)!r} is not JSON serializable by the framework"
        )

    def deserialize(
        self, value: Any, expected_type: Any = None, codec: Any = None
    ) -> Any:
        if value is None:
            return None
        if codec is not None:
            return codec.decode(value)
        if expected_type is None or expected_type is Any:
            return self._deserialize_untyped(value)
        if expected_type in self.codecs:
            return self.codecs[expected_type].decode(value)

        origin = get_origin(expected_type)
        args = get_args(expected_type)
        if _is_union_origin(origin):
            return self._deserialize_union(value, args)
        annotated_type = getattr(typing, "Annotated", None)
        if annotated_type is not None and origin is annotated_type:
            return self.deserialize(value, args[0] if args else Any)
        if _is_dataframe_type(expected_type):
            return DataFrameSplitJsonCodec.decode(
                self._standard_payload(value, _DATAFRAME_TAG)
            )
        if _is_numpy_array_type(expected_type):
            return NumpyArrayCodec.decode(
                self._standard_payload(value, _NUMPY_ARRAY_TAG),
                self.max_inline_array_bytes,
            )

        tagged_value = self._decode_tagged(value)
        if tagged_value is not _NOT_TAGGED:
            if isinstance(expected_type, type) and isinstance(
                tagged_value, expected_type
            ):
                return tagged_value
            raise TypeError(
                f"Serialized {type(tagged_value).__name__} value does not match "
                f"expected type {expected_type!r}"
            )

        if origin is list:
            item_type = args[0] if args else Any
            return [self.deserialize(item, item_type) for item in value]
        if origin is tuple:
            return self._deserialize_tuple(value, args)
        if origin is dict:
            key_type = args[0] if args else Any
            value_type = args[1] if len(args) > 1 else Any
            return {
                self.deserialize(key, key_type): self.deserialize(item, value_type)
                for key, item in value.items()
            }
        if origin in (set, frozenset):
            item_type = args[0] if args else Any
            result = {self.deserialize(item, item_type) for item in value}
            return frozenset(result) if origin is frozenset else result
        if origin is not None:
            return self._deserialize_untyped(value)

        if isinstance(expected_type, type) and is_dataclass(expected_type):
            if not isinstance(value, dict):
                raise TypeError(
                    f"Cannot deserialize {type(value).__name__} as dataclass "
                    f"{expected_type.__name__}"
                )
            type_hints = get_type_hints(expected_type)
            kwargs = {}
            for field in fields(expected_type):
                if not field.init or field.name not in value:
                    continue
                kwargs[field.name] = self.deserialize(
                    value[field.name],
                    type_hints.get(field.name),
                    field.metadata.get("codec"),
                )
            return expected_type(**kwargs)

        if isinstance(expected_type, type) and isinstance(value, expected_type):
            return value
        if isinstance(expected_type, type):
            return expected_type(value)
        return self._deserialize_untyped(value)

    def _tagged(self, type_name: str, value: Any) -> Dict[str, Any]:
        return {
            _SERIALIZED_TYPE_KEY: type_name,
            _SERIALIZED_VALUE_KEY: self.serialize(value),
        }

    def _deserialize_untyped(self, value: Any) -> Any:
        tagged_value = self._decode_tagged(value)
        if tagged_value is not _NOT_TAGGED:
            return tagged_value
        if isinstance(value, dict):
            return {
                self._deserialize_untyped(key): self._deserialize_untyped(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._deserialize_untyped(item) for item in value]
        return value

    def _decode_tagged(self, value: Any) -> Any:
        if not isinstance(value, dict):
            return _NOT_TAGGED
        type_name = value.get(_SERIALIZED_TYPE_KEY)
        if type_name not in (_DATAFRAME_TAG, _NUMPY_ARRAY_TAG):
            return _NOT_TAGGED
        if _SERIALIZED_VALUE_KEY not in value:
            raise TypeError(f"Serialized {type_name} value is missing its payload")
        payload = value[_SERIALIZED_VALUE_KEY]
        if type_name == _DATAFRAME_TAG:
            return DataFrameSplitJsonCodec.decode(payload)
        return NumpyArrayCodec.decode(payload, self.max_inline_array_bytes)

    def _standard_payload(self, value: Any, expected_tag: str) -> Any:
        if not isinstance(value, dict) or _SERIALIZED_TYPE_KEY not in value:
            return value
        actual_tag = value.get(_SERIALIZED_TYPE_KEY)
        if actual_tag != expected_tag:
            raise TypeError(
                f"Expected serialized {expected_tag}, received {actual_tag!r}"
            )
        if _SERIALIZED_VALUE_KEY not in value:
            raise TypeError(f"Serialized {expected_tag} value is missing its payload")
        return value[_SERIALIZED_VALUE_KEY]

    def _deserialize_union(self, value: Any, candidates) -> Any:
        if value is None and _NONE_TYPE in candidates:
            return None
        non_none_candidates = [
            candidate for candidate in candidates if candidate is not _NONE_TYPE
        ]
        ordered_candidates = sorted(
            non_none_candidates,
            key=lambda candidate: _wire_match_score(value, candidate),
            reverse=True,
        )
        errors = []
        for candidate in ordered_candidates:
            try:
                return self.deserialize(value, candidate)
            except (TypeError, ValueError, KeyError, AttributeError) as error:
                errors.append(f"{candidate!r}: {error}")
        details = "; ".join(errors)
        raise TypeError(f"Value cannot be deserialized as {candidates!r}: {details}")

    def _deserialize_tuple(self, value: Any, item_types) -> tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Cannot deserialize {type(value).__name__} as tuple")
        if not item_types:
            return tuple(self._deserialize_untyped(item) for item in value)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return tuple(self.deserialize(item, item_types[0]) for item in value)
        if len(value) != len(item_types):
            raise ValueError(
                f"Tuple payload has {len(value)} items; expected {len(item_types)}"
            )
        return tuple(
            self.deserialize(item, item_type)
            for item, item_type in zip(value, item_types, strict=False)
        )


def _is_union_origin(origin: Any) -> bool:
    union_type = getattr(types, "UnionType", None)
    return origin is Union or (union_type is not None and origin is union_type)


def _wire_match_score(value: Any, expected_type: Any) -> int:
    origin = get_origin(expected_type)
    if expected_type is Any:
        return 0
    if isinstance(expected_type, type) and type(value) is expected_type:
        return 3
    if origin in (list, tuple, set, frozenset) and isinstance(value, (list, tuple)):
        return 2
    if origin is dict and isinstance(value, dict):
        return 2
    if (
        isinstance(expected_type, type)
        and is_dataclass(expected_type)
        and isinstance(value, dict)
    ):
        return 2
    if _tag_name(value) == _DATAFRAME_TAG and _is_dataframe_type(expected_type):
        return 3
    if _tag_name(value) == _NUMPY_ARRAY_TAG and _is_numpy_array_type(expected_type):
        return 3
    if isinstance(expected_type, type) and isinstance(value, expected_type):
        return 1
    return 0


def _tag_name(value: Any):
    if isinstance(value, dict):
        return value.get(_SERIALIZED_TYPE_KEY)
    return None


def _validate_inline_limit(max_inline_array_bytes: int) -> None:
    if not isinstance(max_inline_array_bytes, int) or isinstance(
        max_inline_array_bytes, bool
    ):
        raise TypeError("max_inline_array_bytes must be an integer byte count")
    if max_inline_array_bytes < 0:
        raise ValueError("max_inline_array_bytes cannot be negative")


def _check_array_size(actual_bytes: int, max_inline_array_bytes: int) -> None:
    if actual_bytes <= max_inline_array_bytes:
        return
    raise ValueError(
        f"NumPy array is {actual_bytes} bytes, exceeding the inline limit of "
        f"{max_inline_array_bytes} bytes. Increase ComputationSpec.max_inline_array_bytes "
        "only for transport-safe arrays; large arrays require ArtifactRef/file transfer support."
    )


def _validate_numpy_dtype(dtype) -> None:
    if dtype.hasobject or dtype.fields:
        raise TypeError(
            f"NumPy dtype {dtype!s} is not safe for inline transport. Convert it to a "
            "numeric/string array or use ArtifactRef/file transfer support."
        )


@lru_cache(maxsize=None)
def _optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _require_module(module_name: str, purpose: str):
    module = _optional_module(module_name)
    if module is None:
        raise TypeError(f"{purpose} requires the optional '{module_name}' dependency")
    return module


def _is_dataframe_value(value: Any) -> bool:
    pandas = _optional_module("pandas")
    return pandas is not None and isinstance(value, pandas.DataFrame)


def _is_dataframe_type(expected_type: Any) -> bool:
    pandas = _optional_module("pandas")
    return (
        pandas is not None
        and isinstance(expected_type, type)
        and issubclass(expected_type, pandas.DataFrame)
    )


def _is_numpy_array_value(value: Any) -> bool:
    numpy = _optional_module("numpy")
    return numpy is not None and isinstance(value, numpy.ndarray)


def _is_numpy_array_type(expected_type: Any) -> bool:
    numpy = _optional_module("numpy")
    return (
        numpy is not None
        and isinstance(expected_type, type)
        and issubclass(expected_type, numpy.ndarray)
    )


def _as_numpy_scalar(value: Any) -> Any:
    numpy = _optional_module("numpy")
    if numpy is None:
        return _NOT_TAGGED
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, numpy.floating):
        return float(value)
    if isinstance(value, numpy.bool_):
        return bool(value)
    return _NOT_TAGGED
