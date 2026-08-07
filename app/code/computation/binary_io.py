"""Read and write the historical VBM array file format."""

import os

import numpy as np


def save_binary(path, array) -> None:
    """Write an array using VBM's dtype-and-shape header format."""
    with open(path, "wb+") as binary_file:
        header = str(array.dtype)
        for dimension in array.shape:
            header += f" {dimension}"
        header += "\n"
        binary_file.write(header.encode())
        binary_file.write(array.data.tobytes())
        os.fsync(binary_file)


def load_binary(path):
    """Load an array written in VBM's historical binary format."""
    with open(path, "rb") as binary_file:
        header = binary_file.readline().decode().split()
        dtype = header.pop(0)
        dimensions = tuple(int(dimension) for dimension in header)
        return np.frombuffer(binary_file.read(), dtype=dtype).reshape(dimensions)
