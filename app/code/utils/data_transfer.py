import os
import numpy as np
import logging

def file_to_bytes(file_path: str) -> bytes:
    """Reads any file as bytes."""
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return b""  # Return empty bytes to avoid crashes

def bytes_to_file(image_file_path: str, file_data: bytes):
    # Save the received image bytes to a file
    with open(image_file_path, "wb") as f:
        f.write(file_data)




def save_binary(path, arr):
    with open(path, "wb+") as fh:
        header = "%s" % str(arr.dtype)
        for index in arr.shape:
            header += " %d" % index
        header += "\n"
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)


def load_binary(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)


def compress(transfer_dir):
    import os
    import zipfile

    zf = zipfile.ZipFile(f"{transfer_dir}.zip", "w")
    for dirname, subdirs, files in os.walk(transfer_dir):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    #zf.close()
    return zf

def save_compressed_file(transfer_dir):
    #TODO code this correctly
    import os
    import zipfile

    zf = zipfile.ZipFile(f"{transfer_dir}.zip", "w")
    for dirname, subdirs, files in os.walk(transfer_dir):
        zf.write(dirname)
        for filename in files:
            zf.write(os.path.join(dirname, filename))
    #zf.close()
    return zf
