MNI_TEMPLATE = "./assets/MNI152_T1_1mm_brain.nii"
MASK = "mask.nii"
DOWNSAMPLED_MNI = "mni_downsampled.nii"

import os

import nibabel as nib
import numpy as np
from nilearn.image import resample_img, resample_to_img


def get_downsampled_mni_and_mask(mask_info, clipped_img_affine, clipped_img_header, voxel_size):
    clipped_img = nib.Nifti1Image(mask_info, clipped_img_affine, clipped_img_header)

    # TODO check how to have these asset files at aggregator at startup
    reoriented_mni = resample_to_img(MNI_TEMPLATE, clipped_img, interpolation="linear", force_resample=True)
    downsampled_mni = resample_img(
        reoriented_mni, target_affine=np.eye(3) * voxel_size, interpolation="linear", force_resample=True
    )

    downsampled_mask = resample_to_img(
        clipped_img, downsampled_mni, interpolation="nearest", force_resample=True
    )
    return downsampled_mni, downsampled_mask, clipped_img


def saveBin(path, arr):
    with open(path, "wb+") as fh:
        header = "%s" % str(arr.dtype)
        for index in arr.shape:
            header += " %d" % index
        header += "\n"
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)


def loadBin(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)
