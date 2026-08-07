"""NIfTI loading, masking, and output rendering for VBM regression."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.image import resample_to_img
from PIL import Image

# Fixed at 144 dpi -- higher dpi multiplies matplotlib's raster buffer size
# per image and has caused OOM kills in production when combined with fine
# VoxelSize and ShareLocalStats' per-site image fan-out. Not exposed as a
# computation parameter so it can't be reintroduced by accident.
_IMAGE_DPI = 144

# Figure size (inches) for ortho stat-map plots. At 144 dpi this yields
# ~1200x427px, which prints at 4.0in wide once retagged to _PUBLICATION_DPI
# below -- pick this before touching _IMAGE_DPI if you need a different
# publication print size, since raising render dpi risks the OOM this
# was tuned to avoid.
_FIGSIZE_INCHES = (8.34, 2.97)

# DPI metadata written to the saved PNG after render. This does not resample
# pixel data -- it only retags how the existing pixels are meant to print,
# so it's free (no added render time or memory).
_PUBLICATION_DPI = 300


def _new_figure():
    return plt.figure(figsize=_FIGSIZE_INCHES)


def _relabel_dpi(png_path):
    """Retag a PNG's DPI metadata to _PUBLICATION_DPI without resampling pixels."""
    with Image.open(png_path) as im:
        im.save(png_path, dpi=(_PUBLICATION_DPI, _PUBLICATION_DPI))


def average_nifti(covariates_X, data_path, transfer_dir: str):
    """Reads in all the nifti images and calculates their average."""
    appended_data = 0
    for image in covariates_X.index:
        image_data = nib.load(os.path.join(data_path, image)).dataobj[:]
        appended_data += image_data

    sample_image = nib.load(os.path.join(data_path, covariates_X.index[0]))
    avg_header, avg_affine = sample_image.header, sample_image.affine

    avg_nifti = appended_data / len(covariates_X.index)

    clipped_img = nib.Nifti1Image(avg_nifti, avg_affine, avg_header)
    output_file_name = os.path.join(transfer_dir, "avg_nifti.nii")
    nib.save(clipped_img, output_file_name)

    return output_file_name


def nifti_to_data(data_path, cache_dict, X, mask_file_name, downsampled_mni_file_name):
    """Read nifti files as matrices."""
    voxel_size = cache_dict["voxel_size"]
    try:
        mask_data = nib.load(mask_file_name).get_fdata()
        mask_dim = mask_data.shape
    except (FileNotFoundError, OSError) as error:
        raise ValueError("The transferred VBM mask is unavailable") from error

    y = np.zeros((len(X.index), np.count_nonzero(mask_data)), dtype="f8")
    for index, image in enumerate(X.index):
        input_file = os.path.join(data_path, image)
        if nib.load(input_file).header.get_zooms()[0] == voxel_size:
            image_data = nib.load(input_file).get_fdata()
        else:
            clipped_img = resample_to_img(input_file, downsampled_mni_file_name)
            image_data = clipped_img.get_fdata()

        a = []
        for slicer in range(mask_dim[-1]):
            img_slice = image_data[slicer, ...]
            msk_slice = mask_data[slicer, ...]
            a.extend(img_slice[msk_slice > 0].tolist())

        y[index, :] = a

    return y


def print_beta_images(cache_dict, cache_dir, output_dir, avg_beta_vector, covar_labels):
    """Print regression coefficients as nifti files."""
    beta_df = pd.DataFrame(avg_beta_vector, columns=covar_labels)

    try:
        mask = nib.load(os.path.join(cache_dir, cache_dict["mask_file_name"]))
    except FileNotFoundError as error:
        raise ValueError("The local VBM mask is unavailable") from error

    for column in beta_df:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_fdata() > 0] = beta_df[column]

        image_string = "beta_" + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(output_dir, image_string)

        nib.save(clipped_img, output_file + ".nii")

        display = plotting.plot_stat_map(
            clipped_img, display_mode="ortho", colorbar=True, figure=_new_figure()
        )
        display.savefig(output_file, dpi=_IMAGE_DPI)
        display.close()
        _relabel_dpi(output_file + ".png")


def print_pvals(cache_dict, cache_dir, output_dir, ps_global, ts_global, covar_labels):
    """Print regression coefficients' signed significance (-log10(p) * sign(t)) as nifti files."""
    p_df = pd.DataFrame(ps_global, columns=covar_labels)
    t_df = pd.DataFrame(ts_global, columns=covar_labels)

    try:
        mask = nib.load(os.path.join(cache_dir, cache_dict["mask_file_name"]))
    except FileNotFoundError as error:
        raise ValueError("The local VBM mask is unavailable") from error

    for column in p_df:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_fdata() > 0] = (
            -1 * np.log10(p_df[column]) * np.sign(t_df[column])
        )

        image_string = "signed_significance_" + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(output_dir, image_string)

        nib.save(clipped_img, output_file + ".nii")

        display = plotting.plot_stat_map(
            clipped_img, display_mode="ortho", colorbar=True, figure=_new_figure()
        )
        display.savefig(output_file, dpi=_IMAGE_DPI)
        display.close()
        _relabel_dpi(output_file + ".png")


def print_r2_image(cache_dict, cache_dir, output_dir, beta_df):
    """Print regression coefficients as nifti files."""
    try:
        mask = nib.load(os.path.join(cache_dir, cache_dict["mask_file_name"]))
    except FileNotFoundError as error:
        raise ValueError("The local VBM mask is unavailable") from error

    new_data = np.zeros(mask.shape)
    new_data[mask.get_fdata() > 0] = beta_df

    image_string = "r_squared"

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
    output_file = os.path.join(output_dir, image_string)

    nib.save(clipped_img, output_file + ".nii")

    display = plotting.plot_stat_map(
        clipped_img, display_mode="ortho", colorbar=True, figure=_new_figure()
    )
    display.savefig(output_file, dpi=_IMAGE_DPI)
    display.close()
    _relabel_dpi(output_file + ".png")
