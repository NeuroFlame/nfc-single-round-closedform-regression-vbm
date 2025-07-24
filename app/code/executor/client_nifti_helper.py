import os

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import plotting


def average_nifti(covariates_X, data_path, transfer_dir: str):
    """Reads in all the nifti images and calculates their average"""
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
    """Read nifti files as matrices"""
    voxel_size = cache_dict["voxel_size"]
    try:
        try:
            mask_data = nib.load(mask_file_name).get_fdata()
            mask_dim = mask_data.shape
        except FileNotFoundError:
            raise Exception("Missing Mask file at " + mask_file_name)

    except:
        raise Exception("Missing Mask ")

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
        mask = nib.load(os.path.join(cache_dir, cache_dict['mask_file_name']))
    except FileNotFoundError:
        raise Exception(f"Cannot find mask file at: {cache_dict['mask_file_name']}")

    for column in beta_df:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_fdata() > 0] = beta_df[column]

        image_string = "beta_" + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(output_dir, image_string)

        nib.save(clipped_img, output_file + ".nii")

        plotting.plot_stat_map(
            clipped_img, output_file=output_file, display_mode="ortho", colorbar=True
        )


def print_pvals(cache_dict, cache_dir, output_dir, ps_global, ts_global, covar_labels):
    """Print regression coefficients' p-values as nifti files."""
    p_df = pd.DataFrame(ps_global, columns=covar_labels)
    t_df = pd.DataFrame(ts_global, columns=covar_labels)

    try:
        mask = nib.load(os.path.join(cache_dir, cache_dict['mask_file_name']))
    except FileNotFoundError:
        raise Exception(f"Cannot find mask file at: {cache_dict['mask_file_name']}")

    for column in p_df:
        new_data = np.zeros(mask.shape)
        new_data[mask.get_fdata() > 0] = (
                -1 * np.log10(p_df[column]) * np.sign(t_df[column])
        )

        image_string = "pval_" + str(column)

        clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
        output_file = os.path.join(output_dir, image_string)

        nib.save(clipped_img, output_file + ".nii")

        plotting.plot_stat_map(
            clipped_img, output_file=output_file, display_mode="ortho", colorbar=True
        )


def print_r2_image(cache_dict, cache_dir, output_dir, beta_df):
    """Print regression coefficients as nifti files."""
    try:
        mask = nib.load(os.path.join(cache_dir, cache_dict['mask_file_name']))
    except FileNotFoundError:
        raise Exception(f"Cannot find mask file at: {cache_dict['mask_file_name']}")

    new_data = np.zeros(mask.shape)
    new_data[mask.get_fdata() > 0] = beta_df

    image_string = "r_squared"

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
    output_file = os.path.join(output_dir, image_string)

    nib.save(clipped_img, output_file + ".nii")

    plotting.plot_stat_map(
        clipped_img, output_file=output_file, display_mode="ortho", colorbar=True
    )
