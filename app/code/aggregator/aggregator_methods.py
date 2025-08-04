import os

import jsonpickle
import numpy as np
import pandas as pd
from utils.data_transfer import *

# from ..utils.data_transfer import *

MNI_TEMPLATE = "/workspace/app/code/assets/MNI152_T1_1mm_brain.nii"


def perform_remote_step1_generate_average_mask(site_results, agg_cache_dict, agg_cache_dir):
    """
    Computes the average of all nifti files to generate a mask and determines all the categorical covariates across all
    the clients.

    :param site_results: output sent from clients
    :param agg_cache_dict: cache maintained by the aggregator
    :return: output to be sent to the clients along with the updated cache data
    """
    def return_uniques_and_counts(df):
        """Return unique-values of the categorical variables and their counts"""
        keys, count = dict(), dict()
        keys = (
            df.iloc[:, :].sum(axis=1).apply(set).apply(sorted).to_dict()
        )  # adding all columns
        count = {k: len(v) for k, v in keys.items()}

        return keys, count

    """The first function in the remote computation chain"""
    output_mask_file_name, output_downsampled_mni_filename = _calculate_mask(site_results, agg_cache_dir)
    site_info = {site: site_results[site]["categorical_dict"] for site in site_results.keys()}
    ref_cols = {site: site_results[site]["reference_columns"] for site in site_results.keys()}

    reference_dict = next(iter(ref_cols.values()))
    # assert that reference columns/values for dummy encoding are same across all the sites for
    # the correctness of the decentralized regression. This is always true for coinstac GUI computation.
    assert all(value == reference_dict for value in ref_cols.values()), \
        "Reference values for dummy encoding are not same across all the sites " + str(ref_cols)

    df = pd.DataFrame.from_dict(site_info)
    covar_keys, unique_count = return_uniques_and_counts(df)

    results = {
        "output": {
            "covar_keys": jsonpickle.encode(covar_keys, unpicklable=False),
            "reference_columns": reference_dict,
            "global_unique_count": unique_count
        },
        "cache": {},
        "transfer_files": {"mask_file": output_mask_file_name,
                           "downsampled_mni": output_downsampled_mni_filename}
    }

    return results


def perform_remote_step2_compute_global_model(site_results, agg_cache_dict, agg_cache_dir):
    """
    Computes the global regression beta weights based on the client model parameters and sends it to the clients.

    :param site_results: output sent from clients
    :param agg_cache_dict: cache maintained by the aggregator
    :return: output to be sent to the clients along with the updated cache data
    """
    """The second function in the local computation chain"""
    site_ids = sorted(site_results.keys())
    user_id = site_ids[0]

    X_labels = site_results[user_id]["X_labels"]

    # all_local_stats_dicts = [
    #    site_results[site]["local_stats_list"] for site in site_results
    # ]

    beta_vector_0 = sum([load_binary(site_results[site]["x_transpose_x_file_name"]) for site in site_ids])
    beta_vector_1 = sum([load_binary(site_results[site]["x_transpose_y_file_name"]) for site in site_ids])

    all_lambdas = [site_results[site]["lambda"] for site in site_results]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    try:
        # Add lambda's
        beta_vector_0 = beta_vector_0 + np.unique(all_lambdas) * np.eye(beta_vector_0.shape[0])
        avg_beta_vector = np.transpose(np.dot(np.linalg.inv(beta_vector_0), beta_vector_1))
    except np.linalg.LinAlgError:
        cond = np.linalg.cond(beta_vector_0);
        raise Exception(f"X.^T*X matrix at remote is Singular with condition number: {cond}")

    mean_y_local = [site_results[site]["mean_y_local"] for site in site_results]
    count_y_local = [np.array(site_results[site]["count_local"]) for site in site_results]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.sum(mean_y_global, axis=0) / np.sum(count_y_local, axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    avg_beta_vector_file_name = os.path.join(agg_cache_dir, "avg_beta_vector.npy")
    save_binary(avg_beta_vector_file_name, avg_beta_vector)

    mean_y_global_file_name = os.path.join(agg_cache_dir, "mean_y_global.npy")
    save_binary(mean_y_global_file_name, mean_y_global)

    output_dict = {}

    cache_dict = {
        "avg_beta_vector": "avg_beta_vector.npy",
        "dof_global": dof_global.tolist(),
        "X_labels": X_labels,
        # "local_stats_dict": all_local_stats_dicts,
    }
    results = {'output': {}, 'cache': cache_dict,
               "transfer_files": {"avg_beta_vector": avg_beta_vector_file_name,
                                  "mean_y_global": mean_y_global_file_name}
               }

    return results


def perform_remote_step3_aggregate_metrics(site_results, agg_cache_dict, agg_cache_dir):
    """
    Aggregates all the local performance metrics and computes global metrics.

    :param site_results: output sent from clients
    :param agg_cache_dict: cache maintained by the aggregator
    :return: output to be sent to the clients along with the updated cache data
    """

    from scipy import stats
    from numba import jit, prange
    @jit(nopython=True)
    def remote_stats(MSE, varX_matrix_global, avg_beta_vector):
        my_shape = avg_beta_vector.shape
        ts = np.zeros(my_shape)

        for voxel in prange(my_shape[0]):
            var_covar_beta_global = MSE[voxel] * np.linalg.inv(varX_matrix_global)
            se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
            ts[voxel, :] = avg_beta_vector[voxel, :] / se_beta_global

        return ts

    X_labels = agg_cache_dict["X_labels"]

    avg_beta_vector = load_binary(os.path.join(agg_cache_dir, agg_cache_dict["avg_beta_vector"]))
    dof_global = agg_cache_dict["dof_global"]

    SSE_global = sum([np.array(site_results[site]["SSE_local"]) for site in site_results])
    SST_global = sum([np.array(site_results[site]["SST_local"]) for site in site_results])
    varX_matrix_global = sum(
        [np.array(site_results[site]["varX_matrix_local"]) for site in site_results]
    )

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)
    ts_global = remote_stats(MSE, varX_matrix_global, avg_beta_vector)
    ps_global = [2 * stats.t.sf(np.abs(t), df) for t, df in zip(ts_global, dof_global)]

    output_dict = {
        "r_squared_global": r_squared_global.tolist(),
        "MSE": MSE.tolist(),
        "ts_global": ts_global.tolist(),
        "ps_global": ps_global,
        "X_labels": X_labels
    }
    results = {"output": output_dict, "cache": {}}

    return results


def _calculate_mask(site_results, agg_cache_dir):
    import nibabel as nib
    from nilearn.image import resample_img, resample_to_img

    """Calculates the average of all masks"""
    num_sites = len(site_results.keys())
    site_ids = sorted(site_results.keys())
    avg_of_all = None
    for site, results in site_results.items():
        if avg_of_all is not None:
            avg_of_all += np.asarray(nib.load(results["transfer_average_nifti"]).get_fdata())
        else:
            avg_of_all = np.asarray(nib.load(results["transfer_average_nifti"]).get_fdata())

    avg_of_all = avg_of_all / num_sites

    # Threshold binarizer
    user_id = site_ids[0]
    threshold = site_results[user_id]["threshold"]
    voxel_size = site_results[user_id]["voxel_size"]

    mask_info = avg_of_all > threshold

    user_id = site_ids[0]
    principal_image = nib.load(site_results[user_id]["transfer_average_nifti"])
    header, affine = principal_image.header, principal_image.affine

    # downsampled_mni, downsampled_mask, clipped_img = ch.get_downsampled_mni_and_mask(mask_info, clipped_img_affine, clipped_img_header, voxel_size)

    clipped_img = nib.Nifti1Image(mask_info, affine, header)

    reoriented_mni = resample_to_img(MNI_TEMPLATE, clipped_img, interpolation="linear")
    downsampled_mni = resample_img(
        reoriented_mni, target_affine=np.eye(3) * voxel_size, interpolation="linear"
    )

    downsampled_mask = resample_to_img(
        clipped_img, downsampled_mni, interpolation="nearest"
    )

    # TODO check if this works fine when sent to executors
    output_mask_file_name = os.path.join(agg_cache_dir, "mask.nii")
    output_downsampled_mni_filename = os.path.join(agg_cache_dir, "mni_downsampled.nii")
    nib.save(downsampled_mask, output_mask_file_name)
    nib.save(downsampled_mni, output_downsampled_mni_filename)

    return output_mask_file_name, output_downsampled_mni_filename
