import os.path

import numpy as np
import pandas as pd
import glob
from numba import jit
from . import client_constants

from utils.data_transfer import *
# from ..utils.data_transfer import *
from . import client_input_preprocessor as cip
from . import client_nifti_helper as cnh


def perform_client_step1(covariates_path, data_path, computation_parameters, log_path, cache_dict, cache_dir):
    # Validate the run inputs (covariates, dependent data, and parameters)
    is_valid, covariates_X = cip.validate_and_get_inputs(covariates_path, data_path, computation_parameters, log_path)
    if not is_valid:
        # Halt execution if validation fails
        raise ValueError(f"Invalid run input. Check validation log at {log_path}")

    categorical_dict = {col: list(covariates_X[col].unique()) for col in covariates_X.select_dtypes(include=object)}
    # converting all the values of the reference columns to lowercase as
    # all the covariate values are converted to lowercase while parsing
    reference_dict = dict((k, v.lower()) for k, v in computation_parameters.get("ReferenceColumns",
                                                                                client_constants.DEFAULT_ReferenceColumns).items());

    # Handle dependent nifti files
    threshold = computation_parameters.get("Threshold", client_constants.DEFAULT_Threshold)
    voxel_size = computation_parameters.get("VoxelSize", client_constants.DEFAULT_VoxelSize)
    lamb = computation_parameters.get("Lambda", client_constants.DEFAULT_Lambda)

    # TODO: voxel size cannot be 0 (check how to handle this)
    if voxel_size == 0:
        voxel_size = 1

    # create a folder inside cache_dir that holds files to transfer
    data_transfer_dir = os.path.join(cache_dir, "transfer_client_step1")
    os.makedirs(data_transfer_dir, exist_ok=True)

    average_dependent_filename = cnh.average_nifti(covariates_X, data_path, data_transfer_dir)

    # Output
    output_dict = {
        "categorical_dict": categorical_dict,
        "reference_columns": reference_dict,
        "threshold": threshold,
        "voxel_size": voxel_size
    }

    local_cache_dict = {"X": covariates_X.to_json(orient='split'),
                        "voxel_size": voxel_size,
                        "lambda": lamb}

    results = {"output": output_dict, "cache": local_cache_dict,
               "transfer_files": {"average_nifti": average_dependent_filename}
               }

    return results


def perform_local_step2(agg_result, data_path, log_path, cache_dict, cache_dir):
    # save mask and downsampled mni in cache
    mask_file_name = os.path.join(cache_dir, "mask.nii")
    bytes_to_file(mask_file_name, agg_result["mask_file"])

    downsampled_mni_file_name = os.path.join(cache_dir, "downsampled_mni.nii")
    bytes_to_file(downsampled_mni_file_name, agg_result["downsampled_mni"])

    X = pd.read_json(cache_dict["X"], orient='split')

    """The second function in the local computation chain"""
    regularizer_l2 = cache_dict["lambda"]

    # Local Statistics
    # y = cnh.nifti_to_data(agg_result, data_path, cache_dict, cache_dir, X, mask_file_name,
    #                       downsampled_mni_file_name)
    y = cnh.nifti_to_data(data_path, cache_dict, X, mask_file_name, downsampled_mni_file_name)

    encoded_X = cip.perform_encoding(agg_result, X)

    meanY_vector, lenY_vector = cip.mean_and_len_y(y)
    local_beta_vector, local_pvalues, local_tvalues, local_rsquared, local_X_labels = _local_stats_to_dict_numba(
        encoded_X, y)

    # Global Statistics
    augmented_X = cip.add_site_covariates(agg_result, X)
    X_labels = list(augmented_X.columns)
    biased_X = augmented_X.values.astype("float64")

    XtransposeX_local = cip.multiply(biased_X, biased_X)
    Xtransposey_local = cip.multiply(biased_X, y)

    # Writing covariates and dependents to cache as files
    save_binary(os.path.join(cache_dir, "X.npy"), biased_X)
    save_binary(os.path.join(cache_dir, "y.npy"), y)

    x_transpose_x_file_name = os.path.join(cache_dir, "XTX.npy")
    save_binary(x_transpose_x_file_name, XtransposeX_local)

    x_transpose_y_file_name = os.path.join(cache_dir, "XTy.npy")
    save_binary(x_transpose_y_file_name, Xtransposey_local)

    output_dict = {
        "mean_y_local": meanY_vector.tolist(),
        "count_local": lenY_vector.tolist(),
        # "local_stats_list": local_stats_list,
        "X_labels": X_labels,
        "lambda": regularizer_l2,
    }
    cache_dict = {"covariates": "X.npy",
                  "dependents": "y.npy",
                  "mask_file_name": mask_file_name,
                  "downsampled_mni_file_name": downsampled_mni_file_name,
                  "reference_columns": agg_result["reference_columns"],
                  "local_beta_vector": local_beta_vector.tolist(),
                  "local_pvalues": local_pvalues.tolist(),
                  "local_tvalues": local_tvalues.tolist(),
                  "local_rsquared": local_rsquared.tolist(),
                  "local_X_labels": local_X_labels
                  }
    result = {
        "output": output_dict,
        "cache": cache_dict,
        "transfer_files": {"x_transpose_x_file_name": x_transpose_x_file_name,
                           "x_transpose_y_file_name": x_transpose_y_file_name
                           }
    }

    return result


def perform_local_step3(agg_result, log_path, cache_dict, cache_dir):
    # save avg_beta_vector and mean_y_global files from aggregator in cache_dir
    avg_beta_vector_file_name = os.path.join(cache_dir, "avg_beta_vector.npy")
    bytes_to_file(avg_beta_vector_file_name, agg_result["avg_beta_vector"])

    mean_y_global_file_name = os.path.join(cache_dir, "mean_y_global.npy")
    bytes_to_file(mean_y_global_file_name, agg_result["mean_y_global"])

    biased_X = load_binary(os.path.join(cache_dir, cache_dict["covariates"]))
    y = load_binary(os.path.join(cache_dir, cache_dict["dependents"]))

    avg_beta_vector = load_binary(avg_beta_vector_file_name)
    mean_y_global = load_binary(mean_y_global_file_name)

    varX_matrix_local = cip.multiply(biased_X, biased_X)

    sse_local, sst_local = _stats_calculation(biased_X, y, avg_beta_vector, mean_y_global)

    output_dict = {
        "SSE_local": sse_local.tolist(),
        "SST_local": sst_local.tolist(),
        "varX_matrix_local": varX_matrix_local.tolist(),
    }

    cache_dict = {"avg_beta_vector_file_name": avg_beta_vector_file_name,
                  "mean_y_global_file_name": mean_y_global_file_name
                  }
    result = {"output": output_dict, "cache": cache_dict}

    return result


def perform_local_step4(agg_result, log_path, cache_dict, cache_dir, output_dir):
    avg_beta_vector = load_binary(cache_dict["avg_beta_vector_file_name"])
    r_squared_global = np.asarray(agg_result["r_squared_global"])
    ts_global = np.asarray(agg_result["ts_global"])
    ps_global = np.asarray(agg_result["ps_global"])
    X_labels = np.asarray(agg_result["X_labels"])
    MSE = np.asarray(agg_result["MSE"])

    # Save global output files
    global_output_dir = os.path.join(output_dir, "global_stats")
    os.makedirs(global_output_dir, exist_ok=True)
    cnh.print_pvals(cache_dict, cache_dir, global_output_dir, ps_global, ts_global, X_labels)
    cnh.print_beta_images(cache_dict, cache_dir, global_output_dir, avg_beta_vector, X_labels)
    cnh.print_r2_image(cache_dict, cache_dir, global_output_dir, r_squared_global)

    # Save local output files
    local_output_dir = os.path.join(output_dir, "local_stats")
    os.makedirs(local_output_dir, exist_ok=True)
    local_X_labels = cache_dict["local_X_labels"]

    cnh.print_pvals(cache_dict, cache_dir, local_output_dir, np.asarray(cache_dict["local_pvalues"]).T,
                    np.asarray(cache_dict["local_tvalues"]).T, local_X_labels)
    cnh.print_beta_images(cache_dict, cache_dir, local_output_dir, np.asarray(cache_dict["local_beta_vector"]),
                          local_X_labels)
    cnh.print_r2_image(cache_dict, cache_dir, local_output_dir, np.asarray(cache_dict["local_rsquared"]))

    # Gather local and global png files (for display in UI)
    results_dict = {
        "local_stats": glob.glob(f'{local_output_dir}/*.png'),
        "global_stats": glob.glob(f'{global_output_dir}/*.png'),
    }

    result = {"output": {"html": _persist_html_from_results(results_dict, os.path.join(output_dir, 'index.html'))},
              "cache": {}}

    return result


def _persist_html_from_results(results_dict, output_file_name):
    import dominate
    from dominate import tags

    doc = dominate.document(title='Results')
    global_stats_label = "global_stats"
    local_stats_label = "local_stats"

    # Add style
    with doc.head:
        tags.style('''
             body {
                 font-family: sans-serif;
             }
             hr {
                 width: 100%;
             }
             table {
                 border-collapse: collapse;
                 margin: 25px 0;
                 font-size: 0.9em;
                 font-family: sans-serif;
                 min-width: 400px;
                 width: 100%;
             }
             table thead tr {
                 background-color: #009879;
                 color: #ffffff;
                 text-align: left;
             }
             table tr:nth-of-type(1) td {
                 font-weight: bold;              
             }
             table tr td:nth-of-type(1) {
                 background-color: white;
                 font-weight: bold;
             }
             table tr td[colspan]{
                 background-color: white;
             }
             table thead tr td { 
                 text-transform: capitalize;
             }
             table th,
             table td {
                 padding: 12px 15px;
                 white-space: nowrap;
             }
             table tbody tr {
                 border-bottom: 1px solid #dddddd;
             }

             table tbody tr:nth-of-type(even) {
                 background-color: #efefef;
             }

             table tbody tr.active-row {
                 font-weight: bold;
                 color: #009879;
             }
         ''')

    # Add document body
    with doc:
        tags.h1('Results')
        tags.hr()
        for result_label in [global_stats_label, local_stats_label]:
            tags.h2(result_label)
            for result_img in results_dict.get(result_label, []):
                site_name_prefix = os.path.basename(
                    result_img.split(f'/{local_stats_label}')[0]) + ' : ' if result_label == local_stats_label else ''
                tags.h3(site_name_prefix + result_label + ' - ' + os.path.basename(result_img))
                tags.img(src=result_img[result_img.index(result_label):])
                tags.hr()

    html_content = str(doc)

    with open(output_file_name, 'w') as f:
        f.write(html_content)

    return output_file_name


def _local_stats_to_dict_numba(X, y):
    import scipy as sp
    from numba import prange

    def gather_local_stats_helper(X, y, pinv):
        """Calculate local statistics"""
        size_y = y.shape[1]

        params = np.zeros((X.shape[1], size_y))
        sse = np.zeros(size_y)
        tvalues = np.zeros((X.shape[1], size_y))
        rsquared = np.zeros(size_y)

        for voxel in prange(size_y):
            curr_y = y[:, voxel]
            beta_vector = pinv @ (X.T @ curr_y)
            params[:, voxel] = beta_vector

            curr_y_estimate = np.dot(beta_vector, X.T)

            SSE_global = np.linalg.norm(curr_y - curr_y_estimate) ** 2
            SST_global = np.sum(np.square(curr_y - np.mean(curr_y)))

            sse[voxel] = SSE_global
            r_squared_global = 1 - (SSE_global / SST_global)
            rsquared[voxel] = r_squared_global

            dof_global = len(curr_y) - len(beta_vector)

            MSE = SSE_global / dof_global
            var_covar_beta_global = MSE * pinv
            se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
            ts_global = beta_vector / se_beta_global

            tvalues[:, voxel] = ts_global

        return (params, sse, tvalues, rsquared, dof_global)

    def gather_local_stats(X, y):
        """Calculate local statistics"""
        try:
            pinv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            cond = np.linalg.cond(X.T @ X);
            raise Exception(f"X.^T*X matrix at local is Singular with condition number: {cond}")

        return gather_local_stats_helper(X, y, pinv)

    """Wrap local statistics into a dictionary to be sent to the remote"""
    X_labels = list(X.columns)

    X1 = X.values.astype("float64")

    params, _, tvalues, rsquared, dof_global = gather_local_stats(X1, y)

    pvalues = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)

    beta_vector = params.T

    return beta_vector, pvalues, tvalues, rsquared, X_labels


@jit(nopython=True)
def _stats_calculation(X, y, avg_beta_vec, mean_y_global):
    """Calculate SSE and SST."""
    size_y = y.shape[1]
    sse_local = np.zeros(size_y)
    sst_local = np.zeros(size_y)

    for voxel in range(y.shape[1]):
        y1 = y[:, voxel]
        beta = avg_beta_vec[voxel]
        mean_y = mean_y_global[voxel]

        y1_estimate = np.dot(beta, X.T)
        sse_local[voxel] = np.linalg.norm(y1 - y1_estimate) ** 2
        sst_local[voxel] = np.sum(np.square(y1 - mean_y))

    return sse_local, sst_local
