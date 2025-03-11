import json
import os
import numpy as np
import pandas as pd

import nibabel as nib
from nilearn.image import resample_img, resample_to_img

import jsonpickle

#import dominate
#from dominate.tags import *

import local_helpers as lh
import common_helpers as ch

AGGREGATOR_CACHE = {} # REMOTE CACHE
EXECUTOR_CACHE = {}  # LOCAL CACHE

AGGREGATOR_CACHE_DIR="./debug/agg_cache_dir"
os.makedirs(AGGREGATOR_CACHE_DIR, exist_ok=True)
EXECUTOR_CACHE_DIRS={}

EXECUTOR_OUTPUT_DIRS={}

def get_SSE(y_actual, y_pred):
    return np.sum((y_actual - y_pred) ** 2)


# TODO: Needs file to be sent from executors to aggregator


def calculate_mask(site_results, agg_cache_dir, output_dir):
    """Calculates the average of all masks"""
    num_sites = len(site_results.keys())
    site_ids = sorted(site_results.keys())
    avg_of_all = None
    for site, results in site_results.items():
        if avg_of_all is not None:
            avg_of_all += np.asarray(results["avg_nifti"])
        else:
            avg_of_all = np.asarray(results["avg_nifti"])

    avg_of_all = avg_of_all/num_sites

    # Threshold binarizer
    user_id = site_ids[0]
    threshold = site_results[user_id]["threshold"]
    voxel_size = site_results[user_id]["voxel_size"]

    mask_info = avg_of_all > threshold

    clipped_img_header, clipped_img_affine = site_results[user_id]["avg_header"], site_results[user_id]["avg_affine"]

    downsampled_mni, downsampled_mask, clipped_img = ch.get_downsampled_mni_and_mask(mask_info, clipped_img_affine, clipped_img_header, voxel_size)


    # TODO check how to send these to executors
    #nib.save(downsampled_mask, os.path.join(output_dir, "mask.nii"))
    #nib.save(downsampled_mni, os.path.join(output_dir, "mni_downsampled.nii"))
    nib.save(downsampled_mask, os.path.join(agg_cache_dir, "mask.nii"))
    nib.save(downsampled_mni, os.path.join(agg_cache_dir, "mni_downsampled.nii"))

    return downsampled_mask, downsampled_mni, clipped_img_header, clipped_img_affine, mask_info



def perform_local_step1(covariates_path, data_path, computation_parameters, cache_dict):
    """The first function in the local computation chain"""

    threshold = computation_parameters["threshold"]
    voxel_size = computation_parameters["voxel_size"]
    lamb = computation_parameters["lambda"]

    # TODO: voxel size cannot be 0 (check how to handle this)
    if voxel_size == 0:
        voxel_size = 1

    categorical_dict = lh.parse_for_categorical(computation_parameters, covariates_path, data_path)
    avg_header, avg_affine, avg_nifti, covar_x = lh.average_nifti(computation_parameters, covariates_path,
                                                               data_path, "output_dir")
    # todo write_file(args, args, "cache", "args_file")

    #TODO - publishing local data metrics- Do this as last step of the computation
    # write local counts to a text file (to be sent to remote)
    # counts_file = os.path.join(state_["outputDirectory"], "local_counts.csv")
    # with open(counts_file, "a") as fn:
    #     fn.write("Output of pd.describe(): \n")
    # covar_x.describe().to_csv(counts_file, mode="a", header=True)
    #
    # for column in covar_x:
    #     if covar_x[column].dtype == object:
    #         with open(counts_file, "a") as fn:
    #             fn.write(f"Counts for column: {column}\n")
    #         covar_x[column].value_counts().to_csv(counts_file, mode="a", header=True)
    #
    #         with open(counts_file, "a") as fn:
    #             fn.write(f"Mean of groups for column: {column}\n")
    #         covar_x.groupby([column]).mean().to_csv(counts_file, mode="a", header=True)

    # converting all the values of the reference columns to lowercase as
    # all the covariate values are converted to lowercase while parsing
    # in parsers.parse_covar_info()
    reference_dict = dict((k, v.lower()) for k, v in computation_parameters["reference_columns"].items());

    output_dict = {
        "categorical_dict": categorical_dict,
        "reference_columns": reference_dict,
        "threshold": threshold,
        "voxel_size": voxel_size,
        "avg_nifti": avg_nifti,
        "avg_header": avg_header,
        "avg_affine": avg_affine,
        "computation_phase": "local_0",
    }
    local_cache_dict= {"X": covar_x.to_json(orient='split'),
                 "voxel_size": voxel_size,
                 "lambda": lamb}
    results = {"output": output_dict, "cache": local_cache_dict}


    return results

def perform_remote_step1(site_results, agg_cache_dict, agg_cache_dir):
    def return_uniques_and_counts(df):
        """Return unique-values of the categorical variables and their counts"""
        keys, count = dict(), dict()
        keys = (
            df.iloc[:, :].sum(axis=1).apply(set).apply(sorted).to_dict()
        )  # adding all columns
        count = {k: len(v) for k, v in keys.items()}

        return keys, count


    """The first function in the remote computation chain"""
    downsampled_mask, downsampled_mni, clipped_img_header, clipped_img_affine, mask_info = calculate_mask(site_results, agg_cache_dir, "output_dir")
    site_info = {site: site_results[site]["categorical_dict"] for site in site_results.keys()}
    ref_cols = {site: site_results[site]["reference_columns"] for site in site_results.keys()}

    reference_dict=next(iter(ref_cols.values()))
    # assert that reference columns/values for dummy encoding are same across all the sites for
    # the correctness of the decentralized regression. This is always true for coinstac GUI computation.
    assert all(value == reference_dict  for value in ref_cols.values()), \
        "Reference values for dummy encoding are not same across all the sites "+ str(ref_cols)

    df = pd.DataFrame.from_dict(site_info)
    covar_keys, unique_count = return_uniques_and_counts(df)

    cache_dict = {}
    global_results = {
        "output": {
            "covar_keys": jsonpickle.encode(covar_keys, unpicklable=False),
            "reference_columns": reference_dict,
            "global_unique_count": unique_count,
            "mask_data": downsampled_mask.get_fdata(),
            "mask_header": downsampled_mask.header,
            "mask_affine" : downsampled_mask.affine,
            "downsampled_mni_data": downsampled_mni.get_fdata(),
            "downsampled_mni_header": downsampled_mni.header,
            "downsampled_mni_affine" : downsampled_mni.affine,
            "mask_info":mask_info,
            "clipped_img_header": clipped_img_header,
            "clipped_img_affine": clipped_img_affine,
            "computation_phase": "remote_0",
        },
        "cache": cache_dict ,
    }

    global AGGREGATOR_CACHE
    AGGREGATOR_CACHE.update(cache_dict)

    return global_results


def perform_local_step2(agg_result, data_path, cache_dict, cache_dir, output_dir,
                        transfer_dir="transfer_dir"):
    def vbm_parser(agg_result, X):
        """Parse the nifti (.nii) specific inputspec.json and return the
        covariate matrix (X) as well the dependent matrix (y) as dataframes
        """
        y_info = lh.nifti_to_data(agg_result, data_path, cache_dict, cache_dir, X)
        encoded_covar_info = lh.perform_encoding(agg_result, data_path, cache_dict, X)

        return (encoded_covar_info, y_info)

    """The second function in the local computation chain"""
    X = pd.read_json(cache_dict["X"], orient='split')
    regularizer_l2 = cache_dict["lambda"]

    # Local Statistics
    encoded_X, y = vbm_parser(agg_result, X)
    meanY_vector, lenY_vector = lh.mean_and_len_y(y)
    _, local_stats_list = lh.local_stats_to_dict_numba(encoded_X, y, cache_dir, output_dir)

    # Global Statistics
    augmented_X = lh.add_site_covariates(agg_result, X)
    X_labels = list(augmented_X.columns)
    biased_X = augmented_X.values.astype("float64")

    XtransposeX_local = lh.multiply(biased_X, biased_X)
    Xtransposey_local = lh.multiply(biased_X, y)

    # Writing covariates and dependents to cache as files
    ch.saveBin(os.path.join(cache_dir, "X.npy"), biased_X)
    ch.saveBin(os.path.join(cache_dir, "y.npy"), y)

    # TODO: Writing XTX and XTy to output as files
    #ch.saveBin(os.path.join(transfer_dir, "XTX.npy"), XtransposeX_local)
    #ch.saveBin(os.path.join(transfer_dir, "XTy.npy"), Xtransposey_local)

    output_dict = {
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "mean_y_local": meanY_vector.tolist(),
        "count_local": lenY_vector.tolist(),
        "local_stats_list": local_stats_list,
        "X_labels": X_labels,
        "lambda": regularizer_l2,
    }
    cache_dict = {"covariates": "X.npy", "dependents": "y.npy",
                  "reference_columns": agg_result["reference_columns"]}

    #write_file(args, output_dict, "output", "local_output")

    result = {
        "output": output_dict,
        "cache": cache_dict,
    }

    global EXECUTOR_CACHE
    EXECUTOR_CACHE.update(cache_dict)

    return result

def perform_remote_step2(site_results, agg_cache_dict, agg_cache_dir, transfer_dir="transfer_dir"):
    """The second function in the local computation chain"""
    site_ids = sorted(site_results.keys())
    user_id = site_ids[0]

    X_labels = site_results[user_id]["X_labels"]

    all_local_stats_dicts = [
        site_results[site]["local_stats_list"] for site in site_results
    ]

    beta_vector_0 = sum([np.asarray(site_results[site]["XtransposeX_local"]) for site in site_results])
    beta_vector_1 = sum([np.asarray(site_results[site]["Xtransposey_local"]) for site in site_results])

    all_lambdas = [site_results[site]["lambda"] for site in site_results]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    try:
        #Add lambda's
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

    # saveBin(
    #     os.path.join(args["state"]["transferDirectory"], "avg_beta_vector.npy"),
    #     avg_beta_vector,
    # )
    # saveBin(
    #     os.path.join(args["state"]["transferDirectory"], "mean_y_global.npy"),
    #     mean_y_global,
    # )
    #
    ch.saveBin(os.path.join(agg_cache_dir, "avg_beta_vector.npy"),avg_beta_vector)

    output_dict = {
        "avg_beta_vector": avg_beta_vector.tolist(),
        "mean_y_global": mean_y_global.tolist(),
        "computation_phase": "remote_1",
    }

    cache_dict = {
        "avg_beta_vector": "avg_beta_vector.npy",
        "dof_global": dof_global.tolist(),
        "X_labels": X_labels,
        "local_stats_dict": all_local_stats_dicts,
    }

    result = {"output": output_dict, "cache": cache_dict}

    global AGGREGATOR_CACHE
    AGGREGATOR_CACHE.update(cache_dict)

    return result


def perform_local_step3(agg_result, data_path, cache_dict, cache_dir, output_dir,
                        transfer_dir="transfer_dir"):

    biased_X = ch.loadBin(os.path.join(cache_dir, cache_dict["covariates"]))
    y = ch.loadBin(os.path.join(cache_dir, cache_dict["dependents"]))

    avg_beta_vector = np.asarray(agg_result["avg_beta_vector"])

    mean_y_global =np.asarray(agg_result["mean_y_global"])

    varX_matrix_local = lh.multiply(biased_X, biased_X)

    sse_local, sst_local = lh.stats_calculation( biased_X, y, avg_beta_vector, mean_y_global)

    output_dict = {
        "SSE_local": sse_local.tolist(),
        "SST_local": sst_local.tolist(),
        "varX_matrix_local": varX_matrix_local.tolist(),
    }

    cache_dict = {"avg_beta_vector": avg_beta_vector.tolist()}
    result = {"output": output_dict, "cache": cache_dict}


    global EXECUTOR_CACHE
    EXECUTOR_CACHE.update(cache_dict)

    return  result


def perform_remote_step3(site_results, agg_cache_dict, agg_cache_dir, transfer_dir="transfer_dir"):
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

    avg_beta_vector = ch.loadBin(os.path.join(agg_cache_dir, agg_cache_dict["avg_beta_vector"]))
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

    # print_pvals(args, ps_global, ts_global, X_labels)
    # print_beta_images(args, avg_beta_vector, X_labels)
    # print_r2_image(args, r_squared_global)
    #
    # # Block of code to print local stats as well
    # sites = [site for site in input_list]
    #
    # all_local_stats_dicts = agg_cache_dict["local_stats_dict"]
    # all_local_stats_dicts = dict(zip(sites, all_local_stats_dicts))
    #
    # # List all output files (global stats)
    # global_files = sorted(os.listdir(state_["outputDirectory"]))
    #
    # # move global stats to transfer directory to be moved to local sites
    # os.mkdir(os.path.join(state_["transferDirectory"], "global_stats"))
    # for f in global_files:
    #     shutil.move(
    #         os.path.join(state_["outputDirectory"], f),
    #         os.path.join(state_["transferDirectory"], "global_stats", f),
    #     )
    #
    # # Get list of global pngs files
    # global_png_files = [
    #     f"global_stats/{file}" for file in global_files if file.endswith(".png")
    # ]
    #
    # # Get list of local png files
    # local_png_files = {
    #     site: [file for file in site_files if file.endswith(".png")]
    #     for site, site_files in all_local_stats_dicts.items()
    # }
    #
    # # Gather local and global png files (for display in UI)
    # output_dict = {"local_stats": local_png_files, "global_stats": global_png_files}

    output_dict = {
    "r_squared_global": r_squared_global.tolist(),
    "MSE" : MSE.tolist(),
    "ts_global" : ts_global.tolist(),
    "ps_global" : ps_global,
    "X_labels": X_labels
    }
    computation_output_dict = {"output": output_dict, "cache": {}}


    global AGGREGATOR_CACHE
    AGGREGATOR_CACHE.update({})

    return computation_output_dict


def perform_local_step4(agg_result, data_path, cache_dict, cache_dir, output_dir,
                            transfer_dir="transfer_dir"):
    avg_beta_vector = cache_dict["avg_beta_vector"]
    r_squared_global = np.asarray(agg_result["r_squared_global"])
    MSE = np.asarray(agg_result["MSE"])
    ts_global = np.asarray(agg_result["ts_global"])
    ps_global = np.asarray(agg_result["ps_global"])
    X_labels = np.asarray(agg_result["X_labels"])


    global_output_dir = output_dir+"global"
    os.makedirs(global_output_dir, exist_ok=True)
    lh.print_pvals(cache_dir, global_output_dir, ps_global, ts_global, X_labels)
    lh.print_beta_images(cache_dir, global_output_dir, avg_beta_vector, X_labels)
    lh.print_r2_image(cache_dir, global_output_dir, r_squared_global)

    result = {"output": {}, "cache": {}}

    global EXECUTOR_CACHE
    EXECUTOR_CACHE.update({})

    return result

def run_federated_ssr():
    # Example usage for multiple sites
    sites = ['site1', 'site2']
    computation_parameters=None
    with open('../test_data/server/parameters.json') as f:
        computation_parameters = json.load(f)
    #computation_parameters = json.loads('../test_data/server/parameters.json')

    #Iteration - 1
    site_results = {}
    for site in sites:
        EXECUTOR_CACHE[site]={}
        EXECUTOR_CACHE_DIRS[site] = f"./debug/cache_{site}/"
        os.makedirs(EXECUTOR_CACHE_DIRS[site], exist_ok=True)

        EXECUTOR_OUTPUT_DIRS[site] = f"./debug/output_{site}/"
        os.makedirs(EXECUTOR_OUTPUT_DIRS[site], exist_ok=True)

        covariates_path = os.path.join(f'../test_data/{site}/covariates.csv')
        data_path = os.path.join(f'../test_data/{site}/')
        result = perform_local_step1(covariates_path, data_path, computation_parameters, cache_dict=EXECUTOR_CACHE[site])
        site_results[site] = result["output"]
        EXECUTOR_CACHE[site].update(result["cache"])

    rem_results = perform_remote_step1(site_results, AGGREGATOR_CACHE, AGGREGATOR_CACHE_DIR)
    remote_results = rem_results["output"]

    #Iteration - 2
    site_results = {}
    for site in sites:
        data_path = os.path.join(f'../test_data/{site}/')
        result = perform_local_step2(remote_results, data_path, cache_dict=EXECUTOR_CACHE[site],
                                                 cache_dir=EXECUTOR_CACHE_DIRS[site],
                                                 output_dir=EXECUTOR_OUTPUT_DIRS[site],
                                                 transfer_dir="transfer_dir")
        site_results[site] = result["output"]
        EXECUTOR_CACHE[site].update(result["cache"])

    rem_results = perform_remote_step2(site_results, agg_cache_dict=AGGREGATOR_CACHE,
                                       agg_cache_dir=AGGREGATOR_CACHE_DIR, transfer_dir="transfer_dir")
    remote_results = rem_results["output"]

    # Iteration - 3
    site_results = {}
    for site in sites:
        data_path = os.path.join(f'../test_data/{site}/')
        result = perform_local_step3(remote_results, data_path, cache_dict=EXECUTOR_CACHE[site],
                                     cache_dir=EXECUTOR_CACHE_DIRS[site],
                                     output_dir=EXECUTOR_OUTPUT_DIRS[site],
                                     transfer_dir="transfer_dir")
        site_results[site] = result["output"]
        EXECUTOR_CACHE[site].update(result["cache"])

    rem_results = perform_remote_step3(site_results, agg_cache_dict=AGGREGATOR_CACHE,
                                       agg_cache_dir=AGGREGATOR_CACHE_DIR, transfer_dir="transfer_dir")
    remote_results = rem_results["output"]

    # Iteration - 4
    site_results = {}
    for site in sites:
        data_path = os.path.join(f'../test_data/{site}/')
        result = perform_local_step4(remote_results, data_path, cache_dict=EXECUTOR_CACHE[site],
                                     cache_dir=EXECUTOR_CACHE_DIRS[site],
                                     output_dir=EXECUTOR_OUTPUT_DIRS[site],
                                     transfer_dir="transfer_dir")
        site_results[site] = result["output"]
        EXECUTOR_CACHE[site].update(result["cache"])

    # Save the results
    # Save as png file

run_federated_ssr()