import os

import  numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm

import simplejson as json
from numba import jit, prange

import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import plotting

import common_helpers as ch

def parse_covar_info(computation_parameters, covariates_path, data_path):
    """Read covariate information from the UI"""
    covar_information = computation_parameters["Covariates"]

    #TODO use similar to freesurfer: validate_and_get_inputs(covariates_path: str, data_path: str,
    # computation_parameters: Dict[str, Any], log_path: str) -> bool:

    covar_info = pd.read_csv(covariates_path, index_col="niftifile", header=0)

    # Creating this to catch nan in numeric columns
    temp_df = covar_info.apply(pd.to_numeric, errors='ignore')

    #Remove rows with empty or nan values
    rows_with_nan_cells = np.where(pd.isnull(temp_df))[0].tolist()
    #TODO Fix this rows_with_blank_cells = np.where(temp_df.applymap(lambda x: x == ''))[0].tolist()
    rows_with_blank_cells=[]
    ignore_rows = list(set(rows_with_nan_cells).union(rows_with_blank_cells))
    if len(ignore_rows) > 0:
        # Dropping subjects with empty or nan covariate values
        covar_info.drop(covar_info.index[ignore_rows], inplace=True)

    # Checks for existence of files and if they don't delete row
    for file in covar_info.index:
        if not os.path.isfile(os.path.join(data_path, file)):
            covar_info.drop(file, inplace=True)

    # Raise Exception if none of the files are found
    if covar_info.index.empty:
        raise Exception("Could not find .nii files specified in the covariates csv")

    #TODO - change as per flare datatypes --
    # This is for OLD COINSTAC -
    # convert columns to their original datatypes provided in the input
    # for column, value in computation_parameters["Covariates"][covar_info.index[0]].items():
    #     column_type_in_input = type(value).__name__
    #     if column_type_in_input in ['int', 'float']:
    #         covar_info[column] = covar_info[column].astype(column_type_in_input)
    #     else:
    #         covar_info[column] = covar_info[column].astype("object")

    # convert bool to categorical as soon as possible
    for column in covar_info.select_dtypes(bool):
        covar_info[column] = covar_info[column].astype("object")

    # convert contents of object columns to lowercase
    for column in covar_info.select_dtypes(object):
        covar_info[column] = covar_info[column].astype("str").str.lower()

    return covar_info

def parse_for_categorical(computation_parameters, covariates_path, data_path):
    X = parse_covar_info(computation_parameters, covariates_path, data_path)

    site_dict1 = {col: list(X[col].unique()) for col in X.select_dtypes(include=object)}

    return site_dict1

def average_nifti(computation_parameters, covariates_path, data_path, output_dir):
    """Reads in all the nifti images and calculates their average"""
    input_dir = data_path

    covar_x = parse_covar_info(computation_parameters, covariates_path, data_path)

    appended_data = 0
    for image in covar_x.index:
        image_data = nib.load(os.path.join(input_dir, image)).dataobj[:]
        if (
            np.all(np.isnan(image_data))
            or np.count_nonzero(image_data) == 0
            or image_data.size == 0
        ):
            covar_x = covar_x.drop(index=image)
        else:
            appended_data += image_data

    sample_image = nib.load(os.path.join(input_dir, covar_x.index[0]))
    avg_header, avg_affine = sample_image.header, sample_image.affine

    avg_nifti = appended_data / len(covar_x.index)

    # TODO : Find a way to transfer files from executor to aggregator
    # clipped_img = nib.Nifti1Image(avg_nifti, avg_affine, avg_header)
    # output_file = os.path.join(output_dir, "avg_nifti.nii")
    # nib.save(clipped_img, output_file)

    return avg_header, avg_affine, avg_nifti, covar_x


def nifti_to_data(agg_result, data_path, cache_dict, cache_dir, X):
    """Read nifti files as matrices"""
    voxel_size = cache_dict["voxel_size"]
    try:
        mask_info = agg_result["mask_info"]
    except :
        raise Exception("Missing Mask ")

    #TODO: save downsampled mask.. this is not needed if nii files can be sent from aggregator
    mni_downsampled, downsampled_mask, clipped_img = ch.get_downsampled_mni_and_mask(mask_info, agg_result["clipped_img_affine"], agg_result["clipped_img_header"],
                                    voxel_size)
    nib.save(mni_downsampled, os.path.join(cache_dir, ch.DOWNSAMPLED_MNI))
    nib.save(downsampled_mask, os.path.join(cache_dir, ch.MASK))

    mask_data = nib.load(os.path.join(cache_dir, ch.MASK)).get_fdata()
    mask_dim = mask_data.shape

    #TODO to get this file from aggregator
    # mni_image = os.path.join(args["state"]["baseDirectory"], "mni_downsampled.nii")
    mni_image = os.path.join(cache_dir, ch.DOWNSAMPLED_MNI)
    y = np.zeros((len(X.index), np.count_nonzero(mask_data)), dtype="f8")
    for index, image in enumerate(X.index):
        input_file = os.path.join(data_path, image)
        if nib.load(input_file).header.get_zooms()[0] == voxel_size:
            image_data = nib.load(input_file).get_fdata()
        else:
            clipped_img = resample_to_img(input_file, mni_image, force_resample=True)
            image_data = clipped_img.get_fdata()

        a = []
        for slicer in range(mask_dim[-1]):
            img_slice = image_data[slicer, ...]
            msk_slice = mask_data[slicer, ...]
            a.extend(img_slice[msk_slice > 0].tolist())

        y[index, :] = a

    return y


def mean_and_len_y(y):
    """Caculate the mean and length of each y vector"""
    meanY_vector = y.mean(axis=0)
    #    lenY_vector = y.count(axis=0)
    lenY_vector = np.count_nonzero(~np.isnan(y), axis=0)

    return meanY_vector, lenY_vector


###########################

def create_dummies(data_f, cols, drop_flag=True):
    """Create dummy columns"""
    return pd.get_dummies(data_f, columns=cols, drop_first=drop_flag)


def get_default_dummy_encoding_columns(df):
    """Returns a dictionary of the first sorted unique-value of all categorical variables."""

    default_col_sortedval_dict={}
    categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
    for col_name in categorical_cols:
        default_col_sortedval_dict[col_name]=sorted(df[col_name].unique())[0]

    return default_col_sortedval_dict

def adjust_dummy_encoding_columns(df, ref_col_val_dict, data_def_col_val_dict):
    """ If a column is listed in reference_columns in the input for dummy encoding,
     then the values in this dict is used the reference column from the dataframe,
     otherwise the default sorted first value is used for a column."""

    for col_name in data_def_col_val_dict.keys():
        ref_col_val =col_name +"_"+ ref_col_val_dict.get(col_name, data_def_col_val_dict[col_name])
        df.drop(ref_col_val, inplace=True, axis=1)

    return df

def perform_encoding(agg_result, data_path, cache_dict, data_f):
    """Perform encoding of various categorical variables"""
    cols_categorical = [col for col in data_f if data_f[col].dtype == object]
    cols_mono = [col for col in data_f if data_f[col].nunique() == 1]

    # Dropping columns with unique values
    data_f = data_f.drop(columns=cols_mono)

    # Creating dummies on non-unique categorical variables
    cols_nodrop = [x for x in cols_categorical if x not in cols_mono]
    default_col_sortedval_dict = get_default_dummy_encoding_columns(data_f)
    data_f = create_dummies(data_f, cols_nodrop, False)
    data_f = adjust_dummy_encoding_columns( data_f, agg_result["reference_columns"],
                                           default_col_sortedval_dict)

    data_f = data_f.dropna(axis=0, how="any")
    data_f = sm.add_constant(data_f, has_constant="add")

    return data_f

def merging_globals(X, site_covar_dict, dict_, key):
    """Merge the actual data frame with the created dummy matrix"""
    site_covar_dict.rename(index=dict(enumerate(dict_[key])), inplace=True)
    site_covar_dict.index.name = key
    site_covar_dict.reset_index(level=0, inplace=True)
    X = X.merge(site_covar_dict, on=key, how="left")
    X = X.drop(columns=key)

    return X


def add_site_covariates(agg_result, X):
    """Add site covariates based on information gathered from all sites"""
    all_sites = agg_result["covar_keys"]
    glob_uniq_ct = agg_result["global_unique_count"]
    reference_col_dict= agg_result["reference_columns"]

    all_sites = json.loads(all_sites)

    default_col_sortedval_dict = get_default_dummy_encoding_columns(X)

    for key, val in glob_uniq_ct.items():
        if val == 1:
            X.drop(columns=key, inplace=True)
            default_col_sortedval_dict.pop(key)
        else:
            default_col_sortedval_dict[key] = sorted(all_sites[key])[0]
            covar_dict = pd.get_dummies(all_sites[key], prefix=key, drop_first=False)
            X = merging_globals(X, covar_dict, all_sites, key)

    X = adjust_dummy_encoding_columns(X, reference_col_dict, default_col_sortedval_dict)
    X = X.dropna(axis=0, how="any")
    biased_X = sm.add_constant(X, has_constant="add")

    return biased_X


#########################

def print_beta_images(cache_dir, output_dir, avg_beta_vector, covar_labels):
    """Print regression coefficients as nifti files."""
    beta_df = pd.DataFrame(avg_beta_vector, columns=covar_labels)

    try:
        #TODO load from aggregator result -- if files can be sent
        mask = nib.load(os.path.join(cache_dir, ch.MASK))
    except FileNotFoundError:
        mask = nib.load(os.path.join(cache_dir, ch.MASK))

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


def print_pvals(cache_dir, output_dir, ps_global, ts_global, covar_labels ):
    """Print regression coefficients' p-values as nifti files."""
    p_df = pd.DataFrame(ps_global, columns=covar_labels)
    t_df = pd.DataFrame(ts_global, columns=covar_labels)

    try:
        #TODO load from aggregator result -- if files can be sent
        mask = nib.load(os.path.join(cache_dir, ch.MASK))
    except FileNotFoundError:
        mask = nib.load(os.path.join(cache_dir, ch.MASK))

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


def print_r2_image(cache_dir, output_dir, beta_df):
    """Print regression coefficients as nifti files."""
    try:
        #TODO load from aggregator result -- if files can be sent
        mask = nib.load(os.path.join(cache_dir,ch.MASK))
    except FileNotFoundError:
        mask = nib.load(os.path.join(cache_dir,ch.MASK))

    new_data = np.zeros(mask.shape)
    new_data[mask.get_fdata() > 0] = beta_df

    image_string = "r_squared"

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)
    output_file = os.path.join(output_dir, image_string)

    nib.save(clipped_img, output_file + ".nii")

    plotting.plot_stat_map(
        clipped_img, output_file=output_file, display_mode="ortho", colorbar=True
    )

def local_stats_to_dict_numba(X, y, cache_dir, output_dir):
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

    beta_vector = params.T.tolist()

    local_output_dir = output_dir+"_local"
    os.makedirs(local_output_dir, exist_ok=True)
    print_pvals(cache_dir, local_output_dir, pvalues.T, tvalues.T, X_labels)
    print_beta_images(cache_dir, local_output_dir, beta_vector, X_labels)
    print_r2_image(cache_dir, local_output_dir, rsquared)

    local_stats_list = sorted(os.listdir(local_output_dir))

    return beta_vector, local_stats_list

#################################

@jit(nopython=True)
def multiply(array_a, array_b):
    """Multiplies two matrices"""
    return array_a.T @ array_b


@jit(nopython=True)
def stats_calculation(X, y, avg_beta_vec, mean_y_global):
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