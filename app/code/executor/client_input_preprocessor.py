import os
import numpy as np
import pandas as pd
import nibabel as nib
import statsmodels.api as sm
from numba import jit

from typing import Dict, Any
from distutils.util import strtobool

from utils.logger import _log_message
from . import client_constants

def validate_and_get_inputs(covariates_path: str, data_path: str, computation_parameters: Dict[str, Any], log_path: str) -> bool:
    try:
        # Extract covariates and dependent headers from computation parameters
        #If given as covariate:datatype as input format
        expected_covariates_info = computation_parameters["Covariates"]
        expected_covariates = list(expected_covariates_info.keys())
        ignore_subjects_with_invalid_entries = computation_parameters.get("IgnoreSubjectsWithInvalidData",
                                                                          client_constants.DEFAULT_IgnoreSubjectsWithInvalidData)
        ignore_subjects_with_invalid_entries = bool(strtobool(str(ignore_subjects_with_invalid_entries)))

        _log_message(f' ignore_subjects_with_invalid_entries = {ignore_subjects_with_invalid_entries}', log_path,
                     "info")

        # Load the data
        covariates = pd.read_csv(covariates_path, index_col="niftifilename", header=0, skipinitialspace=True)

        # Validate covariates headers
        covariates_headers = set(covariates.columns)
        if not set(expected_covariates).issubset(covariates_headers):
            error_message = f"Covariates headers do not contain all expected headers. Expected at least {expected_covariates}, but got {covariates_headers}."
            _log_message(error_message, log_path, "info")
            return False, None, None

        _log_message(f'-- Checking covariate file : {str(covariates_path)}', log_path, "info")
        X = convert_data_to_given_type(covariates, expected_covariates_info, log_path, ignore_subjects_with_invalid_entries)
        X = convert_bool_type_to_object(X, expected_covariates_info, log_path)

        #dummy encoding categorical variables
        #X = pd.get_dummies(X, drop_first=True)

        _log_message(f'-- Checking dependent nifti files in : {str(data_path)}', log_path, "info")

        # Checks for existence of files and if they don't delete row
        # Do not load nii files into Y at this step, we need to first compute average_mask in a decentralized way and
        # then load the nii files based on the average_mask
        missing_dependent_files=[]
        invalid_dependent_data_files = []

        for file_name in X.index:
            if not os.path.isfile(os.path.join(data_path, file_name)):
                missing_dependent_files.append(file_name)
                X.drop(file_name, inplace=True)
            else:
                image_data = nib.load(os.path.join(data_path, file_name)).dataobj[:]
                if (np.all(np.isnan(image_data)) or np.count_nonzero(image_data) == 0 or image_data.size == 0 ):
                    invalid_dependent_data_files.append(file_name)
                    X.drop(index=file_name, inplace=True)

        if len(missing_dependent_files) > 0:
            _log_message(f'-- Ignored following subjects with missing files: '
                         f'{str(missing_dependent_files)}', log_path, "info")

        if len(invalid_dependent_data_files) > 0:
            _log_message(f'-- Ignored following subjects with invalid data in nifti files: '
                         f'{str(invalid_dependent_data_files)}', log_path, "info")

        if len(missing_dependent_files) > 0 or len(invalid_dependent_data_files) > 0:
            if not ignore_subjects_with_invalid_entries:
                raise Exception(
                    f'Following filenames are missing or have invalid data in nifti files. Either choose to ignore '
                    f'these or correct the data and try again. See log file for details.')

        # Raise Exception if none of the files are found
        if X.index.empty:
            raise Exception(f"Could not find any .nii files specified in the covariates csv. "
                            f"Looked in datapath: {data_path}")

        # If all checks pass
        return True, X

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        _log_message(error_message, log_path, "error")
        return False, None


def convert_bool_type_to_object(data_df : pd.DataFrame, column_info : dict, log_path : str):
    # All the potential
    try:
        # convert bool to categorical as soon as possible
        for column_name, column_datatype in column_info.items():
            if column_datatype.strip().lower() == "bool":
                data_df[column_name] = data_df[column_name].astype('object')

        # convert contents of object columns to lowercase
        for column_name, column_datatype in column_info.items():
            if column_datatype.strip().lower() == "bool":
                data_df[column_name] = data_df[column_name].astype("str").str.lower()

    except Exception as e:
        error_message = f"An error occurred while converting bool to object type: {str(e)}"
        _log_message(error_message, log_path, "error")
        raise (e)

    return data_df


def convert_data_to_given_type(data_df : pd.DataFrame, column_info : dict, log_path : str, ignore_subjects_with_invalid_entries: bool):
    expected_column_names = column_info.keys()

    all_rows_to_ignore = _validate_data_datatypes(data_df, column_info, log_path)
    if len(all_rows_to_ignore) > 0:
        if ignore_subjects_with_invalid_entries:
            _log_message(f'-- Ignored following rows with incorrect column values: '
                         f'{str(all_rows_to_ignore)}', log_path, "info")

            data_df.drop(data_df.index[all_rows_to_ignore], inplace=True)
        else:
            raise Exception(f'Following rows have empty or invalid entries for covariate columns. Either choose to ignore these'
                            f' rows or correct the data and try again. See log file for details: {str(all_rows_to_ignore)}')

    else:
        _log_message(f' Data validation passed for all the columns: {str(expected_column_names)}', log_path, "info")

    # All the potential
    try:
        for column_name, column_datatype in column_info.items():
            _log_message(f'Casting datatype of column: {column_name} to the requested datatype '
                             f': {column_datatype}', log_path, "info")
            if column_datatype.strip().lower() == "int":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('int') #or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                data_df[column_name] = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('bool')
            else:
                raise Exception(f'Invalid datatype provided in the input for column : {column_name}'
                                f' and datatype: {column_datatype}. Allowed datatypes are int, float, str, bool.')
        # Check for null or NaNs in the converted data
        curr_rows_to_ignore = data_df[data_df.isnull().any(axis=1)].index.tolist()
        if len(curr_rows_to_ignore) > 0:
            if ignore_subjects_with_invalid_entries:
                _log_message(f'-- Ignored following rows with incorrect column values: '
                     f'{str(all_rows_to_ignore)}', log_path, "info")

                data_df.drop(data_df.index[curr_rows_to_ignore], inplace=True)
            else:
                raise Exception(f'Following rows have empty or invalid entries for columns after converting to their '
                                f'respective datatypes. Either choose to ignore these'
                            f' rows or correct the data and try again. See log file for details: {str(all_rows_to_ignore)}')


        data_df = data_df[expected_column_names]

    except Exception as e:
        error_message = f"An error occurred during type conversion for data: {str(e)}"
        _log_message(error_message, log_path, "error")
        raise (e)

    return data_df

def _validate_data_datatypes(data_df : pd.DataFrame, column_info : dict, log_path : str) -> list:
    all_rows_to_ignore=set()
    try:
        for column_name, column_datatype in column_info.items():
            _log_message(f'Validating column: {column_name} with requested datatype '
                             f': {column_datatype}', log_path, "info")
            if column_datatype.strip().lower() == "int":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int') #or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                temp = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                #Converting first to 'int' type to make sure all the possible values are converted correctly
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int') #or .astype('Int64')
            else:
                raise Exception(f'Invalid datatype provided in the input for column : {column_name}'
                                f' and datatype: {column_datatype}. Allowed datatypes are int, float, str, bool.')

            #Check for null or NaNs in the data
            rows_to_ignore = data_df[temp.isnull()].index.tolist()

            #Check for emtpy values in the data
            if column_datatype.strip().lower() == "str":
                rows_to_ignore = data_df[temp.str.strip() == ''].index

            all_rows_to_ignore = all_rows_to_ignore.union(rows_to_ignore)

            if len(rows_to_ignore) > 0:
                _log_message(f'Rows with incorrect values for column {column_name} : '
                         f'{str(rows_to_ignore)}', log_path, "info")
            else:
                _log_message(f'Data validation passed for column: {column_name} to the requested datatype '
                             f': {column_datatype}', log_path, "info")

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        _log_message(error_message, log_path, "error")
        raise (e)

    return list(all_rows_to_ignore)


def perform_encoding(agg_result, data_f):
    """Perform encoding of various categorical variables"""
    cols_categorical = [col for col in data_f if data_f[col].dtype == object]
    cols_mono = [col for col in data_f if data_f[col].nunique() == 1]

    # Dropping columns with unique values
    data_f = data_f.drop(columns=cols_mono)

    # Creating dummies on non-unique categorical variables
    cols_nodrop = [x for x in cols_categorical if x not in cols_mono]
    default_col_sortedval_dict = _get_default_dummy_encoding_columns(data_f)
    data_f = pd.get_dummies(data_f, columns=cols_nodrop, drop_first=False)
    data_f = _adjust_dummy_encoding_columns( data_f, agg_result["reference_columns"],
                                           default_col_sortedval_dict)

    data_f = data_f.dropna(axis=0, how="any")
    data_f = sm.add_constant(data_f, has_constant="add")

    return data_f


def _get_default_dummy_encoding_columns(df):
    """Returns a dictionary of the first sorted unique-value of all categorical variables."""

    default_col_sortedval_dict={}
    categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
    for col_name in categorical_cols:
        default_col_sortedval_dict[col_name]=sorted(df[col_name].unique())[0]

    return default_col_sortedval_dict

def _adjust_dummy_encoding_columns(df, ref_col_val_dict, data_def_col_val_dict):
    """ If a column is listed in reference_columns in the input for dummy encoding,
     then the values in this dict is used the reference column from the dataframe,
     otherwise the default sorted first value is used for a column."""

    for col_name in data_def_col_val_dict.keys():
        ref_col_val =col_name +"_"+ ref_col_val_dict.get(col_name, data_def_col_val_dict[col_name])
        df.drop(ref_col_val, inplace=True, axis=1)

    return df

def mean_and_len_y(y):
    """Caculate the mean and length of each y vector"""
    meanY_vector = y.mean(axis=0)
    #    lenY_vector = y.count(axis=0)
    lenY_vector = np.count_nonzero(~np.isnan(y), axis=0)

    return meanY_vector, lenY_vector

def add_site_covariates(agg_result, X):
    import simplejson as json

    """Add site covariates based on information gathered from all sites"""
    all_sites = agg_result["covar_keys"]
    glob_uniq_ct = agg_result["global_unique_count"]
    reference_col_dict= agg_result["reference_columns"]

    all_sites = json.loads(all_sites)

    default_col_sortedval_dict = _get_default_dummy_encoding_columns(X)

    for key, val in glob_uniq_ct.items():
        if val == 1:
            X.drop(columns=key, inplace=True)
            default_col_sortedval_dict.pop(key)
        else:
            default_col_sortedval_dict[key] = sorted(all_sites[key])[0]
            covar_dict = pd.get_dummies(all_sites[key], prefix=key, drop_first=False)
            X = _merging_globals(X, covar_dict, all_sites, key)

    X = _adjust_dummy_encoding_columns(X, reference_col_dict, default_col_sortedval_dict)
    X = X.dropna(axis=0, how="any")
    biased_X = sm.add_constant(X, has_constant="add")

    return biased_X

def _merging_globals(X, site_covar_dict, dict_, key):
    """Merge the actual data frame with the created dummy matrix"""
    site_covar_dict.rename(index=dict(enumerate(dict_[key])), inplace=True)
    site_covar_dict.index.name = key
    site_covar_dict.reset_index(level=0, inplace=True)
    X = X.merge(site_covar_dict, on=key, how="left")
    X = X.drop(columns=key)

    return X


def _get_default_dummy_encoding_columns(df):
    """Returns a dictionary of the first sorted unique-value of all categorical variables."""

    default_col_sortedval_dict={}
    categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
    for col_name in categorical_cols:
        default_col_sortedval_dict[col_name]=sorted(df[col_name].unique())[0]

    return default_col_sortedval_dict

@jit(nopython=True)
def multiply(array_a, array_b):
    """Multiplies two matrices"""
    return array_a.T @ array_b

