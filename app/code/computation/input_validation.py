"""Validate VBM covariates and construct compatible design matrices."""

import os
from distutils.util import strtobool
from logging import Logger
from typing import Any, Dict

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numba import jit

from . import constants


def _normalize_column_name(column_name: str) -> str:
    """Normalize a CSV column name for compatibility matching."""
    return column_name.strip().lower()


def _build_normalized_column_map(
    columns, file_label: str, logger: Logger
) -> Dict[str, str]:
    """Map normalized headers to originals, rejecting ambiguous duplicates."""
    normalized_column_map = {}

    for column in columns:
        normalized_column = _normalize_column_name(str(column))
        if normalized_column in normalized_column_map:
            error_message = (
                f"{file_label} contains duplicate headers after normalization: "
                f"'{normalized_column_map[normalized_column]}' and '{column}'."
            )
            logger.info(error_message)
            raise ValueError(error_message)

        normalized_column_map[normalized_column] = column

    return normalized_column_map


def _select_columns_case_insensitive(
    dataframe: pd.DataFrame,
    expected_columns: list,
    file_label: str,
    logger: Logger,
) -> pd.DataFrame:
    """Select configured columns case-insensitively and restore canonical names."""
    normalized_column_map = _build_normalized_column_map(
        dataframe.columns, file_label, logger
    )
    missing_columns = [
        column
        for column in expected_columns
        if _normalize_column_name(column) not in normalized_column_map
    ]

    if missing_columns:
        only_covariates_are_missing = file_label == "Covariates" and all(
            _normalize_column_name(column) != "niftifilename"
            for column in missing_columns
        )
        label = "covariate column" if only_covariates_are_missing else "column"
        if len(missing_columns) == 1:
            error_message = f"Missing required {label}: {missing_columns[0]}"
        else:
            error_message = f"Missing required {label}s: {', '.join(missing_columns)}"
        logger.info(error_message)
        raise ValueError(error_message)

    actual_columns = [
        normalized_column_map[_normalize_column_name(column)]
        for column in expected_columns
    ]
    selected_dataframe = dataframe[actual_columns].copy()
    selected_dataframe.columns = expected_columns

    return selected_dataframe


def validate_and_get_inputs(
    covariates_path: str,
    data_path: str,
    computation_parameters: Dict[str, Any],
    logger: Logger,
) -> bool:
    """Converts each dataframe column to its type specified in computation parameters."""
    try:
        # Extract covariates and dependent headers from computation parameters
        # If given as covariate:datatype as input format
        expected_covariates_info = computation_parameters["Covariates"]
        expected_covariates = list(expected_covariates_info.keys())
        ignore_subjects_with_missing_data = computation_parameters.get(
            "IgnoreSubjectsWithMissingData",
            constants.DEFAULT_IgnoreSubjectsWithMissingData,
        )
        ignore_subjects_with_missing_data = bool(
            strtobool(str(ignore_subjects_with_missing_data))
        )

        # Load the data
        covariates = pd.read_csv(covariates_path, header=0, skipinitialspace=True)
        covariates = _select_columns_case_insensitive(
            covariates,
            ["niftifilename", *expected_covariates],
            "Covariates",
            logger,
        )
        covariates.set_index("niftifilename", inplace=True)
        logger.info(f"-- Checking covariate file : {str(covariates_path)}")
        X, invalid_covariate_subjects = convert_data_to_given_type(
            covariates,
            expected_covariates_info,
            logger,
            ignore_subjects_with_missing_data,
        )
        X = convert_bool_type_to_object(X, expected_covariates_info, logger)

        # dummy encoding categorical variables
        # X = pd.get_dummies(X, drop_first=True)

        logger.info(f"-- Checking dependents nifti files in : {str(data_path)}")

        # Checks for existence of files and if they don't delete row
        # Do not load nii files into Y at this step, we need to first compute average_mask in a decentralized way and
        # then load the nii files based on the average_mask
        missing_dependent_files = []
        invalid_dependent_data_files = []

        for file_name in X.index:
            if not os.path.isfile(os.path.join(data_path, file_name)):
                missing_dependent_files.append(file_name)
                X.drop(file_name, inplace=True)
            else:
                image_data = nib.load(os.path.join(data_path, file_name)).dataobj[:]
                if (
                    np.all(np.isnan(image_data))
                    or np.count_nonzero(image_data) == 0
                    or image_data.size == 0
                ):
                    invalid_dependent_data_files.append(file_name)
                    X.drop(index=file_name, inplace=True)

        if len(missing_dependent_files) > 0:
            logger.info(
                f"-- Ignored following subjects with missing files: {str(missing_dependent_files)}"
            )

        if len(invalid_dependent_data_files) > 0:
            logger.info(
                f"-- Ignored following subjects with invalid data in nifti files:"
                f" {str(invalid_dependent_data_files)}"
            )

        if len(missing_dependent_files) > 0 or len(invalid_dependent_data_files) > 0:
            if not ignore_subjects_with_missing_data:
                error_message = (
                    "Following filenames are missing or have invalid data in nifti files. Either choose to ignore "
                    "these or correct the data and try again. See log file for details."
                )
                logger.error(error_message)
                raise Exception(error_message)

        # Raise Exception if none of the files are found
        if X.index.empty:
            error_message = f"Could not find any .nii files specified in the covariates csv. Looked in datapath: {data_path}"
            logger.error(error_message)
            raise Exception(error_message)

        # Build omissions report (populated only when IgnoreSubjectsWithMissingData is True)
        omissions_report = {
            "invalid_covariates": [str(s) for s in invalid_covariate_subjects],
            "missing_nifti": [str(s) for s in missing_dependent_files],
            "invalid_nifti": [str(s) for s in invalid_dependent_data_files],
        }
        total_omitted = (
            len(invalid_covariate_subjects)
            + len(missing_dependent_files)
            + len(invalid_dependent_data_files)
        )
        if total_omitted > 0:
            logger.info(
                f"-- Omissions summary: {total_omitted} subject(s) excluded "
                f"(invalid_covariates={len(invalid_covariate_subjects)}, "
                f"missing_nifti={len(missing_dependent_files)}, "
                f"invalid_nifti={len(invalid_dependent_data_files)})"
            )

        # If all checks pass
        return True, X, omissions_report

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        raise


def convert_bool_type_to_object(
    data_df: pd.DataFrame, column_info: dict, logger: Logger
):
    """Converts bool type covariates to categorical for encoding."""
    try:
        # convert bool to categorical as soon as possible
        for column_name, column_datatype in column_info.items():
            if column_datatype.strip().lower() == "bool":
                data_df[column_name] = data_df[column_name].astype("object")

        # convert contents of object columns to lowercase
        for column_name, column_datatype in column_info.items():
            if column_datatype.strip().lower() == "bool":
                data_df[column_name] = data_df[column_name].astype("str").str.lower()

    except Exception as e:
        error_message = (
            f"An error occurred while converting bool to object type: {str(e)}"
        )
        logger.error(error_message)
        raise (e)

    return data_df


def convert_data_to_given_type(
    data_df: pd.DataFrame,
    column_info: dict,
    logger: Logger,
    ignore_subjects_with_missing_data: bool,
):
    """Convert each dataframe column to its requested type.

    ignore_subjects_with_missing_data is true, then the subjects with missing data will be ignored, otherwise
    it gives errors.

    Returns:
        (data_df, dropped_subjects) where dropped_subjects is a list of index labels
        that were removed due to invalid or missing covariate values.
    """
    expected_column_names = column_info.keys()
    dropped_subjects = []

    all_rows_to_ignore = _validate_data_datatypes(data_df, column_info, logger)
    if len(all_rows_to_ignore) > 0:
        if ignore_subjects_with_missing_data:
            logger.info(
                f"-- Ignored following rows with incorrect column values: {str(all_rows_to_ignore)}"
            )
            data_df.drop(index=all_rows_to_ignore, inplace=True)
            dropped_subjects.extend(all_rows_to_ignore)
        else:
            err_msg = (
                f"Following rows have empty or invalid entries for columns. Either choose to ignore these rows "
                f"or correct the data and try again. See log file for details: {str(all_rows_to_ignore)}"
            )
            logger.error(err_msg)
            raise Exception(err_msg)

    else:
        logger.info(
            f" Data validation passed for all the columns: {str(expected_column_names)}"
        )

    # All the potential
    try:
        for column_name, column_datatype in column_info.items():
            logger.info(
                f"Casting datatype of column: {column_name} to the requested datatype : {column_datatype}"
            )
            if column_datatype.strip().lower() == "int":
                data_df[column_name] = pd.to_numeric(
                    data_df[column_name], errors="coerce"
                ).astype("int")  # or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                data_df[column_name] = pd.to_numeric(
                    data_df[column_name], errors="coerce"
                ).astype("float")
            elif column_datatype.strip().lower() == "str":
                data_df[column_name] = data_df[column_name].astype("object")
            elif column_datatype.strip().lower() == "bool":
                data_df[column_name] = pd.to_numeric(
                    data_df[column_name], errors="coerce"
                ).astype("bool")
            else:
                err_msg = (
                    f"Invalid datatype provided in the input for column : {column_name} and datatype: "
                    f"{column_datatype}. Allowed datatypes are int, float, str, bool."
                )
                logger.error(err_msg)
                raise Exception(err_msg)

        # Check for null or NaNs in the converted data (use label-based drop, not positional)
        curr_rows_to_ignore = data_df[data_df.isnull().any(axis=1)].index.tolist()
        if len(curr_rows_to_ignore) > 0:
            if ignore_subjects_with_missing_data:
                logger.info(
                    f"-- Ignored following rows with NaN values after type conversion: {str(curr_rows_to_ignore)}"
                )
                data_df.drop(index=curr_rows_to_ignore, inplace=True)
                dropped_subjects.extend(curr_rows_to_ignore)
            else:
                err_msg = (
                    f"Following rows have empty or invalid entries for columns after converting to their "
                    f"respective datatypes. Either choose to ignore these rows or correct the data and"
                    f" try again. See log file for details: {str(curr_rows_to_ignore)}"
                )
                logger.error(err_msg)
                raise Exception(err_msg)

        data_df = data_df[expected_column_names]

    except Exception as e:
        error_message = f"An error occurred during type conversion for data: {str(e)}"
        logger.error(error_message)
        raise (e)

    # Deduplicate in case a subject was flagged by both validation passes
    seen = set()
    unique_dropped = []
    for s in dropped_subjects:
        if s not in seen:
            seen.add(s)
            unique_dropped.append(s)

    return data_df, unique_dropped


def _validate_data_datatypes(
    data_df: pd.DataFrame, column_info: dict, logger: Logger
) -> list:
    """Validates if each dataframe column is compatible to the type specified in computation parameters."""
    all_rows_to_ignore = set()
    try:
        for column_name, column_datatype in column_info.items():
            logger.info(
                f"Validating column: {column_name} with requested datatype : {column_datatype}"
            )
            if column_datatype.strip().lower() == "int":
                temp = pd.to_numeric(data_df[column_name], errors="coerce").astype(
                    "int"
                )  # or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                temp = pd.to_numeric(data_df[column_name], errors="coerce").astype(
                    "float"
                )
            elif column_datatype.strip().lower() == "str":
                temp = data_df[column_name].astype("object")
            elif column_datatype.strip().lower() == "bool":
                # Converting first to 'int' type to make sure all the possible values are converted correctly
                temp = pd.to_numeric(data_df[column_name], errors="coerce").astype(
                    "Int64"
                )  # or .astype('int')
            else:
                err_msg = (
                    f"Invalid datatype provided in the input for column : {column_name} and datatype: "
                    f"{column_datatype}. Allowed datatypes are int, float, str, bool."
                )
                logger.error(err_msg)
                raise Exception(err_msg)

            # Check for null or NaNs in the data
            rows_to_ignore = data_df[temp.isnull()].index.tolist()

            all_rows_to_ignore = all_rows_to_ignore.union(rows_to_ignore)

            # Check for emtpy values in the data
            if column_datatype.strip().lower() == "str":
                rows_to_ignore = data_df[temp.str.strip() == ""].index

            all_rows_to_ignore = all_rows_to_ignore.union(rows_to_ignore)

            if len(rows_to_ignore) > 0:
                logger.info(
                    f"Rows with incorrect values for column {column_name} : {str(rows_to_ignore)}"
                )

            else:
                logger.info(
                    f"Data validation passed for column: {column_name} to the requested datatype : {column_datatype}"
                )

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        raise (e)

    return list(all_rows_to_ignore)


def perform_encoding(agg_result, data_f):
    """Perform encoding of various categorical variables."""
    cols_categorical = [col for col in data_f if data_f[col].dtype == object]
    cols_mono = [col for col in data_f if data_f[col].nunique() == 1]

    # Dropping columns with unique values
    data_f = data_f.drop(columns=cols_mono)

    # Creating dummies on non-unique categorical variables
    cols_nodrop = [x for x in cols_categorical if x not in cols_mono]
    default_col_sortedval_dict = _get_default_dummy_encoding_columns(data_f)
    data_f = pd.get_dummies(data_f, columns=cols_nodrop, drop_first=False)
    data_f = _adjust_dummy_encoding_columns(
        data_f, agg_result["reference_columns"], default_col_sortedval_dict
    )

    data_f = data_f.dropna(axis=0, how="any")
    data_f = sm.add_constant(data_f, has_constant="add")

    return data_f


def _get_default_dummy_encoding_columns(df):
    """Returns a dictionary of the first sorted unique-value of all categorical variables."""
    default_col_sortedval_dict = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col_name in categorical_cols:
        default_col_sortedval_dict[col_name] = sorted(df[col_name].unique())[0]

    return default_col_sortedval_dict


def _adjust_dummy_encoding_columns(df, ref_col_val_dict, data_def_col_val_dict):
    """Choose and remove each categorical reference level.

    then the values in this dict is used the reference column from the dataframe,
    otherwise the default sorted first value is used for a column.
    """
    for col_name in data_def_col_val_dict.keys():
        ref_col_val = (
            col_name
            + "_"
            + ref_col_val_dict.get(col_name, data_def_col_val_dict[col_name])
        )
        df.drop(ref_col_val, inplace=True, axis=1)

    return df


def mean_and_len_y(y):
    """Caculate the mean and length of each y vector."""
    meanY_vector = y.mean(axis=0)
    #    lenY_vector = y.count(axis=0)
    lenY_vector = np.count_nonzero(~np.isnan(y), axis=0)

    return meanY_vector, lenY_vector


def add_site_covariates(agg_result, X):
    """Add site covariates based on information gathered from all sites."""
    import simplejson as json

    all_sites = agg_result["covar_keys"]
    glob_uniq_ct = agg_result["global_unique_count"]
    reference_col_dict = agg_result["reference_columns"]

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

    X = _adjust_dummy_encoding_columns(
        X, reference_col_dict, default_col_sortedval_dict
    )
    X = X.dropna(axis=0, how="any")
    biased_X = sm.add_constant(X, has_constant="add")

    return biased_X


def _merging_globals(X, site_covar_dict, dict_, key):
    """Merge the actual data frame with the created dummy matrix."""
    site_covar_dict.rename(index=dict(enumerate(dict_[key])), inplace=True)
    site_covar_dict.index.name = key
    site_covar_dict.reset_index(level=0, inplace=True)
    X = X.merge(site_covar_dict, on=key, how="left")
    X = X.drop(columns=key)

    return X


def _get_default_dummy_encoding_columns(df):
    """Returns a dictionary of the first sorted unique-value of all categorical variables."""
    default_col_sortedval_dict = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col_name in categorical_cols:
        default_col_sortedval_dict[col_name] = sorted(df[col_name].unique())[0]

    return default_col_sortedval_dict


@jit(nopython=True)
def multiply(array_a, array_b):
    """Multiplies two matrices."""
    return array_a.T @ array_b
