import numpy as np
import pandas as pd
import statsmodels.api as sm

import dominate
from dominate.tags import *

from utils.ancillary import *
from . import client_input_preprocessor as cip

def perform_client_step1(covariates_path, data_path, computation_parameters, log_path, cache_dict):
    # Validate the run inputs (covariates, dependent data, and parameters)
    is_valid, covariates, data = cip.validate_and_get_inputs(covariates_path, data_path, computation_parameters, log_path)
    if not is_valid:
        # Halt execution if validation fails
        raise ValueError(f"Invalid run input. Check validation log at {log_path}")

    # TODO .... FILL LOGIC

    results = {'output': output, 'cache':cache_dict}

    return results

