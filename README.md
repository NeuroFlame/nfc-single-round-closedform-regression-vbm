### Computation Description

#### Overview
This computation performs a closed (or normal) form regression using on the datasets with VBM modality from multiple sites using specified covariates and dependent variables. This computation is designed to run within a federated learning environment, where each site performs a local regression analysis, and then global results are aggregated.


#### Example
   ```json
      {
          "Covariates": {
              "sex":"str",
              "isControl":"bool",
              "age":"float"
          },
          "Lambda": 1,
          "Threshold": 0.2,
          "VoxelSize":  4,
          "ReferenceColumns": { "site": "IA" },
          "IgnoreSubjectsWithMissingData" : false
      }
   ```
#### Settings Specification

The computation requires the user to provide a directory path containing "covariates.csv" file as input along with the above example parameter settings.

Below are the specifications for the parameters:

DEFAULT_Threshold = 0.2
DEFAULT_VoxelSize = 4
DEFAULT_ReferenceColumns = {}

| Variable Name                | Type    | Description                                                                                                                                                                        | Allowed Options           | Default | Required |
|------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|---------|---------|
| Covariates                   | dict    | Provide all the covariates that need to be considered for regression along with their type as shown in the example above.                                                          | dict                      | -       | ✅ Yes   |
| Lambda                       | float   | Penalty weight that is applied to all variables in the model during regression. If 0, perform simple linear regression, otherwise it does ridge regression.                        | any value between 0 and 1 | 0       | ❌ No      |
| Threshold                    | float   | This parameter is used in computing average mask. Voxel with values > threshold will only be included in the analysis.                                                             | any value > 0             | 0.2     | ❌ No      |
| VoxelSize                    | int     | Number of voxels used for subsampling/downsampling                                                                                                                                 | any value >= 1            | 1       | ❌ No      |
| ReferenceColumns             | dict    | Provide covariates with default value to be used as referencing in dummy encoding for categorical covariates. By default uses the first sorted value as per pandas dummy encoding. | dict                      | {}      |❌ No   |
| IgnoreSubjectsWithMissingData | boolean | This parameter lets the computation owner to decide how to handle if the data has missing or empty values.                                                                         | true or false             | False   | ❌ No      |

Below is the description of the csv file that is required along with the nifti files:
1. **Covariates File (`covariates.csv`)**

This file must follow a consistent format consisting "**niftifilename**" column having the nifti file name alogn with the other covariate columns. The computation expects these files to match the covariate names specified in the `parameters.json` file.

##### Covariates File (`covariates.csv`)

- **Format**: CSV (Comma-Separated Values)
- **Headers**: The file must include a header row where each column name corresponds to a covariate specified in the `parameters.json` and also have a column with "**niftifilename**".
- **Rows**: Each row represents a subject, where each column contains the value for a specific covariate.
- **Variable Names**: The names of the covariates in the header must match the entries in the `"Covariates"` section of the `parameters.json`.

**General Structure**:
```csv
niftifilename, <Covariate_1>,<Covariate_2>,...,<Covariate_N>
<file_1>,<value_1>,<value_2>,...,<value_N>
<file_1>,<value_1>,<value_2>,...,<value_N>
...
```
---

#### Algorithm Description

The key steps of the algorithm include:

1. **Local Ridge Regression (per site)**:
   - Each site runs regression on its local data, standardizing the covariates (X) and regressing against the voxels (Y).  It computes XtX and XtY matrices and send it to remote to compute aggregated model.
   - Statistical metrics (e.g., t-values, p-values, R-squared) are calculated using an ordinary least squares (OLS) ridge regression model to provide interpretability.

2. **Global Aggregation (controller)**:
   - After each site computes its local regression results, the controller computes the aggregated model by using normal form on the combined XtX and XtY matrices and  computes statistical metrics based on the number of subjects (degrees of freedom) per site.

#### Assumptions
- The data path provided has all the nifti files along with the covariates.csv file.
- The covariates.csv provided by each site follows the specified format (standardized covariate).
- The computation is run in a federated environment, and each site contributes valid data.

#### Output Description

- **Output files: global_stats and local stats with .png and .nii files**
- The global_stats has only global results for all the metrics and local_stats has local results corresponding to that site. 
 
The computation outputs both **site-level** and **global-level** results, which include:
- **Coefficients**: beta values for each covariate.
- **P-Values**: Probability values indicating significance for each covariate.
- **R-Squared**: The proportion of variance explained by the model for each covariate.


#### Running this computation
To locally run this computation, please clone this repo and run the following:

1. _./dockerRun.sh_ (This opens a bash terminal inside the docker container. Run the following commands inside the container terminal.)

2. _python makeJob.py site1,site2_ 

3. _nvflare simulator ./job_

#### Debug this computation using IDE
To debug the code using IDE, run "debugger.py" script and have the following debug configurations:

1. **PARAMETERS value :** "./app -w ./myworkspace -n 1 -c site1"  #change number of sites based on testing

2. **Environment Variables value:** "PYTHONUNBUFFERED=1;PYTHONPATH=<your_local_full_path>/neuroflame_comp_single_round_closed_form_regression_vbm/app/code"
