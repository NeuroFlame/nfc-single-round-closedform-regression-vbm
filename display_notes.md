### Overview

This computation performs a closed (or normal) form regression using on the datasets with VBM modality from multiple sites using specified covariates and dependent variables. This computation is designed to run within a federated learning environment, where each site performs a local regression analysis, and then global results are aggregated.

### Example Settings

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
       "IgnoreSubjectsWithMissingData" : false,
       "ShareLocalStats": false,
       "OutputImageDPI": 144
   }
```

### Settings Specification

| Variable Name | Type | Description | Allowed Options | Default | Required |
| --- | --- | --- | --- | --- | --- |
| `Covariates` | `dict` | Provide all the covariates that need to be considered for regression along with their type as shown in the example above. | dict | - | ✅ true |
| `Lambda` | `float` | Penalty weight that is applied to all variables in the model during regression. If 0, perform simple linear regression, otherwise it does ridge regression. | any value between 0 and 1 | 0 | ❌ false |
| `Threshold` | `float` | This parameter is used in computing average mask. Voxel with values > threshold will only be included in the analysis. | any value > 0 | 0.2 | ❌ false |
| `VoxelSize` | `int` | Number of voxels used for subsampling/downsampling | any value >= 1 | 1 | ❌ false |
| `ReferenceColumns` | `dict` | Provide covariates with default value to be used as referencing in dummy encoding for categorical covariates. By default uses the first sorted value as per pandas dummy encoding. | dict | {} | ❌ false |
| `IgnoreSubjectsWithMissingData` | `boolean` | This parameter lets the computation owner to decide how to handle if the data has missing or empty values. | true or false | false | ❌ false |
| `ShareLocalStats` | `boolean` | When enabled, each site's local beta, p-value, t-value and R² maps are sent through the aggregator and broadcast to all participating sites. This allows every site's report to display per-site brain images and forest plot entries for every site rather than only its own. **Note:** enabling this means voxelwise local model outputs cross site boundaries — only enable when all sites have agreed to share their results with each other. | true or false | false | ❌ false |
| `OutputImageDPI` | `int` | Resolution (dots per inch) of the rendered brain map PNGs. Higher values produce sharper, larger images at the cost of longer render time per image — this cost multiplies with smaller `VoxelSize` and with the number of sites when `ShareLocalStats` is enabled. | any value > 0 | 144 | ❌ false |

### Input Description

The computation requires **a data path** to be provided **having all the nifti files** along with the **covariates.csv** file.

1.  **Covariates File (**`covariates.csv`**)** 
    

This file must follow a consistent format consisting "**niftifilename**" column having the nifti file name alogn with the other covariate columns. The computation expects these files to match the covariate names specified in the `parameters.json` file.

**Covariates File (**`covariates.csv`**)**

*   **Format**: CSV (Comma-Separated Values)
    
*   **Headers**: The file must include a header row where each column name corresponds to a covariate specified in the `parameters.json` and also have a column with "**niftifilename**".
    
*   **Rows**: Each row represents a subject, where each column contains the value for a specific covariate.
    
*   **Variable Names**: The names of the covariates in the header must match the entries in the `"Covariates"` section of the `parameters.json`.
    

**General Structure**:

    niftifilename, <Covariate_1>,<Covariate_2>,...,<Covariate_N>
    <file_1>,<value_1>,<value_2>,...,<value_N>
    <file_1>,<value_1>,<value_2>,...,<value_N>
    ...

### Algorithm Description

The key steps of the algorithm include:

1.  **Local Regression (per site)**:
    
    *   Each site runs regression on its local data, standardizing the covariates (X) and regressing against the voxels (Y). It computes XtX (X transpose \* X)and XtY matrices (X transpose \* Y) and send it to remote to compute aggregated model.
        
    *   Statistical metrics (e.g., t-values, p-values, R-squared) are calculated using an ordinary least squares (OLS) ridge regression model to provide interpretability.
        
2.  **Global Aggregation (controller)**:
    
    *   After each site computes its local regression results, the controller computes the aggregated model by using normal form on the combined XtX (X transpose \* X)and XtY matrices (X transpose \* Y) matrices and computes statistical metrics based on the number of subjects (degrees of freedom) per site.

### Assumptions

*   The data path provided has all the nifti files along with the covariates.csv file.
    
*   The covariates.csv provided by each site follows the specified format (standardized covariate).
    
*   The computation is run in a federated environment, and each site contributes valid data.

### Output Description

*   **Output files: global\_stats and local\_stats directories containing .png and .nii files**

*   `global_stats/` — global federated results for all metrics, identical across all sites.

*   `local_stats/` — local results for the site that generated the report.

*   `local_stats_<SiteName>/` — per-site local results for every participating site. Only present when `ShareLocalStats` is set to `true`.

*   `index.html` — interactive HTML report with tabbed brain image viewer (Global + per-site tabs when `ShareLocalStats` is enabled), covariate summary table, significance bar chart, R² histogram, forest plots, and omitted subjects table.

The computation outputs both **site-level** and **global-level** results, which include:

*   **Coefficients**: beta values for each covariate.
    
*   **Signed Significance**: `-log10(p-value)` maps signed by the direction of the t-value, indicating significance for each covariate.
    
*   **R-Squared**: The proportion of variance explained by the model for each covariate.
