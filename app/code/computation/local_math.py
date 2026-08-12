"""Preserve the historical site-side VBM regression and report mathematics."""

import glob
import os.path
import shutil

import numpy as np
import pandas as pd
from numba import jit

from . import binary_io as dt
from . import constants as client_constants

# from ..utils import data_transfer as dt
from . import input_validation as cip
from . import nifti as cnh


def perform_client_step1_validate_inputs_and_compute_mask(
    covariates_path, data_path, computation_parameters, logger, cache_dict, cache_dir
):
    """Validate covariates and create the local average NIfTI.

    determines the categorical variables, computes the average of all nifti files and send them to the aggregator.

    :param covariates_path: user path provided for the covariates.csv file
    :param data_path: user path provided that has all the nifti files for the subjects in covariates.csv
    :param computation_parameters: input parameters
    :param logger: logger to save logs
    :param cache_dict: client cache dict
    :param cache_dir: client cache dir
    """
    logger.info(f"Computation parameters received: {computation_parameters}")

    # Validate the run inputs (covariates, dependent data, and parameters)
    is_valid, covariates_X, omissions_report = cip.validate_and_get_inputs(
        covariates_path, data_path, computation_parameters, logger
    )
    if not is_valid:
        # Halt execution if validation fails
        raise ValueError(
            "Invalid run input; see the participant's local computation log"
        )

    categorical_dict = {
        col: list(covariates_X[col].unique())
        for col in covariates_X.select_dtypes(include=object)
    }
    # converting all the values of the reference columns to lowercase as
    # all the covariate values are converted to lowercase while parsing
    reference_dict = dict(
        (k, v.lower())
        for k, v in computation_parameters.get(
            "ReferenceColumns", client_constants.DEFAULT_ReferenceColumns
        ).items()
    )

    # Handle dependent nifti files
    threshold = computation_parameters.get(
        "Threshold", client_constants.DEFAULT_Threshold
    )
    voxel_size = computation_parameters.get(
        "VoxelSize", client_constants.DEFAULT_VoxelSize
    )
    lamb = computation_parameters.get("Lambda", client_constants.DEFAULT_Lambda)

    # TODO: voxel size cannot be 0 (check how to handle this)
    if voxel_size == 0:
        voxel_size = 1

    # create a folder inside cache_dir that holds files to transfer
    data_transfer_dir = os.path.join(cache_dir, "transfer_client_step1")
    os.makedirs(data_transfer_dir, exist_ok=True)

    average_dependent_filename = cnh.average_nifti(
        covariates_X, data_path, data_transfer_dir
    )

    # Output
    output_dict = {
        "categorical_dict": categorical_dict,
        "reference_columns": reference_dict,
        "threshold": threshold,
        "voxel_size": voxel_size,
    }

    local_cache_dict = {
        "X": covariates_X.to_json(orient="split"),
        "voxel_size": voxel_size,
        "lambda": lamb,
        "computation_parameters": computation_parameters,
        "omissions_report": omissions_report,
    }

    results = {
        "output": output_dict,
        "cache": local_cache_dict,
        "transfer_files": {"average_nifti": average_dependent_filename},
    }

    return results


def perform_local_step2_compute_local_model(
    agg_result, data_path, logger, cache_dict, cache_dir
):
    """Fit the local voxelwise regression model.

    1. Loads nifti data for each subject and downsampled it to the computed aggregated mask
    2. computes local regression model
    3. Updates site as a covariate in X
    4. Computes XtransposeX and XtransposeY matrices and sends them to aggregator along with other model parameters.

    :param agg_result: results received from aggregator
    :param data_path: user path provided that has all the nifti files for the subjects in covariates.csv
    :param logger: logger to save logs
    :param cache_dict: client cache dict
    :param cache_dir: client cache dir
    """
    # Save streamed mask and template files in the current local staging area.
    mask_file_name = os.path.join(cache_dir, "mask.nii")
    shutil.copy2(agg_result["mask_file"].path, mask_file_name)

    downsampled_mni_file_name = os.path.join(cache_dir, "downsampled_mni.nii")
    shutil.copy2(agg_result["downsampled_mni"].path, downsampled_mni_file_name)

    X = pd.read_json(cache_dict["X"], orient="split")

    """The second function in the local computation chain"""
    regularizer_l2 = cache_dict["lambda"]

    # Local Statistics
    y = cnh.nifti_to_data(
        data_path, cache_dict, X, mask_file_name, downsampled_mni_file_name
    )

    encoded_X = cip.perform_encoding(agg_result, X)

    meanY_vector, lenY_vector = cip.mean_and_len_y(y)
    local_beta_vector, local_pvalues, local_tvalues, local_rsquared, local_X_labels = (
        _local_stats_to_dict_numba(encoded_X, y)
    )

    # Global Statistics
    augmented_X = cip.add_site_covariates(agg_result, X)
    X_labels = list(augmented_X.columns)
    biased_X = augmented_X.values.astype("float64")

    XtransposeX_local = cip.multiply(biased_X, biased_X)
    Xtransposey_local = cip.multiply(biased_X, y)

    # Writing covariates and dependents to cache as files
    dt.save_binary(os.path.join(cache_dir, "X.npy"), biased_X)
    dt.save_binary(os.path.join(cache_dir, "y.npy"), y)

    x_transpose_x_file_name = os.path.join(cache_dir, "XTX.npy")
    dt.save_binary(x_transpose_x_file_name, XtransposeX_local)

    x_transpose_y_file_name = os.path.join(cache_dir, "XTy.npy")
    dt.save_binary(x_transpose_y_file_name, Xtransposey_local)

    output_dict = {
        "mean_y_local": meanY_vector.tolist(),
        "count_local": lenY_vector.tolist(),
        "X_labels": X_labels,
        "lambda": regularizer_l2,
    }

    computation_parameters = cache_dict.get("computation_parameters", {})
    if computation_parameters.get(
        "ShareLocalStats", client_constants.DEFAULT_ShareLocalStats
    ):
        output_dict["share_local_stats"] = True
        output_dict["local_beta_vector"] = local_beta_vector.tolist()
        output_dict["local_pvalues"] = local_pvalues.tolist()
        output_dict["local_tvalues"] = local_tvalues.tolist()
        output_dict["local_rsquared"] = local_rsquared.tolist()
        output_dict["local_X_labels"] = local_X_labels

    cache_dict = {
        "covariates": "X.npy",
        "dependents": "y.npy",
        "mask_file_name": mask_file_name,
        "downsampled_mni_file_name": downsampled_mni_file_name,
        "reference_columns": agg_result["reference_columns"],
        "local_beta_vector": local_beta_vector.tolist(),
        "local_pvalues": local_pvalues.tolist(),
        "local_tvalues": local_tvalues.tolist(),
        "local_rsquared": local_rsquared.tolist(),
        "local_X_labels": local_X_labels,
    }
    result = {
        "output": output_dict,
        "cache": cache_dict,
        "transfer_files": {
            "x_transpose_x_file_name": x_transpose_x_file_name,
            "x_transpose_y_file_name": x_transpose_y_file_name,
        },
    }

    return result


def perform_local_step3_compute_metrics_with_global_params(
    agg_result, logger, cache_dict, cache_dir
):
    """Computes the SSE_local, SST_local and varX_matrix_local based on the global aggregated parameters.

    :param agg_result: results received from aggregator
    :param logger: logger to save logs
    :param cache_dict: client cache dict
    :param cache_dir: client cache dir

    """
    # save avg_beta_vector and mean_y_global files from aggregator in cache_dir
    avg_beta_vector_file_name = os.path.join(cache_dir, "avg_beta_vector.npy")
    shutil.copy2(agg_result["avg_beta_vector"].path, avg_beta_vector_file_name)

    mean_y_global_file_name = os.path.join(cache_dir, "mean_y_global.npy")
    shutil.copy2(agg_result["mean_y_global"].path, mean_y_global_file_name)

    biased_X = dt.load_binary(os.path.join(cache_dir, cache_dict["covariates"]))
    y = dt.load_binary(os.path.join(cache_dir, cache_dict["dependents"]))

    avg_beta_vector = dt.load_binary(avg_beta_vector_file_name)
    mean_y_global = dt.load_binary(mean_y_global_file_name)

    varX_matrix_local = cip.multiply(biased_X, biased_X)

    sse_local, sst_local = _stats_calculation(
        biased_X, y, avg_beta_vector, mean_y_global
    )

    output_dict = {
        "SSE_local": sse_local.tolist(),
        "SST_local": sst_local.tolist(),
        "varX_matrix_local": varX_matrix_local.tolist(),
    }

    cache_dict = {
        "avg_beta_vector_file_name": avg_beta_vector_file_name,
        "mean_y_global_file_name": mean_y_global_file_name,
    }
    result = {"output": output_dict, "cache": cache_dict}

    return result


def perform_local_step4_persist_results(
    agg_result, logger, cache_dict, cache_dir, output_dir
):
    """Persists results in various file formats.

    :param agg_result: results received from aggregator
    :param logger: logger to save logs
    :param cache_dict: client cache dict
    :param cache_dir: client cache dir
    :param output_dir: output directory which saves all the regression .png and .nii files
    """
    avg_beta_vector = dt.load_binary(cache_dict["avg_beta_vector_file_name"])
    r_squared_global = np.asarray(agg_result["r_squared_global"])
    ts_global = np.asarray(agg_result["ts_global"])
    ps_global = np.asarray(agg_result["ps_global"])
    X_labels = np.asarray(agg_result["X_labels"])
    np.asarray(agg_result["MSE"])

    # Save global output files
    global_output_dir = os.path.join(output_dir, "global_stats")
    os.makedirs(global_output_dir, exist_ok=True)
    cnh.print_pvals(
        cache_dict, cache_dir, global_output_dir, ps_global, ts_global, X_labels
    )
    cnh.print_beta_images(
        cache_dict, cache_dir, global_output_dir, avg_beta_vector, X_labels
    )
    cnh.print_r2_image(cache_dict, cache_dir, global_output_dir, r_squared_global)

    # Save local output files
    local_output_dir = os.path.join(output_dir, "local_stats")
    os.makedirs(local_output_dir, exist_ok=True)
    local_X_labels = cache_dict["local_X_labels"]

    cnh.print_pvals(
        cache_dict,
        cache_dir,
        local_output_dir,
        np.asarray(cache_dict["local_pvalues"]).T,
        np.asarray(cache_dict["local_tvalues"]).T,
        local_X_labels,
    )
    cnh.print_beta_images(
        cache_dict,
        cache_dir,
        local_output_dir,
        np.asarray(cache_dict["local_beta_vector"]),
        local_X_labels,
    )
    cnh.print_r2_image(
        cache_dict,
        cache_dir,
        local_output_dir,
        np.asarray(cache_dict["local_rsquared"]),
    )

    shutil.copy2(cache_dict["mask_file_name"], os.path.join(output_dir, "mask.nii"))

    computation_parameters = cache_dict.get("computation_parameters", {})
    site_id_name_map = computation_parameters.get("site_id_name_map", {})

    # Generate per-site local stats images when the aggregator broadcast them
    site_stats = {}
    shared_local_stats = agg_result.get("shared_local_stats", {})
    for site_name, stats in shared_local_stats.items():
        site_output_dir = os.path.join(
            output_dir, f"local_stats_{site_name.replace(' ', '_')}"
        )
        os.makedirs(site_output_dir, exist_ok=True)
        site_local_X_labels = stats["local_X_labels"]
        cnh.print_pvals(
            cache_dict,
            cache_dir,
            site_output_dir,
            np.asarray(stats["local_pvalues"]).T,
            np.asarray(stats["local_tvalues"]).T,
            site_local_X_labels,
        )
        cnh.print_beta_images(
            cache_dict,
            cache_dir,
            site_output_dir,
            np.asarray(stats["local_beta_vector"]),
            site_local_X_labels,
        )
        cnh.print_r2_image(
            cache_dict, cache_dir, site_output_dir, np.asarray(stats["local_rsquared"])
        )
        site_stats[site_name] = glob.glob(f"{site_output_dir}/*.png")

    # Covariate summary stats from cached X
    X = pd.read_json(cache_dict["X"], orient="split")
    covariate_summary = _compute_covariate_summary(X, computation_parameters)

    # Significance counts per covariate (p < 0.05 uncorrected)
    sig_counts = _compute_sig_voxel_counts(
        np.asarray(agg_result["ps_global"]), X_labels
    )

    # R² histogram data
    r2_hist_data = _compute_r2_histogram(r_squared_global)

    # Forest plot data: global betas vs local betas per covariate
    forest_data = _compute_forest_data(
        avg_beta_vector,
        np.asarray(agg_result["ts_global"]),
        np.asarray(agg_result["MSE"]),
        np.asarray(cache_dict["local_beta_vector"]),
        np.asarray(cache_dict["local_tvalues"]),
        X_labels,
        local_X_labels,
        site_id_name_map,
        shared_local_stats=shared_local_stats if shared_local_stats else None,
    )

    # Gather local and global png files (for display in UI)
    results_dict = {
        "local_stats": glob.glob(f"{local_output_dir}/*.png"),
        "global_stats": glob.glob(f"{global_output_dir}/*.png"),
        "site_stats": site_stats,
        "covariate_summary": covariate_summary,
        "sig_counts": sig_counts,
        "r2_hist_data": r2_hist_data,
        "forest_data": forest_data,
        "omissions_report": cache_dict.get("omissions_report", {}),
    }

    result = {
        "output": {
            "html": _persist_html_safe_wrapper(
                results_dict, os.path.join(output_dir, "index.html"), cache_dict
            )
        },
        "cache": {},
    }

    return result


def _compute_covariate_summary(X, computation_parameters):
    """Returns per-covariate summary statistics for the report table."""
    covariate_info = computation_parameters.get("Covariates", {})
    rows = []
    for col in X.columns:
        dtype = covariate_info.get(col, "").strip().lower()
        entry = {"name": col, "dtype": dtype, "n": int(X[col].notna().sum())}
        if dtype in ("int", "float"):
            entry["mean"] = float(round(X[col].mean(), 4))
            entry["std"] = float(round(X[col].std(), 4))
            entry["min"] = float(round(X[col].min(), 4))
            entry["max"] = float(round(X[col].max(), 4))
        else:
            counts = X[col].value_counts().to_dict()
            entry["counts"] = {str(k): int(v) for k, v in counts.items()}
        rows.append(entry)
    return rows


def _compute_sig_voxel_counts(ps_global, X_labels, threshold=0.05):
    """Returns {label, count, pct} for each covariate at p < threshold."""
    result = []
    if ps_global.ndim == 1:
        ps_global = ps_global[np.newaxis, :]
    n_vox = ps_global.shape[1]
    for i, label in enumerate(list(X_labels)):
        if i >= ps_global.shape[0]:
            break
        count = int(np.sum(ps_global[i] < threshold))
        result.append(
            {
                "label": str(label),
                "count": count,
                "pct": round(100.0 * count / n_vox, 2) if n_vox > 0 else 0.0,
            }
        )
    return result


def _compute_r2_histogram(r_squared_global, n_bins=20):
    """Returns histogram bin edges and counts for R² values."""
    r2 = np.asarray(r_squared_global).ravel()
    r2 = r2[np.isfinite(r2)]
    counts, edges = np.histogram(r2, bins=n_bins, range=(0.0, 1.0))
    return {
        "counts": counts.tolist(),
        "edges": [round(float(e), 4) for e in edges],
        "mean": round(float(np.mean(r2)), 4),
        "median": round(float(np.median(r2)), 4),
    }


def _compute_forest_data(
    avg_beta_vector,
    ts_global,
    MSE_global,
    local_beta_vector,
    local_tvalues,
    X_labels,
    local_X_labels,
    site_id_name_map,
    shared_local_stats=None,
):
    """Build per-covariate forest plot data using voxel means.

    Returns list of {covariate, global_beta, global_se, sites:[{name,beta,se}]}
    When shared_local_stats is provided (ShareLocalStats=true), each site gets its
    own entry using its actual local betas. Otherwise a single "Local" entry is shown.
    """
    result = []
    global_labels = list(X_labels)
    local_labels = list(local_X_labels)

    def _orient(arr, n_cov):
        """Ensure shape is (n_vox, n_cov)."""
        a = np.asarray(arr)
        if a.ndim == 2 and a.shape[0] == n_cov and a.shape[0] != a.shape[1]:
            return a.T
        return a

    n_gcov = len(global_labels)
    gb = _orient(avg_beta_vector, n_gcov)
    tsg = _orient(ts_global, n_gcov)
    mse = np.asarray(MSE_global).ravel()

    n_lcov = len(local_labels)
    lb = _orient(local_beta_vector, n_lcov)
    ltv = _orient(local_tvalues, n_lcov)

    for i, label in enumerate(global_labels):
        if label == "const":
            continue
        if gb.ndim != 2 or i >= gb.shape[1]:
            continue
        g_beta = float(np.nanmean(gb[:, i]))
        denom = (
            np.abs(tsg[:, i]) + 1e-9
            if tsg.ndim == 2 and i < tsg.shape[1]
            else np.ones(len(mse))
        )
        g_se = float(np.nanmean(np.sqrt(np.maximum(mse, 0)) / denom))

        sites_data = []
        if shared_local_stats:
            for site_name, stats in shared_local_stats.items():
                s_labels = list(stats["local_X_labels"])
                if label not in s_labels:
                    continue
                li = s_labels.index(label)
                s_lb = _orient(np.asarray(stats["local_beta_vector"]), len(s_labels))
                s_ltv = _orient(np.asarray(stats["local_tvalues"]), len(s_labels))
                if s_lb.ndim == 2 and li < s_lb.shape[1]:
                    l_beta = float(np.nanmean(s_lb[:, li]))
                    l_denom = (
                        np.abs(s_ltv[:, li]) + 1e-9
                        if s_ltv.ndim == 2 and li < s_ltv.shape[1]
                        else np.ones(s_lb.shape[0])
                    )
                    l_se = float(
                        np.nanmean(
                            np.sqrt(np.maximum(mse[: s_lb.shape[0]], 0)) / l_denom
                        )
                    )
                    sites_data.append(
                        {
                            "name": site_name,
                            "beta": round(l_beta, 6),
                            "se": round(l_se, 6),
                        }
                    )
        else:
            if label in local_labels:
                li = local_labels.index(label)
                if lb.ndim == 2 and li < lb.shape[1]:
                    l_beta = float(np.nanmean(lb[:, li]))
                    l_denom = (
                        np.abs(ltv[:, li]) + 1e-9
                        if ltv.ndim == 2 and li < ltv.shape[1]
                        else np.ones(lb.shape[0])
                    )
                    l_se = float(
                        np.nanmean(np.sqrt(np.maximum(mse[: lb.shape[0]], 0)) / l_denom)
                    )
                    sites_data.append(
                        {
                            "name": "Local",
                            "beta": round(l_beta, 6),
                            "se": round(l_se, 6),
                        }
                    )

        result.append(
            {
                "covariate": label,
                "global_beta": round(g_beta, 6),
                "global_se": round(g_se, 6),
                "sites": sites_data,
            }
        )
    return result


def _persist_html_from_results(results_dict, output_file_name, cache_dict=None):
    """Generate the styled HTML VBM regression report.

    Features:
      - Tab switcher: Global / All Local / per-site tabs
      - Per-card image cycling across sources for the same covariate
      - Covariate summary table
      - Significant voxel count bar chart (inline SVG)
      - R² histogram (inline SVG)
      - Forest plots per covariate (inline SVG).
    """
    import json
    import re

    if cache_dict is None:
        cache_dict = {}

    computation_parameters = cache_dict.get("computation_parameters", {})
    site_id_name_map = computation_parameters.get("site_id_name_map", {})

    _SITE_COLORS_SOLID = [
        "rgba(99,102,241,1.0)",
        "rgba(20,184,166,1.0)",
        "rgba(245,158,11,1.0)",
        "rgba(239,68,68,1.0)",
        "rgba(168,85,247,1.0)",
        "rgba(34,197,94,1.0)",
    ]
    _SITE_HEX = ["#6366f1", "#14b8a6", "#f59e0b", "#ef4444", "#a855f7", "#22c55e"]

    # site_id_name_map is only populated when a run goes through NeuroFLAME's
    # provisioning step; local/simulator runs leave it empty even though
    # site_stats may still hold per-site data keyed by raw site id. Fall back
    # to those keys so per-site tabs still render.
    all_sites = sorted(
        set(site_id_name_map.values()) | set(results_dict.get("site_stats", {}).keys())
    )
    solid_map = {
        s: _SITE_COLORS_SOLID[i % len(_SITE_COLORS_SOLID)]
        for i, s in enumerate(all_sites)
    }
    hex_map = {s: _SITE_HEX[i % len(_SITE_HEX)] for i, s in enumerate(all_sites)}

    def nav_slug(t):
        s = (
            t.lower()
            .replace("(", "")
            .replace(")", "")
            .replace("&", "")
            .replace("/", "")
        )
        s = s.replace("²", "2").replace("³", "3")
        return re.sub(r"-+", "-", s.replace(" ", "-")).strip("-")

    def img_to_src(path, out_file):
        try:
            return os.path.relpath(path, os.path.dirname(out_file))
        except Exception:
            return path

    def nii_viewer_src(png_path, out_file):
        """Return relative path to the .nii file for direct linking into NeuroFLAME's viewer."""
        try:
            nii_path = os.path.splitext(png_path)[0] + ".nii"
            if not os.path.exists(nii_path):
                return None
            return os.path.relpath(nii_path, os.path.dirname(out_file))
        except Exception:
            return None

    def section(title, content, slug):
        return (
            f'<div class="container"><div class="section" id="sec-{slug}">'
            f'<div class="section-title">{title}</div>{content}</div></div>'
        )

    n_sites = len(all_sites)
    n_covariates = len(computation_parameters.get("Covariates", {}))
    voxel_size = cache_dict.get("voxel_size", "—")
    lamb = cache_dict.get("lambda", "—")

    # ── Header ───────────────────────────────────────────────────────────────
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:.45rem;font-size:.83rem;color:var(--legend-color)">'
        f'<div style="width:11px;height:11px;border-radius:3px;background:{solid_map[s]}"></div>{s}</div>'
        for s in all_sites
    )
    header = f"""<div class="page-header" id="sec-study-overview">
  <h1>Single Round Closedform Regression VBM</h1>
  <p>Federated voxelwise ridge regression across {n_sites} site{"s" if n_sites != 1 else ""}</p>
  <div class="chips">
    <div class="chip">Sites <b>{n_sites}</b></div>
    <div class="chip">Covariates <b>{n_covariates}</b></div>
    <div class="chip">Voxel size <b>{voxel_size}</b></div>
    <div class="chip">&lambda; (Ridge) <b>{lamb}</b></div>
  </div>
  {f'<div class="site-legend" style="margin-top:.75rem">{legend}</div>' if all_sites else ""}
</div>"""

    # ── Build per-covariate image map across all sources ──────────────────────
    covariate_images = {}
    for img_path in sorted(results_dict.get("global_stats", [])):
        label = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ")
        covariate_images.setdefault(label, []).append(
            {
                "source": "Global",
                "src": img_to_src(img_path, output_file_name),
                "nii_viewer": nii_viewer_src(img_path, output_file_name),
            }
        )
    for site_name, img_paths in sorted(results_dict.get("site_stats", {}).items()):
        for img_path in sorted(img_paths):
            label = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ")
            covariate_images.setdefault(label, []).append(
                {
                    "source": site_name,
                    "src": img_to_src(img_path, output_file_name),
                    "nii_viewer": nii_viewer_src(img_path, output_file_name),
                }
            )

    source_order = ["Global"] + sorted(all_sites)
    all_sources, seen = [], set()
    for s in source_order:
        if s not in seen:
            all_sources.append(s)
            seen.add(s)

    covariate_images_json = json.dumps(covariate_images)

    # Tab bar
    tab_buttons = ""
    for idx, src in enumerate(all_sources):
        active_cls = " tab-active" if idx == 0 else ""
        color = (
            hex_map.get(src, "#6366f1")
            if src not in ("Global", "All Local")
            else "#6366f1"
        )
        dot = (
            f'<span style="width:8px;height:8px;border-radius:50%;background:{color};'
            f'flex-shrink:0;display:inline-block;margin-right:.35rem"></span>'
            if src not in ("Global", "All Local")
            else ""
        )
        tab_buttons += (
            f'<button class="tab-btn{active_cls}" data-source="{src}" onclick="selectTab(this)">'
            f"{dot}{src}</button>"
        )
    tab_bar = f'<div class="tab-bar">{tab_buttons}</div>'

    # Image cards — all frames rendered, JS shows/hides by active source
    all_labels = sorted(covariate_images.keys())
    cards_html = ""
    for card_id, label in enumerate(all_labels):
        frames = covariate_images[label]
        frames_json = json.dumps(frames)
        imgs_html = ""
        nii_links_html = ""
        for fi, frame in enumerate(frames):
            display = "block" if fi == 0 else "none"
            imgs_html += (
                f'<img src="{frame["src"]}" data-source="{frame["source"]}" '
                f'data-frame="{fi}" style="width:100%;border-radius:6px;display:{display}" '
                f'alt="{label} — {frame["source"]}"/>'
            )
            if frame.get("nii_viewer"):
                viewer_url = frame["nii_viewer"]
                nii_links_html += (
                    f'<a data-nii-frame="{fi}" data-nii-rel="{viewer_url}" href="#" target="_top" '
                    f'style="display:{"inline-flex" if fi == 0 else "none"};align-items:center;gap:.3rem;'
                    f"font-size:.72rem;font-weight:600;color:#6366f1;text-decoration:none;"
                    f"padding:.2rem .55rem;border:1px solid rgba(99,102,241,.35);border-radius:5px;"
                    f'background:rgba(99,102,241,.07);transition:all .15s">'
                    f"&#x1F9E0; View NIfTI</a>"
                )
        n_frames = len(frames)
        cycle_controls = ""
        if n_frames > 1:
            cycle_controls = (
                f'<div class="cycle-controls">'
                f'<button class="cycle-btn" onclick="cycleCard(this,-1)">&#8249;</button>'
                f'<span class="cycle-label" data-card="{card_id}">1/{n_frames}</span>'
                f'<button class="cycle-btn" onclick="cycleCard(this,1)">&#8250;</button>'
                f"</div>"
            )
        nii_links_wrap = (
            (
                f'<div class="nii-links" data-card="{card_id}" '
                f'style="margin-top:.5rem;display:flex;gap:.4rem;flex-wrap:wrap">'
                f"{nii_links_html}</div>"
            )
            if nii_links_html
            else ""
        )
        cards_html += (
            f'<div class="img-card" data-card="{card_id}" data-frames=\'{frames_json}\' data-active-frame="0">'
            f'<div class="img-card-header">'
            f'<div class="img-card-title">{label}</div>{cycle_controls}</div>'
            f'<div class="source-badge" data-card="{card_id}">{frames[0]["source"] if frames else ""}</div>'
            f"{imgs_html}{nii_links_wrap}</div>"
        )

    image_section = section(
        "Brain images",
        tab_bar
        + (
            f'<div class="img-grid" id="imgGrid">{cards_html}</div>'
            if cards_html
            else '<p style="color:var(--text3);font-size:.85rem">No images generated.</p>'
        ),
        "brain-images",
    )

    # ── Covariate summary table ───────────────────────────────────────────────
    cov_summary = results_dict.get("covariate_summary", [])
    if cov_summary:
        thead = (
            "<tr><th>Covariate</th><th>Type</th><th>N</th>"
            "<th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Categories</th></tr>"
        )
        tbody = ""
        for row in cov_summary:
            if "mean" in row:
                tbody += (
                    f"<tr><td>{row['name']}</td>"
                    f'<td><span class="dtype-badge">{row["dtype"]}</span></td>'
                    f"<td>{row['n']}</td><td>{row['mean']}</td><td>{row['std']}</td>"
                    f"<td>{row['min']}</td><td>{row['max']}</td><td>—</td></tr>"
                )
            else:
                cats = ", ".join(f"{k} ({v})" for k, v in row.get("counts", {}).items())
                tbody += (
                    f"<tr><td>{row['name']}</td>"
                    f'<td><span class="dtype-badge">{row["dtype"]}</span></td>'
                    f"<td>{row['n']}</td>"
                    f"<td>—</td><td>—</td><td>—</td><td>—</td>"
                    f'<td style="font-size:.78rem">{cats}</td></tr>'
                )
        cov_html = (
            f'<div class="table-wrap"><table class="stats-table">'
            f"<thead>{thead}</thead><tbody>{tbody}</tbody></table></div>"
        )
    else:
        cov_html = '<p style="color:var(--text3);font-size:.85rem">No covariate data available.</p>'
    cov_section = section("Covariate summary", cov_html, "covariate-summary")

    # ── Significant voxels bar chart (inline SVG) ─────────────────────────────
    sig_counts = results_dict.get("sig_counts", [])
    if sig_counts:
        max_count = max(r["count"] for r in sig_counts) or 1
        bar_h, bar_gap, label_w, bar_max_w = 28, 10, 160, 420
        chart_h = len(sig_counts) * (bar_h + bar_gap) + 40
        bars_svg = ""
        for i, r in enumerate(sig_counts):
            y = 20 + i * (bar_h + bar_gap)
            bw = int(bar_max_w * r["count"] / max_count)
            bars_svg += (
                f'<text x="{label_w - 8}" y="{y + bar_h // 2 + 5}" text-anchor="end" '
                f'font-size="12" fill="var(--text2)">{r["label"]}</text>'
                f'<rect x="{label_w}" y="{y}" width="{bw}" height="{bar_h}" rx="4" fill="#6366f1" opacity=".8"/>'
                f'<text x="{label_w + bw + 6}" y="{y + bar_h // 2 + 5}" font-size="11" fill="var(--text3)">'
                f"{r['count']:,} ({r['pct']}%)</text>"
            )
        sig_svg = (
            f'<svg viewBox="0 0 {label_w + bar_max_w + 140} {chart_h}" '
            f'xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:700px">{bars_svg}</svg>'
        )
        sig_content = (
            '<p style="font-size:.82rem;color:var(--text3);margin-bottom:1rem">'
            "Voxels significant at p &lt; 0.05 (uncorrected) per covariate</p>"
            + sig_svg
        )
    else:
        sig_content = '<p style="color:var(--text3);font-size:.85rem">No significance data available.</p>'
    sig_section = section(
        "Significant voxels per covariate",
        sig_content,
        "significant-voxels-per-covariate",
    )

    # ── R² histogram (inline SVG) ─────────────────────────────────────────────
    r2_data = results_dict.get("r2_hist_data", {})
    if r2_data and r2_data.get("counts"):
        counts = r2_data["counts"]
        edges = r2_data["edges"]
        max_c = max(counts) or 1
        svg_w, svg_h = 560, 200
        pad_l, pad_b, pad_t = 50, 40, 20
        plot_w = svg_w - pad_l - 20
        plot_h = svg_h - pad_b - pad_t
        n_bins = len(counts)
        bin_w = plot_w / n_bins
        bars = ""
        for i, c in enumerate(counts):
            bh = int(plot_h * c / max_c)
            bars += (
                f'<rect x="{pad_l + i * bin_w:.1f}" y="{pad_t + plot_h - bh}" '
                f'width="{bin_w - 1:.1f}" height="{bh}" rx="2" fill="#6366f1" opacity=".75"/>'
            )
        x_labels = "".join(
            f'<text x="{pad_l + i * bin_w:.1f}" y="{svg_h - 8}" text-anchor="middle" '
            f'font-size="11" fill="var(--text3)">{edges[i]:.1f}</text>'
            for i in range(0, n_bins + 1, max(1, n_bins // 5))
            if i < len(edges)
        )
        mean_x = pad_l + r2_data["mean"] * plot_w
        mean_line = (
            f'<line x1="{mean_x:.1f}" y1="{pad_t}" x2="{mean_x:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,3"/>'
            f'<text x="{mean_x + 4:.1f}" y="{pad_t + 14}" font-size="10" fill="#ef4444">'
            f"mean={r2_data['mean']}</text>"
        )
        r2_svg = (
            f'<svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;max-width:{svg_w}px">'
            f'<text x="12" y="{svg_h // 2}" text-anchor="middle" font-size="11" fill="var(--text3)" '
            f'transform="rotate(-90,12,{svg_h // 2})">Voxel count</text>'
            f"{bars}{mean_line}{x_labels}"
            f'<text x="{pad_l + plot_w // 2}" y="{svg_h - 24}" text-anchor="middle" '
            f'font-size="11" fill="var(--text3)">R\u00b2</text></svg>'
        )
        r2_content = (
            f'<p style="font-size:.82rem;color:var(--text3);margin-bottom:.75rem">'
            f"Distribution of voxelwise R\u00b2 &mdash; median: <b>{r2_data['median']}</b>, "
            f"mean: <b>{r2_data['mean']}</b></p>" + r2_svg
        )
    else:
        r2_content = '<p style="color:var(--text3);font-size:.85rem">No R\u00b2 data available.</p>'
    r2_section = section("R\u00b2 distribution", r2_content, "r2-distribution")

    # ── Forest plots (inline SVG, one per covariate) ──────────────────────────
    forest_data = results_dict.get("forest_data", [])
    if forest_data:
        fp_svgs = ""
        for fd in forest_data:
            rows = [
                {
                    "name": "Global (federated)",
                    "beta": fd["global_beta"],
                    "se": fd["global_se"],
                    "is_global": True,
                }
            ]
            for s in fd.get("sites", []):
                rows.append(
                    {
                        "name": s["name"],
                        "beta": s["beta"],
                        "se": s["se"],
                        "is_global": False,
                    }
                )

            all_betas = [r["beta"] for r in rows]
            all_ses = [r["se"] for r in rows]
            x_min = min(b - 2 * s for b, s in zip(all_betas, all_ses, strict=False))
            x_max = max(b + 2 * s for b, s in zip(all_betas, all_ses, strict=False))
            span = x_max - x_min or 0.002
            x_min -= span * 0.1
            x_max += span * 0.1

            fp_w, row_h, label_w = 520, 32, 160
            fp_h = len(rows) * row_h + 50
            plot_w = fp_w - label_w - 20

            def to_px(v, _x_min=x_min, _x_max=x_max, _lw=label_w, _pw=plot_w):
                return _lw + int(_pw * (v - _x_min) / (_x_max - _x_min))

            zero_x = to_px(0)
            fp_el = (
                f'<line x1="{zero_x}" y1="10" x2="{zero_x}" y2="{fp_h - 30}" '
                f'stroke="var(--border)" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            for ri, row in enumerate(rows):
                cy = 10 + ri * row_h + row_h // 2
                bx = to_px(row["beta"])
                lo = to_px(row["beta"] - 2 * row["se"])
                hi = to_px(row["beta"] + 2 * row["se"])
                color = (
                    "#6366f1"
                    if row["is_global"]
                    else hex_map.get(row["name"], "#14b8a6")
                )
                fw = "700" if row["is_global"] else "400"
                shape = (
                    f'<rect x="{bx - 6}" y="{cy - 6}" width="12" height="12" fill="{color}"/>'
                    if row["is_global"]
                    else f'<circle cx="{bx}" cy="{cy}" r="6" fill="{color}"/>'
                )
                fp_el += (
                    f'<text x="{label_w - 6}" y="{cy + 4}" text-anchor="end" font-size="11" '
                    f'fill="var(--text2)" font-weight="{fw}">{row["name"]}</text>'
                    f'<line x1="{lo}" y1="{cy}" x2="{hi}" y2="{cy}" stroke="{color}" stroke-width="2"/>'
                    f'<line x1="{lo}" y1="{cy - 5}" x2="{lo}" y2="{cy + 5}" stroke="{color}" stroke-width="2"/>'
                    f'<line x1="{hi}" y1="{cy - 5}" x2="{hi}" y2="{cy + 5}" stroke="{color}" stroke-width="2"/>'
                    f"{shape}"
                    f'<text x="{hi + 6}" y="{cy + 4}" font-size="10" fill="var(--text3)">{row["beta"]:.4f}</text>'
                )
            n_ticks = 5
            for ti in range(n_ticks + 1):
                tv = x_min + ti * (x_max - x_min) / n_ticks
                tx = label_w + int(plot_w * ti / n_ticks)
                fp_el += (
                    f'<line x1="{tx}" y1="{fp_h - 30}" x2="{tx}" y2="{fp_h - 24}" '
                    f'stroke="var(--border)" stroke-width="1"/>'
                    f'<text x="{tx}" y="{fp_h - 14}" text-anchor="middle" font-size="10" '
                    f'fill="var(--text3)">{tv:.3f}</text>'
                )
            fp_svgs += (
                f'<div style="margin-bottom:1.5rem">'
                f'<div style="font-size:.8rem;font-weight:600;color:var(--text2);margin-bottom:.4rem">'
                f"{fd['covariate']}</div>"
                f'<svg viewBox="0 0 {fp_w} {fp_h}" xmlns="http://www.w3.org/2000/svg" '
                f'style="width:100%;max-width:{fp_w}px">{fp_el}</svg></div>'
            )
        forest_content = (
            '<p style="font-size:.82rem;color:var(--text3);margin-bottom:1rem">'
            "Mean voxelwise beta \u00b1 2 SE averaged over brain. "
            "Square\u00a0=\u00a0global federated estimate; circle\u00a0=\u00a0local site estimate.</p>"
            + fp_svgs
        )
    else:
        forest_content = '<p style="color:var(--text3);font-size:.85rem">No forest plot data available.</p>'
    forest_section = section("Forest plots", forest_content, "forest-plots")

    # ── Omitted subjects section ──────────────────────────────────────────────
    omissions_report = results_dict.get("omissions_report", {})
    invalid_covariates = omissions_report.get("invalid_covariates", [])
    missing_nifti = omissions_report.get("missing_nifti", [])
    invalid_nifti = omissions_report.get("invalid_nifti", [])
    total_omitted = len(invalid_covariates) + len(missing_nifti) + len(invalid_nifti)

    if total_omitted > 0:

        def _omit_table(subjects, reason, color):
            if not subjects:
                return ""
            rows_html = "".join(
                f'<tr><td style="font-family:monospace;font-size:.8rem">{s}</td>'
                f'<td><span style="background:{color}22;color:{color};border-radius:4px;'
                f'padding:2px 8px;font-size:.76rem;font-weight:600">{reason}</span></td></tr>'
                for s in subjects
            )
            return rows_html

        omit_rows = (
            _omit_table(invalid_covariates, "Invalid/missing covariate", "#f59e0b")
            + _omit_table(missing_nifti, "NIfTI file not found", "#ef4444")
            + _omit_table(invalid_nifti, "NIfTI data invalid/empty", "#8b5cf6")
        )
        omit_html = (
            f'<p style="font-size:.82rem;color:var(--text3);margin-bottom:.9rem">'
            f"<b>{total_omitted}</b> subject{' was' if total_omitted == 1 else 's were'} excluded "
            f"before analysis because <code>IgnoreSubjectsWithMissingData</code> is enabled.</p>"
            f'<div class="table-wrap"><table class="stats-table">'
            f"<thead><tr><th>Subject (niftifilename)</th><th>Reason</th></tr></thead>"
            f"<tbody>{omit_rows}</tbody></table></div>"
        )
    else:
        omit_html = (
            '<p style="color:var(--text3);font-size:.85rem">'
            "No subjects were omitted — all passed validation.</p>"
        )
    omissions_section = section("Omitted subjects", omit_html, "omitted-subjects")

    # ── Nav / sidebar ─────────────────────────────────────────────────────────
    nav_titles = [
        "Study Overview",
        "Brain images",
        "Covariate summary",
        "Significant voxels per covariate",
        "R\u00b2 distribution",
        "Forest plots",
        "Omitted subjects",
    ]
    nav_items_html = "\n".join(
        '<button class="nav-item" data-sec="sec-' + nav_slug(t) + '" '
        "onclick=\"document.getElementById('sec-"
        + nav_slug(t)
        + "').scrollIntoView({behavior:'smooth',block:'start'})\">"
        + t
        + "</button>"
        for t in nav_titles
    )

    body = (
        header
        + image_section
        + cov_section
        + sig_section
        + r2_section
        + forest_section
        + omissions_section
    )
    sidebar_and_body = (
        '<div class="layout">'
        '<aside class="sidebar" id="sidebar"><div class="sidebar-inner">'
        '<div class="sidebar-header">'
        '<span class="sidebar-label">Sections</span>'
        '<button class="sidebar-toggle" onclick="toggleSidebar()">&#x2715;</button>'
        "</div>" + nav_items_html + "</div></aside>"
        '<button class="sidebar-peek" id="sidebarPeek" onclick="toggleSidebar()">Sections</button>'
        '<div class="main-content">' + body + "</div></div>"
    )

    # ── JS: tab switcher + per-card cycling ───────────────────────────────────
    js = (
        f"var _COV_IMAGES = {covariate_images_json};\n"
        f"var _ALL_SOURCES = {json.dumps(all_sources)};\n"
        "var _activeSource = _ALL_SOURCES[0];\n"
        "\nfunction selectTab(btn) {\n"
        "  document.querySelectorAll('.tab-btn').forEach(function(b) { b.classList.remove('tab-active'); });\n"
        "  btn.classList.add('tab-active');\n"
        "  _activeSource = btn.getAttribute('data-source');\n"
        "  syncGridToSource(_activeSource);\n"
        "}\n"
        "\nfunction syncGridToSource(source) {\n"
        "  document.querySelectorAll('.img-card[data-card]').forEach(function(card) {\n"
        "    var frames = JSON.parse(card.getAttribute('data-frames') || '[]');\n"
        "    var idx = frames.findIndex(function(f) { return f.source === source; });\n"
        "    if (idx === -1) idx = 0;\n"
        "    showFrame(card, idx);\n"
        "  });\n"
        "}\n"
        "\nfunction cycleCard(btn, dir) {\n"
        "  var card = btn.closest('.img-card');\n"
        "  var frames = JSON.parse(card.getAttribute('data-frames') || '[]');\n"
        "  var cur = parseInt(card.getAttribute('data-active-frame') || '0');\n"
        "  var next = (cur + dir + frames.length) % frames.length;\n"
        "  showFrame(card, next);\n"
        "  var newSource = frames[next].source;\n"
        "  document.querySelectorAll('.tab-btn').forEach(function(b) {\n"
        "    b.classList.toggle('tab-active', b.getAttribute('data-source') === newSource);\n"
        "  });\n"
        "  _activeSource = newSource;\n"
        "}\n"
        "\nfunction showFrame(card, idx) {\n"
        "  var frames = JSON.parse(card.getAttribute('data-frames') || '[]');\n"
        "  if (!frames.length) return;\n"
        "  idx = Math.max(0, Math.min(idx, frames.length - 1));\n"
        "  card.setAttribute('data-active-frame', idx);\n"
        "  card.querySelectorAll('img[data-frame]').forEach(function(img) {\n"
        "    img.style.display = parseInt(img.getAttribute('data-frame')) === idx ? 'block' : 'none';\n"
        "  });\n"
        "  var label = card.querySelector('.cycle-label[data-card]');\n"
        "  if (label) label.textContent = (idx + 1) + '/' + frames.length;\n"
        "  var badge = document.querySelector('.source-badge[data-card=\"' + card.getAttribute('data-card') + '\"]');\n"
        "  if (badge) badge.textContent = frames[idx].source;\n"
        "  var niiWrap = document.querySelector('.nii-links[data-card=\"' + card.getAttribute('data-card') + '\"]');\n"
        "  if (niiWrap) {\n"
        "    niiWrap.querySelectorAll('[data-nii-frame]').forEach(function(el) {\n"
        "      el.style.display = parseInt(el.getAttribute('data-nii-frame')) === idx ? 'inline-flex' : 'none';\n"
        "    });\n"
        "  }\n"
        "}\n"
        "function toggleSidebar() {\n"
        "  var sb = document.getElementById('sidebar'), pk = document.getElementById('sidebarPeek');\n"
        "  var h = sb.classList.toggle('hidden');\n"
        "  try { localStorage.setItem('sidebarHidden', h ? '1' : '0'); } catch(e) {}\n"
        "  if (pk) pk.classList.toggle('visible', h);\n"
        "}\n"
        "(function() {\n"
        "  try {\n"
        "    if (localStorage.getItem('sidebarHidden') === '1') {\n"
        "      var sb = document.getElementById('sidebar'), pk = document.getElementById('sidebarPeek');\n"
        "      if (sb) sb.classList.add('hidden');\n"
        "      if (pk) pk.classList.add('visible');\n"
        "    }\n"
        "  } catch(e) {}\n"
        "}());\n"
        "document.addEventListener('DOMContentLoaded', function() {\n"
        "  var items = document.querySelectorAll('.nav-item[data-sec]');\n"
        "  if (items.length) {\n"
        "    var secs = Array.from(items).map(function(el) {\n"
        "      return document.getElementById(el.getAttribute('data-sec'));\n"
        "    }).filter(Boolean);\n"
        "    function onScroll() {\n"
        "      var atBottom = (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 10;\n"
        "      var y = window.scrollY + 140, active = secs[0];\n"
        "      if (atBottom) { active = secs[secs.length - 1]; }\n"
        "      else { secs.forEach(function(s) { if (s.offsetTop <= y) active = s; }); }\n"
        "      items.forEach(function(el) {\n"
        "        el.classList.toggle('active', active && el.getAttribute('data-sec') === active.id);\n"
        "      });\n"
        "    }\n"
        "    window.addEventListener('scroll', onScroll, {passive: true});\n"
        "    onScroll();\n"
        "  }\n"
        "  syncGridToSource(_activeSource);\n"
        "});\n"
    )

    css = """
:root {
  --bg:#ffffff;--bg3:#f1f5f9;--border:#e2e8f0;
  --text:#0f172a;--text2:#334155;--text3:#64748b;
  --header-bg:linear-gradient(135deg,#e0e9ff 0%,#f8fafc 100%);--header-border:#e2e8f0;
  --chip-bg:#f1f5f9;--chip-color:#475569;--chip-b:#6366f1;--legend-color:#334155;--card-bg:#ffffff;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;margin:1rem 0}
button{font-family:inherit}
.page-header{background:var(--header-bg);border-bottom:1px solid var(--header-border);padding:2rem 2.5rem}
.page-header h1{font-size:1.7rem;font-weight:700;color:var(--text);letter-spacing:-.02em}
.page-header p{color:var(--text3);margin-top:.35rem;font-size:.93rem}
.chips{display:flex;gap:.6rem;margin-top:1rem;flex-wrap:wrap}
.chip{background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;padding:.25rem .75rem;font-size:.78rem;color:var(--chip-color)}
.chip b{color:var(--chip-b)}
.site-legend{display:flex;gap:.9rem;flex-wrap:wrap}
.container{max-width:1400px;margin:0 auto;padding:2rem 0}
.section{margin-bottom:3rem}
.section-title{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3);margin-bottom:1.1rem;padding-bottom:.5rem;border-bottom:1px solid var(--border)}
.tab-bar{display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:1rem}
.tab-btn{background:var(--chip-bg);border:1px solid var(--border);border-radius:999px;padding:.3rem .9rem;font-size:.8rem;font-weight:500;color:var(--text2);cursor:pointer;display:flex;align-items:center;transition:all .15s}
.tab-btn:hover{background:var(--bg3);color:var(--text)}
.tab-btn.tab-active{background:#6366f1;border-color:#6366f1;color:#fff;font-weight:600}
.img-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:1rem}
.img-card{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;overflow:hidden;padding:.85rem;transition:box-shadow .15s}
.img-card:hover{box-shadow:0 4px 16px rgba(99,102,241,.12)}
.img-card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:.4rem}
.img-card-title{font-size:.8rem;font-weight:600;color:var(--text2);text-transform:capitalize}
.source-badge{font-size:.72rem;font-weight:600;color:#6366f1;background:rgba(99,102,241,.1);border-radius:999px;padding:.15rem .55rem;margin-bottom:.45rem;display:inline-block}
.cycle-controls{display:flex;align-items:center;gap:.3rem}
.cycle-btn{background:none;border:1px solid var(--border);border-radius:6px;width:22px;height:22px;cursor:pointer;font-size:1rem;color:var(--text3);display:flex;align-items:center;justify-content:center;line-height:1;padding:0}
.cycle-btn:hover{background:var(--bg3);color:var(--text)}
.cycle-label{font-size:.72rem;color:var(--text3);min-width:28px;text-align:center}
.layout{display:flex;align-items:flex-start;padding:1.5rem 2rem 0}
.sidebar{width:190px;flex-shrink:0;position:sticky;top:1.5rem;max-height:calc(100vh - 3rem);overflow-y:auto;margin-right:1.5rem;transition:width .2s,opacity .2s,margin .2s}
.sidebar.hidden{width:0;opacity:0;overflow:hidden;margin-right:0;pointer-events:none}
.sidebar-inner{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:.6rem .5rem}
.sidebar-header{display:flex;align-items:center;justify-content:space-between;padding:.2rem .3rem .45rem;border-bottom:1px solid var(--border);margin-bottom:.4rem}
.sidebar-label{font-size:.67rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--text3)}
.sidebar-toggle{background:none;border:none;cursor:pointer;font-size:.75rem;color:var(--text3);padding:.15rem .35rem;border-radius:5px;line-height:1}
.sidebar-toggle:hover{background:var(--bg3);color:var(--text)}
.nav-item{display:block;width:100%;padding:.4rem .65rem;border-radius:7px;font-size:.78rem;color:var(--text3);line-height:1.35;cursor:pointer;border:none;background:none;text-align:left;transition:background .12s,color .12s}
.nav-item:hover{background:var(--bg3);color:var(--text)}
.nav-item.active{background:rgba(99,102,241,.13);color:#6366f1;font-weight:600}
.main-content{flex:1;min-width:0}
.sidebar-peek{position:fixed;left:0;top:50%;transform:translateY(-50%);background:var(--card-bg);border:1px solid var(--border);border-left:none;border-radius:0 8px 8px 0;padding:.55rem .35rem;cursor:pointer;font-size:.72rem;color:var(--text3);display:none;z-index:200;writing-mode:vertical-rl;letter-spacing:.06em;line-height:1}
.sidebar-peek:hover{color:var(--text)}
.sidebar-peek.visible{display:block}
.table-wrap{overflow-x:auto}
.stats-table{width:100%;border-collapse:collapse;font-size:.82rem}
.stats-table th{background:var(--bg3);color:var(--text3);font-weight:700;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em;padding:.55rem .75rem;text-align:left;border-bottom:2px solid var(--border)}
.stats-table td{padding:.5rem .75rem;border-bottom:1px solid var(--border);color:var(--text2)}
.stats-table tr:last-child td{border-bottom:none}
.stats-table tr:hover td{background:var(--bg3)}
.dtype-badge{background:var(--chip-bg);border:1px solid var(--border);border-radius:4px;padding:.1rem .4rem;font-size:.72rem;color:var(--chip-color);font-family:monospace}
"""

    html_content = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8"/>\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1.0"/>\n'
        "<title>FreeSurfer Voxelwise Regression Report</title>\n"
        f"<style>{css}</style>\n</head>\n<body>\n"
        f"{sidebar_and_body}\n"
        f"<script>{js}</script>\n"
        "<script>\n"
        'document.addEventListener("DOMContentLoaded", function() {\n'
        '  var base = window.location.href.split("?")[0].replace(/\\/[^\\/]*$/, "");\n'
        "  var origin = window.location.origin;\n"
        '  document.querySelectorAll("a[data-nii-rel]").forEach(function(a) {\n'
        '    var rel = a.getAttribute("data-nii-rel").replace(/^\\.?\\//, "");\n'
        '    var niiUrl = base + "/" + rel;\n'
        '    a.href = origin + "/nii-viewer?url=" + encodeURIComponent(niiUrl);\n'
        '    a.target = "_blank";\n'
        '    a.removeAttribute("data-nii-rel");\n'
        "  });\n"
        "});\n"
        "</script>\n</body>\n</html>"
    )

    try:
        html_content = html_content.encode("utf-8", errors="replace").decode("utf-8")
        with open(output_file_name, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as _write_err:
        with open(output_file_name, "w", encoding="utf-8") as f:
            f.write(f"<pre>HTML write error: {_write_err}</pre>")
        raise

    return output_file_name


def _persist_html_safe_wrapper(results_dict, output_file_name, cache_dict=None):
    """Render the report and propagate failures to the terminal error boundary."""
    return _persist_html_from_results(results_dict, output_file_name, cache_dict)


def _local_stats_to_dict_numba(X, y):
    import scipy as sp
    from numba import prange

    def gather_local_stats_helper(X, y, pinv):
        """Calculate local statistics."""
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
        """Calculate local statistics."""
        try:
            pinv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError as error:
            cond = np.linalg.cond(X.T @ X)
            raise ValueError(
                f"Local design matrix is singular with condition number {cond}"
            ) from error

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
