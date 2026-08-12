"""Adapt the historical VBM mathematics to the computation-only workflow API."""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
from framework import ArtifactRef, artifact, with_state

from . import binary_io, local_math, remote_math


def load_vbm_data(data_dir: str) -> str:
    """Return the runtime-owned participant data directory."""
    return data_dir


def prepare_local_mask(
    inputs: str,
    *,
    parameters: Dict[str, Any],
    artifact_dir: str,
    logger,
):
    """Validate local inputs and produce the site-average NIfTI artifact."""
    result = local_math.perform_client_step1_validate_inputs_and_compute_mask(
        os.path.join(inputs, "covariates.csv"),
        inputs,
        parameters,
        logger,
        {},
        artifact_dir,
    )
    payload = dict(result["output"])
    payload["average_nifti"] = artifact(
        "avg_nifti.nii",
        result["transfer_files"]["average_nifti"],
        "application/nifti",
    )
    return with_state(payload, result["cache"])


def aggregate_mask(
    site_results: Dict[str, dict],
    *,
    parameters: Dict[str, Any],
    artifact_dir: str,
):
    """Compute the global mask and downsampled MNI artifacts."""
    historical_inputs = {}
    for site_name, site_result in site_results.items():
        historical_inputs[site_name] = dict(site_result)
        historical_inputs[site_name]["transfer_average_nifti"] = site_result[
            "average_nifti"
        ].path
    remote_cache = {
        "site_id_name_map": parameters.get("site_id_name_map", {}),
    }
    result = remote_math.perform_remote_step1_generate_average_mask(
        historical_inputs,
        remote_cache,
        artifact_dir,
    )
    payload = dict(result["output"])
    payload["mask_file"] = artifact(
        "mask.nii", result["transfer_files"]["mask_file"], "application/nifti"
    )
    payload["downsampled_mni"] = artifact(
        "mni_downsampled.nii",
        result["transfer_files"]["downsampled_mni"],
        "application/nifti",
    )
    remote_cache.update(result.get("cache", {}))
    remote_cache["_cache_dir"] = artifact_dir
    return with_state(payload, remote_cache)


def fit_local_model(
    global_mask: dict,
    *,
    state: dict,
    data_dir: str,
    artifact_dir: str,
    logger,
):
    """Fit the historical local voxelwise model and export sufficient statistics."""
    result = local_math.perform_local_step2_compute_local_model(
        global_mask,
        data_dir,
        logger,
        state,
        artifact_dir,
    )
    next_state = {**state, **result["cache"]}
    # Historical code used one persistent cache directory. The current framework
    # gives each stage its own private artifact directory, so retain absolute paths
    # for arrays consumed by later stages.
    next_state["covariates"] = os.path.join(artifact_dir, "X.npy")
    next_state["dependents"] = os.path.join(artifact_dir, "y.npy")
    payload = dict(result["output"])

    for key, logical_name in (
        ("mean_y_local", "mean_y_local.npy"),
        ("count_local", "count_local.npy"),
    ):
        payload[key] = _array_artifact(
            np.asarray(payload[key]), artifact_dir, logical_name
        )

    local_stat_keys = (
        "local_beta_vector",
        "local_pvalues",
        "local_tvalues",
        "local_rsquared",
    )
    for key in local_stat_keys:
        state_path = os.path.join(artifact_dir, f"state_{key}.npy")
        binary_io.save_binary(state_path, np.asarray(next_state.pop(key)))
        next_state[f"{key}_file"] = state_path
        if key in payload:
            payload[key] = artifact(f"{key}.npy", state_path)

    payload["x_transpose_x"] = artifact(
        "XTX.npy",
        result["transfer_files"]["x_transpose_x_file_name"],
    )
    payload["x_transpose_y"] = artifact(
        "XTy.npy",
        result["transfer_files"]["x_transpose_y_file_name"],
    )
    return with_state(payload, next_state)


def aggregate_global_model(
    site_results: Dict[str, dict],
    *,
    state: dict,
    artifact_dir: str,
):
    """Combine local design products into the global regression model."""
    historical_inputs = {}
    for site_name, site_result in site_results.items():
        item = dict(site_result)
        item["mean_y_local"] = _load_ref(item["mean_y_local"]).tolist()
        item["count_local"] = _load_ref(item["count_local"]).tolist()
        item["x_transpose_x_file_name"] = item.pop("x_transpose_x").path
        item["x_transpose_y_file_name"] = item.pop("x_transpose_y").path
        for key in (
            "local_beta_vector",
            "local_pvalues",
            "local_tvalues",
            "local_rsquared",
        ):
            if key in item:
                item[key] = _load_ref(item[key]).tolist()
        historical_inputs[site_name] = item

    result = remote_math.perform_remote_step2_compute_global_model(
        historical_inputs,
        state,
        artifact_dir,
    )
    payload = dict(result["output"])
    payload["avg_beta_vector"] = artifact(
        "avg_beta_vector.npy", result["transfer_files"]["avg_beta_vector"]
    )
    payload["mean_y_global"] = artifact(
        "mean_y_global.npy", result["transfer_files"]["mean_y_global"]
    )
    next_state = {**state, **result.get("cache", {}), "_cache_dir": artifact_dir}
    return with_state(payload, next_state)


def compute_local_metrics(
    global_model: dict,
    *,
    state: dict,
    artifact_dir: str,
    logger,
):
    """Compute local sums needed for global model-fit statistics."""
    result = local_math.perform_local_step3_compute_metrics_with_global_params(
        global_model,
        logger,
        state,
        artifact_dir,
    )
    payload = {
        key: _array_artifact(np.asarray(value), artifact_dir, f"{key}.npy")
        for key, value in result["output"].items()
    }
    return with_state(payload, {**state, **result["cache"]})


def aggregate_global_metrics(
    site_results: Dict[str, dict],
    *,
    state: dict,
    artifact_dir: str,
):
    """Compute the historical global R-squared, t, and p statistics."""
    historical_inputs = {
        site_name: {key: _load_ref(value).tolist() for key, value in result.items()}
        for site_name, result in site_results.items()
    }
    result = remote_math.perform_remote_step3_aggregate_metrics(
        historical_inputs,
        state,
        state["_cache_dir"],
    )
    output = result["output"]
    payload = {"X_labels": output["X_labels"]}
    for key in ("r_squared_global", "MSE", "ts_global", "ps_global"):
        payload[key] = _array_artifact(
            np.asarray(output[key]), artifact_dir, f"{key}.npy"
        )

    shared_stats = {}
    for site_index, (site_name, stats) in enumerate(
        output.get("shared_local_stats", {}).items()
    ):
        shared_stats[site_name] = {"local_X_labels": stats["local_X_labels"]}
        for key in (
            "local_beta_vector",
            "local_pvalues",
            "local_tvalues",
            "local_rsquared",
        ):
            logical_name = f"shared-{site_index}-{key}.npy"
            shared_stats[site_name][key] = _array_artifact(
                np.asarray(stats[key]), artifact_dir, logical_name
            )
    if shared_stats:
        payload["shared_local_stats"] = shared_stats
    return payload


def write_vbm_outputs(
    global_metrics: dict,
    *,
    state: dict,
    output_dir: str,
    artifact_dir: str,
    logger,
) -> None:
    """Render the same NIfTI, PNG, and HTML outputs as the historical runtime."""
    historical_result = {"X_labels": global_metrics["X_labels"]}
    for key in ("r_squared_global", "MSE", "ts_global", "ps_global"):
        historical_result[key] = _load_ref(global_metrics[key]).tolist()
    if "shared_local_stats" in global_metrics:
        historical_result["shared_local_stats"] = {}
        for site_name, stats in global_metrics["shared_local_stats"].items():
            historical_result["shared_local_stats"][site_name] = {
                "local_X_labels": stats["local_X_labels"],
                **{
                    key: _load_ref(stats[key]).tolist()
                    for key in (
                        "local_beta_vector",
                        "local_pvalues",
                        "local_tvalues",
                        "local_rsquared",
                    )
                },
            }

    output_state = dict(state)
    for key in (
        "local_beta_vector",
        "local_pvalues",
        "local_tvalues",
        "local_rsquared",
    ):
        output_state[key] = binary_io.load_binary(output_state[f"{key}_file"]).tolist()
    local_math.perform_local_step4_persist_results(
        historical_result,
        logger,
        output_state,
        artifact_dir,
        output_dir,
    )


def _array_artifact(
    array: np.ndarray, directory: str, logical_name: str
) -> ArtifactRef:
    path = os.path.join(directory, logical_name)
    binary_io.save_binary(path, array)
    return artifact(logical_name, path, "application/x-neuroflame-numpy-array")


def _load_ref(value: ArtifactRef) -> np.ndarray:
    if not isinstance(value, ArtifactRef):
        raise TypeError("Expected a transferred artifact reference")
    return binary_io.load_binary(value.path)
