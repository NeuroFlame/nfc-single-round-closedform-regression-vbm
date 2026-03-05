# aggregator_methods.py

import os
import json
import glob
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jsonpickle
import numpy as np
import pandas as pd
from nvflare.apis.fl_context import FLContext

from utils import data_transfer as dt
from utils.utils import get_output_directory_path

logger = logging.getLogger(__name__)

# NOTE: this path is what your containers have been using.
# If you support alternate layouts, you can override via env if needed.
MNI_TEMPLATE = os.getenv("MNI_TEMPLATE_PATH", "/workspace/app/code/assets/MNI152_T1_1mm_brain.nii")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _json_sanitize(obj):
    """
    Recursively convert numpy/pandas types into JSON-serializable types.
    """
    import numpy as _np

    if obj is None:
        return None

    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)

    if isinstance(obj, _np.ndarray):
        return obj.tolist()

    try:
        import pandas as _pd

        if isinstance(obj, (_pd.Series, _pd.Index)):
            return obj.tolist()
        if isinstance(obj, _pd.DataFrame):
            return obj.to_dict(orient="list")
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    return obj


def _safe_get(d: dict, key: Any, default=None):
    try:
        return d.get(key, default) if isinstance(d, dict) else default
    except Exception:
        return default


def _ensure_headless_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # noqa
    except Exception:
        pass


def _zip_dir_contents(src_dir: str, zip_path: str) -> None:
    """
    Zip the *contents* of src_dir into zip_path (no top-level folder inside zip).
    """
    import zipfile

    if not zip_path:
        raise ValueError("zip_path must be a non-empty string")

    src_dir_abs = os.path.abspath(src_dir)

    # If missing, create empty zip (defensive)
    if not src_dir or not os.path.isdir(src_dir_abs):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED):
            pass
        return

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir_abs):
            for fn in files:
                full_path = os.path.join(root, fn)
                rel_path = os.path.relpath(full_path, src_dir_abs)
                z.write(full_path, arcname=rel_path)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default

def _truthy(v: Optional[str]) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _running_in_docker() -> bool:
    try:
        return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
    except Exception:
        return False

def _normalize_fileserver_url(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    # Common container trap: localhost/127.0.0.1 points to *this* container.
    if (u.startswith("http://localhost") or u.startswith("https://localhost") or "://127.0.0.1" in u) and _running_in_docker():
        return (
            u.replace("localhost", "host.docker.internal")
             .replace("127.0.0.1", "host.docker.internal")
        )
    return u


def upload_results_to_fileserver(results_zip_path: str, fl_ctx: Optional[FLContext] = None) -> Dict[str, Any]:
    """Upload results.zip to NeuroFLAME fileServer so clients/observers can download it.

    Endpoint:
      POST {NEUROFLAME_FILESERVER_URL}/upload_results/{NEUROFLAME_CONSORTIUM_ID}/{NEUROFLAME_RUN_ID}
      multipart/form-data field name: 'file'
      header: x-access-token: {NEUROFLAME_ACCESS_TOKEN}

    Required env vars (ONLY these):
      - NEUROFLAME_FILESERVER_URL
      - NEUROFLAME_CONSORTIUM_ID
      - NEUROFLAME_RUN_ID
      - NEUROFLAME_ACCESS_TOKEN

    Optional debug env vars:
      - NEUROFLAME_RESULTS_UPLOAD_ENABLED (default: true)
      - NEUROFLAME_RESULTS_UPLOAD_DEBUG   (default: false)
      - NEUROFLAME_FILESERVER_PREFLIGHT   (default: false)  # GET /health before upload (non-fatal)
    """

    def _running_in_docker() -> bool:
        try:
            return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        except Exception:
            return False

    def _normalize_fileserver_url(u: Optional[str]) -> Optional[str]:
        """Inside Docker, rewrite localhost/127.0.0.1 to host.docker.internal (Mac Docker Desktop)."""
        if not u:
            return u
        s = u.strip()
        if not s:
            return s
        if _running_in_docker() and (
            s.startswith("http://localhost")
            or s.startswith("https://localhost")
            or "://127.0.0.1" in s
        ):
            return s.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
        return s

    enabled = _env("NEUROFLAME_RESULTS_UPLOAD_ENABLED", "true")
    debug = _env("NEUROFLAME_RESULTS_UPLOAD_DEBUG", "false")
    preflight = _env("NEUROFLAME_FILESERVER_PREFLIGHT", "false")

    if not _truthy(enabled):
        logger.info("NeuroFLAME results upload disabled via NEUROFLAME_RESULTS_UPLOAD_ENABLED=%s", enabled)
        return {"uploaded": False, "reason": "disabled"}

    # --- ONLY the env vars you specified ---
    file_server = _env("NEUROFLAME_FILESERVER_URL")
    consortium_id = _env("NEUROFLAME_CONSORTIUM_ID")
    run_id = _env("NEUROFLAME_RUN_ID")
    token = _env("NEUROFLAME_ACCESS_TOKEN")

    if not all([file_server, consortium_id, run_id, token]):
        missing = [
            k for k, v in {
                "NEUROFLAME_FILESERVER_URL": file_server,
                "NEUROFLAME_CONSORTIUM_ID": consortium_id,
                "NEUROFLAME_RUN_ID": run_id,
                "NEUROFLAME_ACCESS_TOKEN": token,
            }.items()
            if not v
        ]
        logger.warning("NeuroFLAME results upload skipped: missing env vars: %s", ", ".join(missing))
        return {"uploaded": False, "reason": "missing_env", "missing": missing}

    # Normalize localhost inside container -> host.docker.internal
    file_server = _normalize_fileserver_url(file_server)

    if not os.path.exists(results_zip_path):
        logger.error("NeuroFLAME results upload failed: results zip not found at %s", results_zip_path)
        return {"uploaded": False, "reason": "zip_missing", "results_zip_path": results_zip_path}

    try:
        file_size = os.path.getsize(results_zip_path)
    except Exception:
        file_size = None

    try:
        import requests
    except Exception as e:
        logger.error("NeuroFLAME results upload failed: requests not available: %s", e)
        return {"uploaded": False, "reason": "requests_missing", "error": str(e)}

    url = f"{file_server.rstrip('/')}/upload_results/{consortium_id}/{run_id}"
    headers = {"x-access-token": token}

    logger.info("Uploading results.zip to %s", url)

    if _truthy(debug):
        logger.info(
            "NeuroFLAME results upload debug",
            extra={
                "url": url,
                "results_zip_path": results_zip_path,
                "file_size": file_size,
                "running_in_docker": _running_in_docker(),
            },
        )

    if _truthy(preflight):
        try:
            health_url = f"{file_server.rstrip('/')}/health"
            r = requests.get(health_url, timeout=5)
            if _truthy(debug):
                logger.info("fileServer preflight GET /health status=%s", r.status_code)
        except Exception:
            if _truthy(debug):
                logger.info("fileServer preflight skipped/failed (non-fatal)", exc_info=True)

    try:
        with open(results_zip_path, "rb") as f:
            files = {"file": ("results.zip", f, "application/zip")}
            resp = requests.post(url, headers=headers, files=files, timeout=120)

        if 200 <= resp.status_code < 300:
            logger.info("NeuroFLAME results uploaded successfully (status=%s, size=%s)", resp.status_code, file_size)
            return {"uploaded": True, "status": resp.status_code, "url_used": url, "file_size": file_size}

        try:
            body = (resp.text or "")[:2000]
        except Exception:
            body = "<unreadable>"

        logger.error("NeuroFLAME results upload failed (status=%s): %s", resp.status_code, body)
        return {"uploaded": False, "status": resp.status_code, "url_used": url, "file_size": file_size, "body": body}

    except Exception as e:
        logger.exception("NeuroFLAME results upload exception: %s", e)
        return {"uploaded": False, "reason": "exception", "error": str(e), "url_used": url, "file_size": file_size}


# -----------------------------------------------------------------------------
# Remote aggregation steps (round 0/1/2)
# -----------------------------------------------------------------------------

def perform_remote_step1_generate_average_mask(site_results: dict, agg_cache_dict: dict, agg_cache_dir: str) -> dict:
    """
    Computes the average of all nifti files to generate a mask and determines
    all the categorical covariates across all the clients.

    Returns:
      {
        "output": {...},       # JSON-ish
        "cache": {...},        # cached values for later rounds
        "transfer_files": {    # file paths on server to transfer back to clients
          "mask_file": "...",
          "downsampled_mni": "..."
        }
      }
    """

    def return_uniques_and_counts(df: pd.DataFrame) -> Tuple[dict, dict]:
        """Return unique-values of the categorical variables and their counts"""
        keys = df.iloc[:, :].sum(axis=1).apply(set).apply(sorted).to_dict()  # adding all columns
        count = {k: len(v) for k, v in keys.items()}
        return keys, count

    output_mask_file_name, output_downsampled_mni_filename = _calculate_mask(site_results, agg_cache_dir)

    site_info = {site: site_results[site]["categorical_dict"] for site in site_results.keys()}
    ref_cols = {site: site_results[site]["reference_columns"] for site in site_results.keys()}

    reference_dict = next(iter(ref_cols.values()))

    # correctness check for decentralized regression dummy encoding
    assert all(value == reference_dict for value in ref_cols.values()), (
        "Reference values for dummy encoding are not same across all the sites " + str(ref_cols)
    )

    df = pd.DataFrame.from_dict(site_info)
    covar_keys, unique_count = return_uniques_and_counts(df)

    results = {
        "output": {
            "covar_keys": jsonpickle.encode(covar_keys, unpicklable=False),
            "reference_columns": reference_dict,
            "global_unique_count": unique_count,
        },
        "cache": {},
        "transfer_files": {
            "mask_file": output_mask_file_name,
            "downsampled_mni": output_downsampled_mni_filename,
        },
    }
    return results


def perform_remote_step2_compute_global_model(site_results: dict, agg_cache_dict: dict, agg_cache_dir: str) -> dict:
    """
    Computes the global regression beta weights based on the client model parameters
    and sends it to the clients.
    """

    site_ids = sorted(site_results.keys())
    if not site_ids:
        raise RuntimeError("No site_results provided to step2")

    user_id = site_ids[0]
    X_labels = site_results[user_id]["X_labels"]

    beta_vector_0 = sum([dt.load_binary(site_results[site]["x_transpose_x_file_name"]) for site in site_ids])
    beta_vector_1 = sum([dt.load_binary(site_results[site]["x_transpose_y_file_name"]) for site in site_ids])

    all_lambdas = [site_results[site]["lambda"] for site in site_results]

    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")

    try:
        # Add lambda's (ridge)
        beta_vector_0 = beta_vector_0 + np.unique(all_lambdas) * np.eye(beta_vector_0.shape[0])
        avg_beta_vector = np.transpose(np.dot(np.linalg.inv(beta_vector_0), beta_vector_1))
    except np.linalg.LinAlgError:
        cond = np.linalg.cond(beta_vector_0)
        raise Exception(f"X.^T*X matrix at remote is Singular with condition number: {cond}")

    mean_y_local = [site_results[site]["mean_y_local"] for site in site_results]
    count_y_local = [np.array(site_results[site]["count_local"]) for site in site_results]
    mean_y_global = np.array(mean_y_local) * np.array(count_y_local)
    mean_y_global = np.sum(mean_y_global, axis=0) / np.sum(count_y_local, axis=0)

    dof_global = sum(count_y_local) - avg_beta_vector.shape[1]

    avg_beta_vector_file_name = os.path.join(agg_cache_dir, "avg_beta_vector.npy")
    dt.save_binary(avg_beta_vector_file_name, avg_beta_vector)

    mean_y_global_file_name = os.path.join(agg_cache_dir, "mean_y_global.npy")
    dt.save_binary(mean_y_global_file_name, mean_y_global)

    cache_dict = {
        "avg_beta_vector": "avg_beta_vector.npy",
        "dof_global": dof_global.tolist(),
        "X_labels": X_labels,
    }

    results = {
        "output": {},
        "cache": cache_dict,
        "transfer_files": {
            "avg_beta_vector": avg_beta_vector_file_name,
            "mean_y_global": mean_y_global_file_name,
        },
    }
    return results


def perform_remote_step3_aggregate_metrics(site_results: dict, agg_cache_dict: dict, agg_cache_dir: str) -> dict:
    """
    Aggregates all the local performance metrics and computes global metrics.
    """
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
    avg_beta_vector = dt.load_binary(os.path.join(agg_cache_dir, agg_cache_dict["avg_beta_vector"]))
    dof_global = agg_cache_dict["dof_global"]

    SSE_global = sum([np.array(site_results[site]["SSE_local"]) for site in site_results])
    SST_global = sum([np.array(site_results[site]["SST_local"]) for site in site_results])
    varX_matrix_global = sum([np.array(site_results[site]["varX_matrix_local"]) for site in site_results])

    r_squared_global = 1 - (SSE_global / SST_global)
    MSE = SSE_global / np.array(dof_global)
    ts_global = remote_stats(MSE, varX_matrix_global, avg_beta_vector)
    ps_global = [2 * stats.t.sf(np.abs(t), df) for t, df in zip(ts_global, dof_global)]

    output_dict = {
        "r_squared_global": r_squared_global.tolist(),
        "MSE": MSE.tolist(),
        "ts_global": ts_global.tolist(),
        "ps_global": ps_global,
        "X_labels": X_labels,
    }
    return {"output": output_dict, "cache": {}}


def _calculate_mask(site_results: dict, agg_cache_dir: str) -> Tuple[str, str]:
    import nibabel as nib
    from nilearn.image import resample_img, resample_to_img

    num_sites = len(site_results.keys())
    if num_sites == 0:
        raise RuntimeError("_calculate_mask called with empty site_results")

    avg_of_all = None
    for site, results in site_results.items():
        if "transfer_average_nifti" not in results:
            raise RuntimeError(f"Missing transfer_average_nifti for site={site}")
        arr = np.asarray(nib.load(results["transfer_average_nifti"]).get_fdata())
        avg_of_all = (avg_of_all + arr) if avg_of_all is not None else arr

    avg_of_all = avg_of_all / num_sites

    # Threshold binarizer
    site_ids = sorted(site_results.keys())
    user_id = site_ids[0]
    threshold = site_results[user_id]["threshold"]
    voxel_size = site_results[user_id]["voxel_size"]

    mask_info = avg_of_all > threshold

    principal_image = nib.load(site_results[user_id]["transfer_average_nifti"])
    header, affine = principal_image.header, principal_image.affine

    clipped_img = nib.Nifti1Image(mask_info, affine, header)

    if not os.path.exists(MNI_TEMPLATE):
        raise RuntimeError(f"MNI template not found at: {MNI_TEMPLATE}")

    mni_img = nib.load(MNI_TEMPLATE)
    reoriented_mni = resample_to_img(mni_img, clipped_img, interpolation="linear")
    downsampled_mni = resample_img(reoriented_mni, target_affine=np.eye(3) * voxel_size, interpolation="linear")
    downsampled_mask = resample_to_img(clipped_img, downsampled_mni, interpolation="nearest")

    output_mask_file_name = os.path.join(agg_cache_dir, "mask.nii")
    output_downsampled_mni_filename = os.path.join(agg_cache_dir, "mni_downsampled.nii")
    nib.save(downsampled_mask, output_mask_file_name)
    nib.save(downsampled_mni, output_downsampled_mni_filename)

    return output_mask_file_name, output_downsampled_mni_filename


# -----------------------------------------------------------------------------
# Results payload builder (server-side)
# -----------------------------------------------------------------------------

def extract_site_stats(site_results_by_round: dict) -> dict:
    """
    Extract per-site results to support server-side "local_stats" rendering.

    site_results_by_round structure (from aggregator.py):
      { round_num: { site_name: site_result_dict } }
    """
    if not isinstance(site_results_by_round, dict):
        return {}

    sites = set()
    for _, round_map in site_results_by_round.items():
        if isinstance(round_map, dict):
            sites.update(round_map.keys())

    out: Dict[str, Any] = {}
    for site in sorted(sites):
        r0 = _safe_get(site_results_by_round, 0, {}).get(site, {})
        r1 = _safe_get(site_results_by_round, 1, {}).get(site, {})
        r2 = _safe_get(site_results_by_round, 2, {}).get(site, {})

        merged: Dict[str, Any] = {}
        for src in (r0, r1, r2):
            if isinstance(src, dict):
                merged.update(src)

        # Compute local r2 if SSE/SST are present
        sse_local = merged.get("SSE_local", None)
        sst_local = merged.get("SST_local", None)
        if merged.get("local_rsquared", None) is None and sse_local is not None and sst_local is not None:
            try:
                merged["local_rsquared"] = (1 - (np.asarray(sse_local) / np.asarray(sst_local))).tolist()
            except Exception:
                pass

        out[str(site)] = _json_sanitize(merged)

    return out


def build_global_results_payload(step3_output: dict, agg_cache_dict: dict, site_results_by_round: Optional[dict] = None) -> dict:
    payload: Dict[str, Any] = {"results": step3_output or {}}

    if isinstance(agg_cache_dict, dict):
        if "X_labels" in agg_cache_dict:
            payload["X_labels"] = agg_cache_dict["X_labels"]
        if "dof_global" in agg_cache_dict:
            payload["dof_global"] = agg_cache_dict["dof_global"]

    payload["site_stats"] = extract_site_stats(site_results_by_round or {})
    return _json_sanitize(payload)


# -----------------------------------------------------------------------------
# Server-side results writer (writes output/results/* and output/results.zip)
# -----------------------------------------------------------------------------

def _persist_html_from_results_server(results_dict_in: dict, output_file_name: str) -> str:
    """
    Writes a simple index.html referencing PNGs under global_stats/ and local_stats/<site>/.
    """
    try:
        import dominate
        from dominate import tags

        doc = dominate.document(title="Results")

        with doc.head:
            tags.style(
                """
                body { font-family: sans-serif; }
                hr { width: 100%; }
                img { max-width: 100%; height: auto; }
                """
            )

        with doc:
            tags.h1("Results")
            tags.hr()

            root = results_dict_in.get("results", results_dict_in)
            for label in ["global_stats", "local_stats"]:
                tags.h2(label)
                for img_path in root.get(label, []):
                    site_prefix = ""
                    if label == "local_stats":
                        try:
                            site_prefix = img_path.split("local_stats/", 1)[1].split("/", 1)[0] + " : "
                        except Exception:
                            site_prefix = ""
                    tags.h3(site_prefix + label + " - " + os.path.basename(img_path))
                    tags.img(src=img_path)
                    tags.hr()

        with open(output_file_name, "w", encoding="utf-8") as f:
            f.write(str(doc))
        return output_file_name

    except Exception:
        # Fallback HTML
        root = results_dict_in.get("results", results_dict_in)
        html = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8' />",
            "<title>Results</title></head><body>",
            "<h1>Results</h1><hr/>",
        ]
        for label in ["global_stats", "local_stats"]:
            html.append(f"<h2>{label}</h2>")
            for img in root.get(label, []):
                html.append(f"<h3>{label} - {os.path.basename(img)}</h3>")
                html.append(f"<img src='{img}' style='max-width:100%; height:auto;'/>")
                html.append("<hr/>")
        html.append("</body></html>")
        with open(output_file_name, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        return output_file_name


def _render_local_stats_for_site(
    site_name: str,
    site_payload: dict,
    cache_dict_for_cnh: dict,
    cache_dir: str,
    out_dir: str,
) -> None:
    """
    Render local_stats images for a single site (defensive; renders what it can).
    """
    from executor import client_nifti_helper as cnh

    os.makedirs(out_dir, exist_ok=True)

    local_p = site_payload.get("local_pvalues", None)
    local_t = site_payload.get("local_tvalues", None)
    local_b = site_payload.get("local_beta_vector", None)
    local_r2 = site_payload.get("local_rsquared", None) or site_payload.get("r_squared_local", None)

    local_labels = (
        site_payload.get("local_X_labels", None)
        or site_payload.get("local_x_labels", None)
        or site_payload.get("X_labels", None)
    )

    # pvals/tvals
    try:
        if local_p is not None and local_t is not None and local_labels is not None:
            p_arr = np.asarray(local_p)
            t_arr = np.asarray(local_t)
            try:
                cnh.print_pvals(cache_dict_for_cnh, cache_dir, out_dir, p_arr.T, t_arr.T, local_labels)
            except Exception:
                cnh.print_pvals(cache_dict_for_cnh, cache_dir, out_dir, p_arr, t_arr, local_labels)
    except Exception:
        logger.debug("Local p/t rendering failed for site=%s", site_name, exc_info=True)

    # betas
    try:
        if local_b is not None and local_labels is not None:
            cnh.print_beta_images(cache_dict_for_cnh, cache_dir, out_dir, np.asarray(local_b), local_labels)
    except Exception:
        logger.debug("Local beta rendering failed for site=%s", site_name, exc_info=True)

    # r2
    try:
        if local_r2 is not None:
            cnh.print_r2_image(cache_dict_for_cnh, cache_dir, out_dir, np.asarray(local_r2))
    except Exception:
        logger.debug("Local r2 rendering failed for site=%s", site_name, exc_info=True)


def write_results_artifacts_from_aggregator(
    global_results: dict,
    agg_cache_dir: str,
    fl_ctx: Optional[FLContext] = None,
    output_dir: Optional[str] = None,
    clean_first: bool = True,
) -> Dict[str, Any]:
    """
    Writes server-side artifacts under:
      <output_dir>/results/
    And creates:
      <output_dir>/results.zip

    NOTE: This function DOES NOT recurse. It is safe to call once at round 2 end.

    Returns:
      {
        "output_dir": "...",
        "results_dir": ".../results",
        "results_zip": ".../results.zip",
        "index_html": ".../results/index.html",
        "results_json": ".../results/results.json",
      }
    """
    if not isinstance(global_results, dict):
        raise RuntimeError("global_results must be a dict")

    if not output_dir:
        if not fl_ctx:
            raise RuntimeError("output_dir not provided and fl_ctx is None; cannot resolve NVFlare output directory.")
        output_dir = get_output_directory_path(fl_ctx)

    _ensure_headless_matplotlib()

    out_dir = Path(output_dir)
    results_dir = out_dir / "results"
    artifacts_dir = results_dir / "artifacts"
    global_stats_dir = results_dir / "global_stats"
    local_stats_root = results_dir / "local_stats"

    agg_cache_dir_abs = os.path.abspath(str(agg_cache_dir))

    if clean_first and results_dir.exists():
        shutil.rmtree(results_dir)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    global_stats_dir.mkdir(parents=True, exist_ok=True)
    local_stats_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1) Copy binary artifacts from agg_cache_dir -> results/artifacts
    # ---------------------------------------------------------------------
    candidate_files = [
        "mask.nii",
        "mni_downsampled.nii",
        "avg_beta_vector.npy",
        "mean_y_global.npy",
    ]

    included = []
    for fname in candidate_files:
        src = Path(agg_cache_dir_abs) / fname
        if src.exists() and src.is_file():
            shutil.copy2(src, artifacts_dir / fname)
            included.append(fname)

    required_for_render = ["mask.nii", "mni_downsampled.nii", "avg_beta_vector.npy"]
    missing = [f for f in required_for_render if f not in included]
    if missing:
        raise RuntimeError(
            f"Server rendering requires artifacts in agg_cache_dir but they were missing: {missing}. "
            f"Have: {included}. agg_cache_dir={agg_cache_dir_abs}"
        )

    # ---------------------------------------------------------------------
    # 2) Render GLOBAL stats PNG outputs
    # ---------------------------------------------------------------------
    if "results" not in global_results or not isinstance(global_results["results"], dict):
        raise RuntimeError("global_results missing required dict key: 'results'")

    metrics = global_results["results"]
    required_metric_keys = ["r_squared_global", "ts_global", "ps_global", "X_labels"]
    missing_metric_keys = [k for k in required_metric_keys if k not in metrics]
    if missing_metric_keys:
        raise RuntimeError(f"global_results['results'] missing required keys: {missing_metric_keys}")

    r_squared_global = np.asarray(metrics["r_squared_global"])
    ts_global = np.asarray(metrics["ts_global"])
    ps_global = np.asarray(metrics["ps_global"])
    X_labels = np.asarray(metrics["X_labels"])

    avg_beta_vector_path = os.path.join(agg_cache_dir_abs, "avg_beta_vector.npy")
    avg_beta_vector = dt.load_binary(avg_beta_vector_path)

    cache_dict_for_cnh = {
        # names expected by client_nifti_helper
        "mask_file_name": "mask.nii",
        "downsampled_mni_file_name": "mni_downsampled.nii",
        "mask_file": "mask.nii",
        "downsampled_mni": "mni_downsampled.nii",
        "avg_beta_vector": "avg_beta_vector.npy",
        "avg_beta_vector_file_name": "avg_beta_vector.npy",
    }

    from executor import client_nifti_helper as cnh

    cnh.print_pvals(cache_dict_for_cnh, agg_cache_dir_abs, str(global_stats_dir), ps_global, ts_global, X_labels)
    cnh.print_beta_images(cache_dict_for_cnh, agg_cache_dir_abs, str(global_stats_dir), avg_beta_vector, X_labels)
    cnh.print_r2_image(cache_dict_for_cnh, agg_cache_dir_abs, str(global_stats_dir), r_squared_global)

    rendered_global_pngs = sorted(glob.glob(f"{str(global_stats_dir)}/*.png"))
    if not rendered_global_pngs:
        raise RuntimeError("Server PNG rendering produced zero .png files in results/global_stats/")

    # ---------------------------------------------------------------------
    # 3) Render LOCAL stats PNG outputs per site
    # ---------------------------------------------------------------------
    site_stats = global_results.get("site_stats", {})
    rendered_local_pngs = []

    if isinstance(site_stats, dict) and site_stats:
        for site_name, payload in site_stats.items():
            if not isinstance(payload, dict):
                continue
            site_dir = local_stats_root / str(site_name)
            _render_local_stats_for_site(
                site_name=str(site_name),
                site_payload=payload,
                cache_dict_for_cnh=cache_dict_for_cnh,
                cache_dir=agg_cache_dir_abs,
                out_dir=str(site_dir),
            )

        rendered_local_pngs = sorted(glob.glob(f"{str(local_stats_root)}/*/*.png"))

    # ---------------------------------------------------------------------
    # 4) Write index.html into results/
    # ---------------------------------------------------------------------
    index_html_path = results_dir / "index.html"

    index_payload = {
        "global_stats": [f"global_stats/{os.path.basename(p)}" for p in rendered_global_pngs],
        "local_stats": [
            f"local_stats/{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}"
            for p in rendered_local_pngs
        ],
    }

    _persist_html_from_results_server(index_payload, str(index_html_path))

    # ---------------------------------------------------------------------
    # 5) Write results.json into results/
    # ---------------------------------------------------------------------
    results_json_path = results_dir / "results.json"

    global_results_out = dict(global_results)
    global_results_out["included_artifacts"] = included
    global_results_out["rendered_global_pngs"] = index_payload["global_stats"]
    global_results_out["rendered_local_pngs"] = index_payload["local_stats"]

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(global_results_out), f, indent=2)

    # ---------------------------------------------------------------------
    # 6) Zip contents of results/ into output_dir/results.zip
    # ---------------------------------------------------------------------
    results_zip_path = out_dir / "results.zip"
    _zip_dir_contents(str(results_dir), str(results_zip_path))

    # NEW: Upload results.zip to NeuroFLAME fileServer
    upload_meta: Dict[str, Any] = {}
    try:
        upload_meta = upload_results_to_fileserver(str(results_zip_path), fl_ctx=fl_ctx)
        if upload_meta.get("uploaded"):
            logger.info("Uploaded results.zip to fileServer successfully")
        else:
            logger.warning("results.zip upload did not complete: %s", upload_meta)
    except Exception as e:
        # Never hard-fail the run just because upload failed (clients may still read from output dir)
        logger.exception("results.zip upload threw exception: %s", e)
        upload_meta = {"uploaded": False, "reason": "exception", "error": str(e)}

    logger.info("Wrote results to output dir: %s", str(out_dir))

    return {
        "output_dir": str(out_dir),
        "results_dir": str(results_dir),
        "results_zip": str(results_zip_path),
        "index_html": str(index_html_path),
        "results_json": str(results_json_path),
        "upload_meta": upload_meta,
    }