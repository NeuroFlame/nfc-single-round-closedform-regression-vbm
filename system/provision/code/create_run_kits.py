# create_run_kits.py
import os
import shutil
import json
import logging
from typing import List, Dict, Optional

from .create_job import create_job

# Set up logging
logger = logging.getLogger(__name__)


def _normalize_role(role: Optional[str]) -> str:
    r = (role or "observer").strip().lower()
    return "contributor" if r == "contributor" else "observer"


def _copy_directory(src: str, dest: str) -> None:
    """Deterministic overwrite."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _resolve_run_kit_root(site_kit_root: str) -> str:
    """
    Resolve the directory that NeuroFLAME will mount as /workspace/runKit.

    NeuroFLAME canonical (expected):
        <site_kit_root>/startup/...

    Compatibility fallback (only if the kit already contains it):
        <site_kit_root>/runKit/startup/...

    IMPORTANT:
      - We prefer the NeuroFLAME canonical layout.
      - We do NOT create nested runKit folders in provisioning.
    """
    # Prefer NeuroFLAME canonical layout
    direct = site_kit_root
    if os.path.isdir(os.path.join(direct, "startup")):
        return direct

    # Fallback for legacy/alternate kit layouts
    nested = os.path.join(site_kit_root, "runKit")
    if os.path.isdir(nested) and os.path.isdir(os.path.join(nested, "startup")):
        logger.warning(
            "Detected nested kit layout under '<site>/runKit/startup'. "
            "NeuroFLAME canonical layout is '<site>/startup'. Using nested layout for compatibility.",
            extra={"site_kit_root": site_kit_root, "resolved": nested},
        )
        return nested

    raise FileNotFoundError(
        f"Could not resolve runKit root under: {site_kit_root}\n"
        f"Expected either:\n"
        f"  - {os.path.join(site_kit_root, 'startup')}\n"
        f"  - {os.path.join(site_kit_root, 'runKit', 'startup')}"
    )


def _write_participant_role_json(run_kit_root: str, role: str) -> None:
    """
    Write role metadata used by NeuroFLAME edge launcher to decide mount policy.

    Canonical location:
        <run_kit_root>/startup/participant_role.json

    Payload contains BOTH keys for compatibility:
      - "role" (used by current runStart.ts)
      - "participant_role" (allowed for future migration/clarity)
    """
    role_norm = _normalize_role(role)

    payload = {
        "role": role_norm,
        "participant_role": role_norm,
    }

    startup_dir = os.path.join(run_kit_root, "startup")
    os.makedirs(startup_dir, exist_ok=True)

    out_path = os.path.join(startup_dir, "participant_role.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Wrote participant_role.json to: {out_path}")


def create_run_kits(
    path_app: str,
    user_ids: List[str],
    user_roles: Dict[str, str],
    startup_kits_path: str,
    output_directory: str,
    computation_parameters: str,
    host_identifier: str,
    admin_name: str,
    neuroflame_context: Dict[str, str] = None,
) -> None:
    """
    Provision run artifacts for NeuroFLAME + NVFlare.

    Per-site:
      - Copy startup kit folder into: runKits/<site_name>
      - Write: <mounted runKit>/startup/participant_role.json
        (used by NeuroFLAME edge launcher to decide whether to mount /workspace/data)

    Central node bundle:
      - centralNode/job         (created by create_job.py)
      - centralNode/server      (startup kit)
      - centralNode/admin       (startup kit)
      - centralNode/parameters.json

    Contract with create_job.py:
      - create_job.py must generate a VALID NVFlare job definition under centralNode/job,
        including:
          - meta.json with deploy_map assigning exactly ONE app per site
          - app_* folders consistent with that deploy_map
          - correct client config materialization (config_fed_client.json)
    """
    logger.info("Running create_run_kits command")
    os.makedirs(output_directory, exist_ok=True)

    # Normalize roles and split
    normalized_roles: Dict[str, str] = {}
    contributors: List[str] = []
    observers: List[str] = []

    for uid in user_ids:
        s_uid = str(uid)
        role = _normalize_role(user_roles.get(s_uid))
        normalized_roles[s_uid] = role
        if role == "contributor":
            contributors.append(s_uid)
        else:
            observers.append(s_uid)

    logger.info(f"Contributors: {contributors}")
    logger.info(f"Observers: {observers}")

    # Copy requested site startup kits + inject participant_role.json
    for uid in user_ids:
        s_uid = str(uid)
        role = normalized_roles[s_uid]

        src = os.path.join(startup_kits_path, s_uid)
        if not os.path.isdir(src):
            available = sorted(
                d for d in os.listdir(startup_kits_path)
                if os.path.isdir(os.path.join(startup_kits_path, d))
            )
            raise FileNotFoundError(
                f"Expected startup kit folder for site '{s_uid}' at: {src}\n"
                f"Available folders: {available}"
            )

        dest_site_kit_root = os.path.join(output_directory, s_uid)
        _copy_directory(src, dest_site_kit_root)

        run_kit_root = _resolve_run_kit_root(dest_site_kit_root)
        _write_participant_role_json(run_kit_root, role)

    # Central node bundle
    central_node_path = os.path.join(output_directory, "centralNode")
    os.makedirs(central_node_path, exist_ok=True)

    # Job bundle
    job_path = os.path.join(central_node_path, "job")
    create_job(
        app_path=path_app,
        job_path=job_path,
        min_clients=(len(contributors) + len(observers)),
        user_ids=[str(u) for u in user_ids],
        user_roles=normalized_roles,
        server_site_name="server",
        observer_app_path=None,
    )

    # Server kit
    server_startup_kit_path = os.path.join(startup_kits_path, host_identifier)
    if not os.path.isdir(server_startup_kit_path):
        available = sorted(
            d for d in os.listdir(startup_kits_path)
            if os.path.isdir(os.path.join(startup_kits_path, d))
        )
        raise FileNotFoundError(
            f"Server startup kit folder not found: {server_startup_kit_path}\n"
            f"Available folders: {available}"
        )
    _copy_directory(server_startup_kit_path, os.path.join(central_node_path, "server"))

    # Admin kit
    admin_startup_kit_path = os.path.join(startup_kits_path, admin_name)
    if not os.path.isdir(admin_startup_kit_path):
        available = sorted(
            d for d in os.listdir(startup_kits_path)
            if os.path.isdir(os.path.join(startup_kits_path, d))
        )
        raise FileNotFoundError(
            f"Admin startup kit folder not found: {admin_startup_kit_path}\n"
            f"Available folders: {available}"
        )
    _copy_directory(admin_startup_kit_path, os.path.join(central_node_path, "admin"))

    # Parameters
    # computation_parameters is a JSON string passed from NeuroFLAME.
    # We augment it with a `neuroflame` block so the NVFlare server can later upload results.zip to fileServer
    # without needing extra env var injection.
    params_obj: Dict[str, Any]
    try:
        params_obj = json.loads(computation_parameters) if computation_parameters else {}
        if not isinstance(params_obj, dict):
            params_obj = {"value": params_obj}
    except Exception:
        # fallback: keep original string under a key
        params_obj = {"raw": computation_parameters}

    nf_ctx = neuroflame_context or {}
    if nf_ctx:
        # Normalize keys
        params_obj.setdefault("neuroflame", {})
        if isinstance(params_obj["neuroflame"], dict):
            params_obj["neuroflame"].update(nf_ctx)
        else:
            params_obj["neuroflame"] = dict(nf_ctx)

    with open(os.path.join(central_node_path, "parameters.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(params_obj))

    logger.info("RunKits created successfully.")
