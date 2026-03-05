import os
import shutil
import json
from typing import Dict, Any, List, Optional


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _uniq(items: Optional[List[str]]) -> List[str]:
    if not items:
        return []
    seen = set()
    out: List[str] = []
    for x in items:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def _normalize_roles(
    user_ids: List[str],
    user_roles: Dict[str, str],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for uid in user_ids:
        s_uid = str(uid)
        role = str(user_roles.get(s_uid, "observer")).strip().lower()
        out[s_uid] = "contributor" if role == "contributor" else "observer"
    return out


def generate_job_meta(
    contributors: List[str],
    observers: List[str],
    min_clients: int,
    server_site_name: str = "server",
) -> Dict[str, Any]:
    """
    Multi-app deployment (NVFlare rule: max ONE app per site):
      - app_server   -> server only
      - app_contrib  -> contributor clients only
      - app_observer -> observer clients only
    """
    contributors = _uniq(contributors)
    observers = _uniq(observers)

    return {
        "resource_spec": {},
        "min_clients": int(min_clients),
        "deploy_map": {
            "app_server": [server_site_name],
            "app_contrib": contributors,
            "app_observer": observers,
        },
    }


def _apply_client_config(app_dir: str, which: str) -> None:
    """
    Materialize the exact filename NVFlare expects:

        config/config_fed_client_<which>.json  ->  config/config_fed_client.json

    where <which> is:
      - "contributor"
      - "observer"

    Your repo should include BOTH templates:
      - config_fed_client_contributor.json
      - config_fed_client_observer.json
    """
    if which not in ("contributor", "observer"):
        raise ValueError(f"which must be 'contributor' or 'observer', got: {which}")

    cfg_dir = os.path.join(app_dir, "config")
    src = os.path.join(cfg_dir, f"config_fed_client_{which}.json")
    dst = os.path.join(cfg_dir, "config_fed_client.json")

    if not os.path.isfile(src):
        raise FileNotFoundError(
            f"Missing template client config: {src}\n"
            f"Expected templates:\n"
            f"  - {os.path.join(cfg_dir, 'config_fed_client_contributor.json')}\n"
            f"  - {os.path.join(cfg_dir, 'config_fed_client_observer.json')}"
        )

    shutil.copyfile(src, dst)


def create_job(
    app_path: str,
    job_path: str,
    min_clients: int,
    user_ids: List[str],
    user_roles: Dict[str, str],  # keyed by user_id: "contributor"|"observer"
    server_site_name: str = "server",
    observer_app_path: Optional[str] = None,  # optional separate observer template app
) -> None:
    """
    Creates a multi-app NVFlare job directory:

      job/
        meta.json
        app_server/...
          config/participant_roles.json
        app_contrib/...
          config/config_fed_client.json (materialized from config_fed_client_contributor.json)
        app_observer/...
          config/config_fed_client.json (materialized from config_fed_client_observer.json)

    Notes:
      - participant_roles.json is written into app_server/config so your server controller can read it.
      - min_clients MUST equal the number of contributors (not total clients).
      - Each site receives exactly ONE app (prevents "Multiple apps to be deployed to ... ['server']").
    """
    if not os.path.isdir(app_path):
        raise FileNotFoundError(f"Source app path '{app_path}' does not exist.")

    user_ids = [str(u) for u in user_ids]
    normalized_roles = _normalize_roles(user_ids, user_roles)

    contributors: List[str] = [uid for uid in user_ids if normalized_roles[uid] == "contributor"]
    observers: List[str] = [uid for uid in user_ids if normalized_roles[uid] == "observer"]

    min_clients = int(min_clients)
    if min_clients < 0:
        raise ValueError("min_clients must be >= 0")

    # Require observers to be present at run start so they receive app_observer
    required = len(contributors) + len(observers)
    if min_clients != required:
        raise ValueError(
            f"min_clients ({min_clients}) must equal contributors+observers ({required}) "
            f"for observer-required runs."
        )

    os.makedirs(job_path, exist_ok=True)

    # 1) app_server (server workflow/controller lives here)
    app_server_path = os.path.join(job_path, "app_server")
    shutil.copytree(app_path, app_server_path, dirs_exist_ok=True)

    # Write participant roles where the server/controller can read it
    cfg_dir = os.path.join(app_server_path, "config")
    os.makedirs(cfg_dir, exist_ok=True)

    roles_payload = {
        "contributors": contributors,
        "observers": observers,
        # IMPORTANT: keys must match NVFlare site names exactly (your user_ids)
        "user_roles": normalized_roles,
    }
    _write_json(os.path.join(cfg_dir, "participant_roles.json"), roles_payload)

    # 2) app_contrib (contributors)
    app_contrib_path = os.path.join(job_path, "app_contrib")
    shutil.copytree(app_path, app_contrib_path, dirs_exist_ok=True)
    _apply_client_config(app_contrib_path, "contributor")

    # 3) app_observer (observers)
    app_observer_path = os.path.join(job_path, "app_observer")
    if observer_app_path:
        if not os.path.isdir(observer_app_path):
            raise FileNotFoundError(f"Observer app path '{observer_app_path}' does not exist.")
        shutil.copytree(observer_app_path, app_observer_path, dirs_exist_ok=True)
        # Still apply observer client config to ensure observer behavior
        _apply_client_config(app_observer_path, "observer")
    else:
        shutil.copytree(app_path, app_observer_path, dirs_exist_ok=True)
        _apply_client_config(app_observer_path, "observer")

    # meta.json (one app per site)
    meta = generate_job_meta(
        contributors,
        observers,
        min_clients=min_clients,
        server_site_name=server_site_name,
    )
    _write_json(os.path.join(job_path, "meta.json"), meta)
