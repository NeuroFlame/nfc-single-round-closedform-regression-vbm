"""Resolve data, output, and parameter paths for local and hosted runs."""

import logging
import os

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext


def find_repo_root_path() -> str:
    """Find the nearest ancestor containing the app and system directories."""
    path = os.getcwd()
    while not all(
        os.path.isdir(os.path.join(path, name)) for name in ("system", "app")
    ):
        parent = os.path.dirname(path)
        if parent == path:
            raise FileNotFoundError("Repo root directory could not be found.")
        path = parent
    return path


def get_data_directory_path(fl_ctx: FLContext) -> str:
    """Resolve the current site's data directory."""
    env_path = os.getenv("DATA_DIR")
    if env_path and os.path.exists(env_path):
        logging.info(f"Data directory path from environment: {env_path}")
        return env_path

    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
    data_path = os.path.join(find_repo_root_path(), "test_data", str(site_name))
    if os.path.exists(data_path):
        logging.info(f"Data directory path for simulator and poc: {data_path}")
        return data_path
    raise FileNotFoundError("Data directory path could not be determined.")


def get_output_directory_path(fl_ctx: FLContext) -> str:
    """Resolve and create the current participant's output directory."""
    env_path = os.getenv("OUTPUT_DIR")
    if env_path:
        os.makedirs(env_path, exist_ok=True)
        logging.info(f"Output directory path from environment: {env_path}")
        return env_path

    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME) or "server"
    job_id = str(fl_ctx.get_job_id())
    output_path = os.path.join(
        find_repo_root_path(), "test_output", job_id, str(site_name)
    )
    os.makedirs(output_path, exist_ok=True)
    logging.info(f"Output directory path for simulator and poc: {output_path}")
    return output_path


def get_parameters_file_path(fl_ctx: FLContext) -> str:
    """Resolve the computation parameters JSON file."""
    env_path = os.getenv("PARAMETERS_FILE_PATH")
    if env_path and os.path.exists(env_path):
        logging.info(f"Parameters file path from environment: {env_path}")
        return env_path

    parameters_path = os.path.join(
        find_repo_root_path(), "test_data", "server", "parameters.json"
    )
    if os.path.exists(parameters_path):
        logging.info(f"Parameters file path for simulator and poc: {parameters_path}")
        return parameters_path
    raise FileNotFoundError("Parameters file path could not be determined.")
