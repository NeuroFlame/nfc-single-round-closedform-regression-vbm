import os
import logging
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

def find_workspace_path() -> str:
    """Find the workspace directory starting from the current working directory and going up."""
    current_path = os.getcwd()

    while True:
        candidate_path = os.path.join(current_path, "workspace")
        if os.path.isdir(candidate_path):
            logging.info(f"Workspace directory found at: {os.path.abspath(candidate_path)}")
            return os.path.abspath(candidate_path)
        
        parent_path = os.path.dirname(current_path)
        if current_path == parent_path:
            break  # Reached the root directory
        
        current_path = parent_path

    raise FileNotFoundError("Workspace directory could not be found.")

def get_data_directory_path(fl_ctx: FLContext) -> str:
    """Determine and return the data directory path based on the available paths."""
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)

    # Check if the environment variable DATA_DIR is set
    env_path = os.getenv("DATA_DIR")
    if env_path and os.path.exists(env_path):
        logging.info(f"Data directory path from environment: {env_path}")
        return env_path

    # If DATA_DIR is not set, use the simulator and poc path
    workspace_path = find_workspace_path()
    simulator_and_poc_path = os.path.join(workspace_path, f"test_data/{site_name}")
    if os.path.exists(simulator_and_poc_path):
        logging.info(f"Data directory path for simulator and poc: {simulator_and_poc_path}")
        return simulator_and_poc_path

    raise FileNotFoundError("Data directory path could not be determined.")

def get_output_directory_path(fl_ctx: FLContext) -> str:
    """Determine and return the output directory path based on the available paths."""
    job_id = fl_ctx.get_job_id()
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)

    # Check if the environment variable OUTPUT_DIR is set
    env_path = os.getenv("OUTPUT_DIR")
    if env_path:
        os.makedirs(env_path, exist_ok=True)
        logging.info(f"Output directory path from environment: {env_path}")
        return env_path

    # If OUTPUT_DIR is not set, use the simulator and poc path
    workspace_path = find_workspace_path()
    simulator_and_poc_path = os.path.join(workspace_path, f"test_output/{job_id}/{site_name}")
    os.makedirs(simulator_and_poc_path, exist_ok=True)
    logging.info(f"Output directory path for simulator and poc: {simulator_and_poc_path}")
    return simulator_and_poc_path

def get_parameters_file_path(fl_ctx: FLContext) -> str:
    """Determine and return the parameters file path based on the available paths."""

    # Check if the environment variable PARAMETERS_FILE_PATH is set and the file exists
    env_path = os.getenv("PARAMETERS_FILE_PATH")
    if env_path and os.path.exists(env_path):
        logging.info(f"Parameters file path from environment: {env_path}")
        return env_path

    # If PARAMETERS_FILE_PATH is not set, use the simulator and poc path
    workspace_path = find_workspace_path()
    simulator_and_poc_path = os.path.abspath(os.path.join(workspace_path, "test_data/server/parameters.json"))
    if os.path.exists(simulator_and_poc_path):
        logging.info(f"Parameters file path for simulator and poc: {simulator_and_poc_path}")
        return simulator_and_poc_path

    raise FileNotFoundError("Parameters file path could not be determined.")
