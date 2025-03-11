import os
import logging
from .generate_project_file import generate_project_file
from .create_startup_kits import create_startup_kits
from .create_run_kits import create_run_kits
from typing import List


# Set up logging
logger = logging.getLogger(__name__)

def provision_run(
    user_ids: List[str],
    path_run: str,
    path_app: str,
    computation_parameters: str,
    fed_learn_port: int,
    admin_port: int,
    host_identifier: str,
) -> None:
    # Configurable variables
    admin_name = 'admin@admin.com'

    path_startup_kits = os.path.join(path_run, 'startupKits/')
    path_run_kits = os.path.join(path_run, 'runKits/')
    

    # Ensure all necessary directories are created
    ensure_directory_exists(path_run)
    ensure_directory_exists(path_startup_kits)
    ensure_directory_exists(path_run_kits)

    generate_project_file(
        project_name='project',
        host_identifier=host_identifier,
        fed_learn_port=fed_learn_port,
        admin_port=admin_port,
        output_file_path=os.path.join(path_run, 'Project.yml'),
        site_names=user_ids,
    )

    create_startup_kits(
        project_file_path=os.path.join(path_run, 'Project.yml'),
        output_directory=path_startup_kits,
    )

    create_run_kits(
        path_app=path_app,
        user_ids=user_ids,
        startup_kits_path=os.path.join(path_startup_kits, 'project', 'prod_00'),
        output_directory=path_run_kits,
        computation_parameters=computation_parameters,
        host_identifier=host_identifier,
        admin_name=admin_name,
    )

def ensure_directory_exists(directory_path: str) -> None:
    try:
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Directory ensured: {directory_path}")
    except Exception as error:
        logger.error(f"Failed to ensure directory: {directory_path} with error: {error}")
        raise

# Example usage:
# provision_run(['user1', 'user2'], '/path/to/run', '{"param": "value"}', 8000, 9000, 'example.com')
