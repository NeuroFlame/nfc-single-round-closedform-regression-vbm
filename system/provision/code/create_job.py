import os
import shutil
import json
from typing import Dict, Any

def generate_job_meta(min_clients: int) -> Dict[str, Any]:
    return {
        "resource_spec": {},
        "min_clients": min_clients,
        "deploy_map": {
            "app": ["@ALL"]
        }
    }

def create_job(app_path: str, job_path: str, min_clients: int) -> None:
    if not os.path.isdir(app_path):
        raise FileNotFoundError(f"Source app path '{app_path}' does not exist.")
    
    # Prepare the destination path for the app
    job_app_path = os.path.join(job_path, 'app')
    os.makedirs(job_app_path, exist_ok=True)

    # Copy the app directory
    shutil.copytree(app_path, job_app_path, dirs_exist_ok=True)

    # Generate and write job_meta to meta.json
    job_meta = generate_job_meta(min_clients)
    with open(os.path.join(job_path, 'meta.json'), 'w') as meta_file:
        json.dump(job_meta, meta_file, indent=2)

# Example usage:
# create_job('/path/to/app_folder', '/path/to/job_folder', min_clients=2)
