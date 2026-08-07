"""Create an NVFlare simulation job from the computation application.

High-level script for computation authors to create a job folder for NVFlare simulation.
This script assumes a specific directory structure and injects min_clients into the workflow config.
Not intended as a general-purpose NVFlare job creation utility.

Usage (from project root):
    python makeJob.py site1,site2,site3

This will:
    - Remove and recreate the 'job' folder
    - Copy server/client configs
    - Set min_clients in the first workflow to the number of sites
    - Generate meta.json for the job
"""

import argparse
import json
import os
import shutil
import sys

JOB_FOLDER = "job"
SERVER_FOLDER = "server"
SERVER_CONFIG_SRC = "app/config/config_fed_server.json"
CLIENT_CONFIG_SRC = "app/config/config_fed_client.json"


def get_spec_task_names():
    """Load the computation spec and return its generated site task names."""
    code_path = os.path.abspath("app/code")
    sys.path.insert(0, code_path)
    try:
        from computation.spec import SPEC
        from framework.workflow import get_task_names
    finally:
        if sys.path and sys.path[0] == code_path:
            sys.path.pop(0)

    return get_task_names(SPEC.workflow)


def parse_sites_arg(sites_arg):
    """Parse a comma-separated site list into normalized site names."""
    if " " in sites_arg:
        raise ValueError(
            "Please provide site names as a single comma-separated string, e.g. --sites=site1,site2"
        )
    return [s.strip() for s in sites_arg.split(",") if s.strip()]


def create_job_folder(job_folder):
    """Replace the generated job directory with an empty directory."""
    if os.path.exists(job_folder):
        shutil.rmtree(job_folder)
    os.makedirs(job_folder)


def create_server_folder(job_folder):
    """Create the server app directory and copy its configuration."""
    server_app_folder = os.path.join(job_folder, SERVER_FOLDER)
    config_dir = os.path.join(server_app_folder, "config")
    os.makedirs(config_dir)
    config_dst = os.path.join(config_dir, "config_fed_server.json")
    shutil.copy(SERVER_CONFIG_SRC, config_dst)
    return config_dst


def update_min_clients_in_workflow(config_path, n_sites):
    """Set the workflow's minimum-client count in server configuration."""
    with open(config_path, "r+") as f:
        config = json.load(f)
        if "workflows" in config and len(config["workflows"]) > 0:
            if "args" not in config["workflows"][0]:
                config["workflows"][0]["args"] = {}
            config["workflows"][0]["args"]["min_clients"] = n_sites
        f.seek(0)
        json.dump(config, f, indent=2)
        f.truncate()


def update_tasks_in_client_config(config_path, task_names):
    """Set generated computation task names in client configuration."""
    with open(config_path, "r+") as f:
        config = json.load(f)
        if "executors" in config and len(config["executors"]) > 0:
            config["executors"][0]["tasks"] = task_names
        f.seek(0)
        json.dump(config, f, indent=2)
        f.truncate()


def create_client_folders(job_folder, sites, task_names):
    """Create one configured client app directory for every site."""
    for site in sites:
        client_app_folder = os.path.join(job_folder, site)
        config_dir = os.path.join(client_app_folder, "config")
        os.makedirs(config_dir)
        config_path = os.path.join(config_dir, "config_fed_client.json")
        shutil.copy(CLIENT_CONFIG_SRC, config_path)
        update_tasks_in_client_config(config_path, task_names)


def create_meta_json(job_folder, job_name, sites):
    """Write NVFlare job metadata and its deployment map."""
    meta = {
        "name": job_name,
        "deploy_map": {"server": ["server"], **{site: [site] for site in sites}},
    }
    with open(os.path.join(job_folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    """Create a local simulation job from command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create NVFlare job folder structure and meta.json."
    )
    parser.add_argument(
        "sites",
        type=str,
        help="Comma-separated list of client site names, e.g. site1,site2",
    )
    args = parser.parse_args()

    sites = parse_sites_arg(args.sites)
    n_sites = len(sites)
    job_name = JOB_FOLDER
    task_names = get_spec_task_names()

    create_job_folder(JOB_FOLDER)
    server_config_path = create_server_folder(JOB_FOLDER)
    update_min_clients_in_workflow(server_config_path, n_sites)
    create_client_folders(JOB_FOLDER, sites, task_names)
    create_meta_json(JOB_FOLDER, job_name, sites)

    print(f"Job folder '{JOB_FOLDER}' created with {n_sites} client(s) and meta.json.")


if __name__ == "__main__":
    main()
