import json
import os
import logging
import sys
import argparse
from provision.code.provision_run import provision_run

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

def load_provision_input(provision_input_path: str) -> dict:
    try:
        with open(provision_input_path, 'r') as file:
            provision_input = json.load(file)
        logger.info(f"Provision input loaded from {provision_input_path}")
        return provision_input
    except Exception as e:
        logger.error(f"Failed to load provision input: {e}")
        raise

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run provisioning script")
    
    # Input provision file argument
    parser.add_argument(
        '--input',
        default='/provisioning/provision_input.json',
        help='Path to the provision input file (default: /provisioning/provision_input.json)'
    )


    args = parser.parse_args()
    provision_input_path = args.input
    
    # Load provision input
    provision_input = load_provision_input(provision_input_path)
    path_run = os.path.join('/provisioning')

    # Extract arguments from provision input
    user_ids = provision_input.get('user_ids', [])
    user_roles = provision_input.get('user_roles', {}) 
    computation_parameters = provision_input.get('computation_parameters')
    fed_learn_port = provision_input.get('fed_learn_port')
    admin_port = provision_input.get('admin_port')
    host_identifier = provision_input.get('host_identifier')

    # Optional NeuroFLAME fileServer context (for results distribution)
    # These are passed through to runKits/parameters.json so the NVFlare server can upload results.zip
    neuroflame_context = {
        "file_server_url": provision_input.get("fileServerUrl") or provision_input.get("fileserver_url") or provision_input.get("file_server_url"),
        "consortium_id": provision_input.get("consortiumId") or provision_input.get("consortium_id"),
        "run_id": provision_input.get("runId") or provision_input.get("run_id"),
        "token": provision_input.get("downloadToken") or provision_input.get("uploadToken") or provision_input.get("token"),
    }
    # Drop empty values
    neuroflame_context = {k: v for k, v in neuroflame_context.items() if v}
    if neuroflame_context:
        logger.info(f"neuroflame_context keys present: {list(neuroflame_context.keys())}")


    logger.info(f"user_ids: {user_ids}")
    logger.info(f"user_roles: {user_roles}")
    logger.info(f'computation_parameters: {computation_parameters}')
    logger.info(f"path_run: {path_run}")
    logger.info(f"fed_learn_port: {fed_learn_port}")
    logger.info(f"admin_port: {admin_port}")
    logger.info(f"host_identifier: {host_identifier}")

    # Call the provision_run function with the loaded arguments
    provision_run(
        user_ids=user_ids,
        user_roles=user_roles,
        path_run=path_run,
        path_app="/workspace/app",
        computation_parameters=computation_parameters,
        fed_learn_port=fed_learn_port,
        admin_port=admin_port,
        host_identifier=host_identifier,
        neuroflame_context=neuroflame_context,

    )

if __name__ == '__main__':
    main()
