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
    user_ids = provision_input.get('user_ids')
    computation_parameters = provision_input.get('computation_parameters')
    fed_learn_port = provision_input.get('fed_learn_port')
    admin_port = provision_input.get('admin_port')
    host_identifier = provision_input.get('host_identifier')
    
    print(f'user_ids: {user_ids}')
    print(f'path_run: {path_run}')
    print(f'computation_parameters: {computation_parameters}')
    print(f'fed_learn_port: {fed_learn_port}')
    print(f'admin_port: {admin_port}')
    print(f'host_identifier: {host_identifier}')

    
    # Call the provision_run function with the loaded arguments
    provision_run(
        user_ids=user_ids,
        path_run=path_run,
        path_app="/workspace/app",
        computation_parameters=computation_parameters,
        fed_learn_port=fed_learn_port,
        admin_port=admin_port,
        host_identifier=host_identifier,
    )

if __name__ == '__main__':
    main()
