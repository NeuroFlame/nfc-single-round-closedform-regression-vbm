"""Generate the NVFlare project configuration used for provisioning."""

import logging
from typing import List

import yaml

# Set up logging
logger = logging.getLogger(__name__)


def generate_project_file(
    project_name: str,
    host_identifier: str,
    fed_learn_port: int,
    output_file_path: str,
    site_names: List[str],
) -> None:
    """Write a project YAML file for the server, admin, and sites."""
    # Define the structure of the YAML content according to the provided specifications
    data = {
        "api_version": 3,
        "name": project_name,
        "participants": [
            {
                "name": host_identifier,
                "org": "nvidia",
                "type": "server",
                "fed_learn_port": fed_learn_port,
            },
            {
                "name": "admin@admin.com",
                "org": "nvidia",
                "type": "admin",
                "role": "project_admin",
            },
            *[
                {
                    "name": site_name,
                    "org": "nvidia",
                    "type": "client",
                }
                for site_name in site_names
            ],
        ],
        "builders": [
            {
                "path": "nvflare.lighter.impl.workspace.WorkspaceBuilder",
                "args": {
                    "template_file": "master_template.yml",
                },
            },
            {
                "path": "nvflare.lighter.impl.static_file.StaticFileBuilder",
                "args": {
                    "config_folder": "config",
                    "scheme": "grpc",
                },
            },
            {"path": "nvflare.lighter.impl.cert.CertBuilder"},
            {"path": "nvflare.lighter.impl.signature.SignatureBuilder"},
        ],
        "description": "project yaml file",
    }

    # Convert the data structure to a YAML formatted string using safe_dump
    yaml_content = yaml.safe_dump(data, sort_keys=False)

    # Write the YAML content to the specified output file path
    try:
        with open(output_file_path, "w", encoding="utf8") as file:
            file.write(yaml_content)
        logger.info(f"Project file generated successfully at: {output_file_path}")
    except Exception as error:
        logger.error(f"Failed to generate project file at {output_file_path}: {error}")
        raise  # Propagate the error for further handling


# Example usage:
# generate_project_file("MyProject", "example.com", 8000, "output.yaml", ["site1", "site2"])
