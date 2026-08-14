"""Assemble server, admin, and site startup kits for one run."""

import json
import logging
import os
import shutil
from typing import List

from .create_job import create_job

# Set up logging
logger = logging.getLogger(__name__)

_SERVER_RUNTIME_CLASSES = (
    "runtime.aggregator.RuntimeAggregator",
    "runtime.controller.RuntimeController",
    "framework.artifact_transfer.ArtifactTransfer",
)
_CLIENT_RUNTIME_CLASSES = (
    "runtime.executor.RuntimeExecutor",
    "framework.artifact_transfer.ArtifactTransfer",
)
_SERVER_ADMIN_TIMEOUT_SECONDS = 120.0


def create_run_kits(
    path_app: str,
    user_ids: List[str],
    startup_kits_path: str,
    output_directory: str,
    computation_parameters: str,
    host_identifier: str,
    admin_name: str,
) -> None:
    """Create complete central and site run-kit directories."""
    logger.info("Running create_run_kits command")

    try:
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Get site directories excluding the host_identifier and adminName
        site_directories = [
            name
            for name in os.listdir(startup_kits_path)
            if os.path.isdir(os.path.join(startup_kits_path, name))
            and name not in [host_identifier, admin_name]
        ]

        logger.info(f"Found site directories: {site_directories}")

        # Copy each site's startupKit to the runKits directory
        for site in site_directories:
            source_path = os.path.join(startup_kits_path, site)
            destination_path = os.path.join(output_directory, site)
            logger.info(f"Copying {source_path} to {destination_path}")
            disable_site_log_streaming(source_path)
            copy_directory(source_path, destination_path)
            extend_component_allow_list(destination_path, _CLIENT_RUNTIME_CLASSES)
            disable_site_log_streaming(destination_path)
            write_computation_parameters(destination_path, computation_parameters)

        # Create the central node runKit
        central_node_path = os.path.join(output_directory, "centralNode")
        os.makedirs(central_node_path, exist_ok=True)
        logger.info(f"Created central node directory at {central_node_path}")
        job_path = os.path.join(central_node_path, "job")
        create_job(path_app, job_path, min_clients=len(user_ids))

        # Copy the server's startupKit to the central node runKit
        server_startup_kit_path = os.path.join(startup_kits_path, host_identifier)
        copy_directory(
            server_startup_kit_path, os.path.join(central_node_path, "server")
        )
        configure_server_admin_timeout(os.path.join(central_node_path, "server"))
        extend_component_allow_list(
            os.path.join(central_node_path, "server"), _SERVER_RUNTIME_CLASSES
        )
        logger.info(
            f"Copied server startup kit from {server_startup_kit_path} to {central_node_path}/server"
        )

        # Copy the admin's startupKit to the central node runKit
        admin_startup_kit_path = os.path.join(startup_kits_path, admin_name)
        copy_directory(admin_startup_kit_path, os.path.join(central_node_path, "admin"))
        logger.info(
            f"Copied admin startup kit from {admin_startup_kit_path} to {central_node_path}/admin"
        )

        write_computation_parameters(central_node_path, computation_parameters)

        logger.info("RunKits created successfully.")
    except Exception as error:
        logger.error(f"Error creating runKits: {error}")
        raise  # Rethrow or handle as needed


# Helper function to copy directories recursively
def copy_directory(src: str, dest: str) -> None:
    """Replace a destination directory with a recursive source copy."""
    if os.path.exists(dest):
        shutil.rmtree(dest)  # Remove existing destination directory
        logger.info(f"Removed existing directory at {dest}")
    shutil.copytree(src, dest)
    logger.info(f"Copied directory from {src} to {dest}")


def write_computation_parameters(kit_path: str, computation_parameters: str) -> None:
    """Write the shared computation parameters into a server or client run kit."""
    parameters_path = os.path.join(kit_path, "parameters.json")
    with open(parameters_path, "w", encoding="utf-8") as parameters_file:
        parameters_file.write(computation_parameters)
    logger.info(f"Created computation parameters at {parameters_path}")


def configure_server_admin_timeout(kit_path: str) -> None:
    """Allow enough time to deploy computation jobs over participant networks."""
    config_path = os.path.join(kit_path, "startup", "fed_server.json")
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    servers = config.get("servers")
    if not isinstance(servers, list) or not servers:
        raise ValueError(f"Invalid servers list in '{config_path}'")
    for server in servers:
        if not isinstance(server, dict):
            raise TypeError(f"Invalid server entry in '{config_path}'")
        server["admin_timeout"] = _SERVER_ADMIN_TIMEOUT_SECONDS
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")


def extend_component_allow_list(kit_path: str, class_paths: tuple[str, ...]) -> None:
    """Authorize reviewed NeuroFlame runtime classes in a provisioned site kit."""
    resources_path = os.path.join(kit_path, "local", "resources.json.default")
    with open(resources_path, encoding="utf-8") as resources_file:
        resources = json.load(resources_file)
    allow_list = resources.setdefault("class_allow_list", [])
    if not isinstance(allow_list, list):
        raise TypeError(f"Invalid class_allow_list in '{resources_path}'")
    for class_path in class_paths:
        if class_path not in allow_list:
            allow_list.append(class_path)
    resources["class_list_enforcement_mode"] = "enforce"
    with open(resources_path, "w", encoding="utf-8") as resources_file:
        json.dump(resources, resources_file, indent=2)
        resources_file.write("\n")


def disable_site_log_streaming(kit_path: str) -> None:
    """Keep participant logs local unless a deployment explicitly opts in."""
    resources_path = os.path.join(kit_path, "local", "resources.json.default")
    with open(resources_path, encoding="utf-8") as resources_file:
        resources = json.load(resources_file)
    resources["allow_log_streaming"] = False
    with open(resources_path, "w", encoding="utf-8") as resources_file:
        json.dump(resources, resources_file, indent=2)
        resources_file.write("\n")


# Example usage:
# create_run_kits('/path/to/startupKits', '/path/to/outputDirectory', '{"param": "value"}', 'example.com', 'admin@admin.com')
