import yaml
import logging
from typing import List

# Set up logging
logger = logging.getLogger(__name__)

def generate_project_file(
    project_name: str,
    host_identifier: str,
    fed_learn_port: int,
    admin_port: int,
    output_file_path: str,
    site_names: List[str]
) -> None:
    # Define the structure of the YAML content according to the provided specifications
    data = {
        'api_version': 3,
        'name': project_name,
        'participants': [
            {
                'name': host_identifier,
                'org': 'nvidia',
                'type': 'server',
                'admin_port': admin_port,
                'fed_learn_port': fed_learn_port,
            },
            {
                'name': 'admin@admin.com',
                'org': 'nvidia',
                'type': 'admin',
                'role': 'project_admin',
            },
            *[
                {
                    'name': site_name,
                    'org': 'nvidia',
                    'type': 'client',
                } for site_name in site_names
            ]
        ],
        'builders': [
            {
                'path': 'nvflare.lighter.impl.workspace.WorkspaceBuilder',
                'args': {
                    'template_file': 'master_template.yml',
                },
            },
            {
                'path': 'nvflare.lighter.impl.template.TemplateBuilder',
            },
            {
                'path': 'nvflare.lighter.impl.static_file.StaticFileBuilder',
                'args': {
                    'config_folder': 'config',
                    'overseer_agent': {
                        'path': 'nvflare.ha.dummy_overseer_agent.DummyOverseerAgent',
                        'overseer_exists': False,
                        'args': {
                            'sp_end_point': f'{host_identifier}:{fed_learn_port}:{admin_port}',
                        },
                    },
                },
            },
            {'path': 'nvflare.lighter.impl.cert.CertBuilder'},
            {'path': 'nvflare.lighter.impl.signature.SignatureBuilder'},
        ],
        'description': 'project yaml file',
    }

    # Convert the data structure to a YAML formatted string using safe_dump
    yaml_content = yaml.safe_dump(data, sort_keys=False)

    # Write the YAML content to the specified output file path
    try:
        with open(output_file_path, 'w', encoding='utf8') as file:
            file.write(yaml_content)
        logger.info(f"Project file generated successfully at: {output_file_path}")
    except Exception as error:
        logger.error(f"Failed to generate project file at {output_file_path}: {error}")
        raise  # Propagate the error for further handling

# Example usage:
# generate_project_file("MyProject", "example.com", 8000, 9000, "output.yaml", ["site1", "site2"])
