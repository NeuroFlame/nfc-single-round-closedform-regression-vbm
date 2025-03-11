import os
import zipfile
import logging
from typing import List

# Set up logging
logger = logging.getLogger(__name__)

def prepare_hosting_directory(source_dir: str, target_dir: str, exclude: List[str]) -> None:
    # Ensure target_dir exists
    os.makedirs(target_dir, exist_ok=True)

    directories = [
        name for name in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, name)) and name not in exclude
    ]

    for folder_name in directories:
        output_zip_path = os.path.join(target_dir, f'{folder_name}.zip')
        folder_path = os.path.join(source_dir, folder_name)
        create_zip_from_folder(folder_path, output_zip_path)

def create_zip_from_folder(source_folder: str, output_zip_path: str) -> None:
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=source_folder)
                zipf.write(file_path, arcname)
    logger.info(f'Created zip: {output_zip_path}')

# Example usage:
# prepare_hosting_directory('/path/to/sourceDir', '/path/to/targetDir', ['exclude_folder1', 'exclude_folder2'])
