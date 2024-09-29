"""
File Processing Script for AI-assisted coding

This script processes multiple code files and combines their contents into a single output file,
making it easier to use with AI-powered code analysis tools or language models.

Features:
- Process individual files or entire directories recursively
- Filter files based on allowed extensions
- Combine file contents into a single output file with clear separators
- Flexible mode selection: 'files' for specific files, 'folders' for directory processing

Usage:
1. Run the script: python script_name.py
2. Choose the processing mode:
   - 'files': Process specific files listed in `files_to_process`
   - 'folders': Process directories recursively
3. If 'folders' mode is selected, choose between:
   - 'root': Process the entire project directory
   - Enter: Process predefined folders listed in `folders_to_process`
4. The script will create 'codebase.txt' containing the processed file contents

Configuration:
- output_filename: Name of the output file (default: 'codebase.txt')
- separator: String used to separate file contents (default: 80 dashes)
- allowed_extensions: List of file extensions to process
- folders_to_process: List of folders to process in 'folders' mode
- files_to_process: List of specific files to process in 'files' mode

Note: Ensure you have the necessary permissions to read the input files and write to the output file.
"""


import os
from typing import List, TextIO

def is_allowed_extension(file: str, allowed_extensions: List[str]) -> bool:
    """
    Check if the file has an allowed extension.

    Args:
        file (str): The filename to check.
        allowed_extensions (List[str]): List of allowed file extensions.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    return any(file.lower().endswith(ext) for ext in allowed_extensions)

def process_file(file_path: str, output_file: TextIO, separator: str) -> None:
    """
    Process a single file and write its contents to the output file.

    Args:
        file_path (str): Path to the file to process.
        output_file (TextIO): The output file object to write to.
        separator (str): The separator string to use between files.

    Raises:
        Exception: If there's an error reading or writing the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            output_file.write(f"<{file_path}>\n{content}\n{separator}\n\n\n")
        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory: str, output_file: TextIO, separator: str, allowed_extensions: List[str]) -> None:
    """
    Recursively process all files in a directory with allowed extensions.

    Args:
        directory (str): The directory to process.
        output_file (TextIO): The output file object to write to.
        separator (str): The separator string to use between files.
        allowed_extensions (List[str]): List of allowed file extensions.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if is_allowed_extension(file, allowed_extensions):
                file_path = os.path.join(root, file)
                process_file(file_path, output_file, separator)

def main():
    """
    Main function to run the file processing script.
    """
    # Configuration
    output_filename = 'D:/temp/tmp_codebase/codebase.txt'
    separator = '-' * 80  # 80 dashes as a separator
    allowed_extensions = ['.py', '.json', '.md', '.env']

    # Hardcoded lists
    folders_to_process = ['D:/projects/samples/reusable_samples']
    files_to_process = [
        'indexing.py',
        'aoai.py',
        'document_processing.py',
        'cosmosdb.py',
        'search.py'
    ]

    print(f"Current working directory: {os.getcwd()}")

    # Choose mode
    mode = input("Enter 'files' for specific files or 'folders' for recursive folder processing: ").strip().lower()

    with open(output_filename, 'w', encoding='utf-8') as output_file:
        if mode == 'files':
            for file_path in files_to_process:
                if os.path.isfile(file_path):
                    process_file(file_path, output_file, separator)
                else:
                    print(f"File not found: {file_path}")
        elif mode == 'folders':
            root_option = input("Enter 'root' to process the entire project, or press Enter to use predefined folders: ").strip().lower()
            if root_option == 'root':
                process_directory('.', output_file, separator, allowed_extensions)
            else:
                for folder in folders_to_process:
                    if os.path.isdir(folder):
                        print(f"Processing folder: {folder}")
                        process_directory(folder, output_file, separator, allowed_extensions)
                    else:
                        print(f"Directory not found: {folder}")
        else:
            print("Invalid mode selected. Please run the script again and choose 'files' or 'folders'.")

    print(f"File contents have been written to {output_filename}")

if __name__ == "__main__":
    main()