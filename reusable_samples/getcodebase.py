import os

def is_allowed_extension(file, allowed_extensions):
    return any(file.lower().endswith(ext) for ext in allowed_extensions)

def process_file(file_path, output_file, separator):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            output_file.write(f"<{file_path}>\n{content}\n{separator}\n\n\n")
        print(f"Processed: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_directory(directory, output_file, separator, allowed_extensions):
    for root, _, files in os.walk(directory):
        for file in files:
            if is_allowed_extension(file, allowed_extensions):
                file_path = os.path.join(root, file)
                process_file(file_path, output_file, separator)

def main():
    # Configuration
    output_filename = 'codebase.txt'
    separator = '-' * 80  # 80 dashes as a separator
    allowed_extensions = ['.py', '.json', '.md', '.env']

    # Hardcoded lists
    folders_to_process = ['.venv/Lib/site-packages/graphrag/config']
    files_to_process = [
        'indexing.py',
        'requirements.txt',
        'search.py',
        'upload.py',
        'apis.py',
        'chat.py'
        
    ]

    #print current directory
    print(os.getcwd())
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