import os
import glob
from fnmatch import fnmatch

# Base directory
base_dir = 'D:/projects/samples/'
output_file = 'D:/temp/tmp_codebase/sample_codebase.txt'

# Patterns to match
patterns = [
    '**/*.py',
    '**/*.tsx',
    '**/*.ts',
    '**/*.json',
    '**/*.js',
    '**/*.html',
    '**/*.css',
    '**/*.md',
    '**/*.md',
    '**/*.gitignore',
    '**/README.md',
    '**/example.env',  # Include example environment files
]

# Files or directories to exclude
exclude_patterns = [
    '**/package-lock.json',  # Exclude package-lock.json
    '**/__pycache__/**',
    '**/node_modules/**',
    '**/.git/**',
    '**/.vscode/**',
    '**/dist/**',
    '**/*.log',
    '**/*.pyc',
    '**/*.pyo',
    'LICENSE',
    '**/.env',          # Exclude actual .env files
    '**/*.env',         # Exclude any other env files with sensitive data
    '**/env/**',
    '**/venv/**',
    '**/venv*',         # Exclude virtual environments
    '**/*.egg-info/**',
    '**/build/**',
    '**/dist/**',
]

header = """You are helping me develop my application. I will provide you my codebase and ask questions or request changes.  Here is my codebase:

"""

footer = """\n<end codebase> Before generating the code, give a brief summary of what you are planning to do, then generate.
When generating the code, you don't always have to generate the full file, if its html then always give the full thing.
Keep it concise and focus on the updated code. I just want the specific code changes, a brief explanation, and nothing else. 

Please help me with the following: 


"""

# Collect files matching patterns
files = []
for pattern in patterns:
    files.extend(glob.glob(os.path.join(base_dir, pattern), recursive=True))

# Remove duplicates
files = list(set(files))

# Filter out excluded files
def is_excluded(file_path):
    for pattern in exclude_patterns:
        if fnmatch(file_path, pattern):
            return True
    return False

files = [f for f in files if not is_excluded(f)]

# Specify the output file path


# Open the output file in write mode
with open(output_file, 'w', encoding='utf-8', errors='ignore') as f:
    # Write the header
    f.write(header)
    
    # Iterate over all collected files
    for filepath in files:
        relative_path = os.path.relpath(filepath, base_dir)
        # Open the file in read mode
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as code_file:
            # Write the filename to the output file
            f.write(f"<File: {relative_path}>\n")
            # Write the code from the file to the output file
            f.write(code_file.read())
            # Add a separator between files
            f.write('\n' + '-'*80 + '\n')
    
    # Write the footer
    f.write(footer)