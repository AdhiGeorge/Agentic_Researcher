"""
Fix agent imports script for Agentic Researcher

This script automatically updates all agent files to use absolute imports
and adds the project root to the Python path for direct execution.
"""

import os
import re
import glob

def fix_imports_in_file(file_path):
    """Fix relative imports in an agent file to enable direct execution"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the file already has the project_root code
    if "project_root = os.path.abspath" in content:
        print(f"File {file_path} already fixed. Skipping.")
        return
    
    # Add imports for os and sys if they don't exist
    os_import_line = "import os\n"
    sys_import_line = "import sys\n"
    
    if "import os" not in content:
        # Find the import section and add os import
        import_section = re.search(r"(import .*?)(?=\n\n|\n[^\n])", content, re.DOTALL)
        if import_section:
            content = content.replace(import_section.group(1), import_section.group(1) + os_import_line)
        else:
            # If no import section found, add it to the top after any docstring
            docstring_match = re.search(r'^(""".*?"""\n)', content, re.DOTALL)
            if docstring_match:
                content = content.replace(docstring_match.group(1), docstring_match.group(1) + os_import_line)
            else:
                # Add to the very top if no docstring
                content = os_import_line + content
    
    if "import sys" not in content:
        # Find the import section and add sys import
        import_section = re.search(r"(import .*?)(?=\n\n|\n[^\n])", content, re.DOTALL)
        if import_section:
            content = content.replace(import_section.group(1), import_section.group(1) + sys_import_line)
        else:
            # If no import section found, add it after os import
            if "import os" in content:
                content = content.replace("import os\n", "import os\n" + sys_import_line)
            else:
                # Add to the top if no os import
                docstring_match = re.search(r'^(""".*?"""\n)', content, re.DOTALL)
                if docstring_match:
                    content = content.replace(docstring_match.group(1), docstring_match.group(1) + sys_import_line)
                else:
                    # Add to the very top if no docstring
                    content = sys_import_line + content
    
    # Add project root to Python path
    project_root_code = """
# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
"""
    
    # Find a good insertion point after imports but before code
    imports_done = False
    
    # Look for an existing comment about imports
    import_comment_match = re.search(r'(# .*?import.*?\n)', content)
    if import_comment_match:
        content = content.replace(import_comment_match.group(1), project_root_code)
        imports_done = True
    
    # If no import comment found, insert after import section
    if not imports_done:
        import_section_match = re.search(r'((?:import|from) .*?\n\n)', content, re.DOTALL)
        if import_section_match:
            content = content.replace(import_section_match.group(1), import_section_match.group(1) + project_root_code)
            imports_done = True
    
    # If still not inserted, try after os and sys imports
    if not imports_done and "import sys" in content:
        import_sys_match = re.search(r'(import sys.*?\n)', content)
        if import_sys_match:
            content = content.replace(import_sys_match.group(1), import_sys_match.group(1) + project_root_code)
            imports_done = True
    
    # Replace relative imports with absolute imports
    content = re.sub(r'from \.\.(\.?)([a-zA-Z0-9_]+)', r'from src.\2', content)
    content = re.sub(r'from \.(\.?)([a-zA-Z0-9_]+)', r'from src.agents.\2', content)
    
    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed imports in {file_path}")

def main():
    """Main function to fix imports in all agent files"""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = script_dir
    
    # Find all agent files
    agent_files = []
    for root, _, files in os.walk(os.path.join(project_root, 'src', 'agents')):
        for file in files:
            if file.endswith('.py'):
                agent_files.append(os.path.join(root, file))
    
    print(f"Found {len(agent_files)} agent files to process")
    
    # Fix imports in each file
    for file_path in agent_files:
        fix_imports_in_file(file_path)

if __name__ == "__main__":
    main()
