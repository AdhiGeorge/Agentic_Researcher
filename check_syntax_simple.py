# Simple script to check for syntax errors in streamlit_app.py
import ast

def check_syntax(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            source = file.read()
        
        ast.parse(source)
        print("No syntax errors found in", file_path)
        return True
    except SyntaxError as e:
        print(f"Syntax error in {file_path} at line {e.lineno}, column {e.offset}: {e.msg}")
        if hasattr(e, 'text') and e.text:
            print(f"Problematic line: {e.text.strip()}")
        return False

if __name__ == "__main__":
    check_syntax("src/ui/streamlit_app.py")
