# Quick script to check for syntax errors in streamlit_app.py
try:
    from src.ui.streamlit_app import main
    print("✅ No syntax errors found in streamlit_app.py")
except SyntaxError as e:
    print(f"❌ Syntax error: {e}")
except Exception as e:
    print(f"❗ Other error: {e}")
