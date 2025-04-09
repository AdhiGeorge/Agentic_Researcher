
import streamlit as st
from dotenv import load_dotenv
import sys
import os

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from src.frontend.app import run_app

if __name__ == "__main__":
    run_app()
