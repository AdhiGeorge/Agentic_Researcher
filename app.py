#!/usr/bin/env python
"""
Agentic Researcher - Streamlit UI Application Entry Point

This file serves as the entry point for the Streamlit UI application.
It's kept separate from main.py which handles console functionality.

Usage:
    streamlit run app.py
"""
import os
import sys
import streamlit as st

# Add project root to Python path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the Streamlit UI components for the full implementation
from src.ui.streamlit_app import (
    initialize_session_state, 
    load_ui_config, 
    load_projects,
    create_ui,
    create_research_tab,
    create_history_tab,
    create_settings_tab
)

# Import main function directly from streamlit_app
from src.ui.streamlit_app import main

# Run the app directly
if __name__ == "__main__":
    main()