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

# Import the Streamlit app module with the correct path
from src.ui.streamlit_app import main

# Call the main function from the streamlit app
if __name__ == "__main__":
    import asyncio
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    from streamlit.runtime.scriptrunner.script_run_context import add_script_run_ctx
    
    # Create and run the async event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Get the Streamlit script context
    ctx = get_script_run_ctx()
    
    # Run the main function
    task = loop.create_task(main())
    add_script_run_ctx(task, ctx)
    loop.run_until_complete(task)

"""
Example Usage:

1. From the command line, navigate to the project directory:
   cd path/to/Agentic-Researcher

2. Run the Streamlit application:
   streamlit run app.py

3. The application will open in your default web browser. From there, you can:
   - Enter research queries in the main tab
   - View research history in the History tab
   - Configure API keys and settings in the Settings tab

Example Query: "What are the latest advancements in quantum computing?"

This will trigger a multi-agent research process that:
   1. Plans the research approach using the planner agent
   2. Conducts web searches and extracts information using researcher agent
   3. Synthesizes findings using the answer agent
   4. Presents a comprehensive answer in the UI
"""
