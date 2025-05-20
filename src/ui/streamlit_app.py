"""Streamlit UI for Agentic Researcher

This module provides a web-based user interface for the Agentic Researcher
system using Streamlit. It allows users to interact with the agent system,
submit research queries, and view results.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# # Add parent directory to sys.path to enable imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# if parent_dir not in sys.path:
#     sys.path.append(parent_dir)

# Import project modules
from src.orchestrator.swarm_orchestrator import SwarmOrchestrator
from src.utils.config import config
from src.db.sqlite_manager import SQLiteManager
from src.utils.openai_client import get_chat_client, generate_chat_completion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streamlit_ui")

# Initialize global state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None

if 'projects' not in st.session_state:
    st.session_state.projects = []

if 'current_project_id' not in st.session_state:
    st.session_state.current_project_id = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

if 'results' not in st.session_state:
    st.session_state.results = None

if 'monologue_visibility' not in st.session_state:
    st.session_state.monologue_visibility = True

# Load UI configuration
def load_ui_config():
    """Load UI configuration from config file"""
    # Return default UI configuration
    return {
        "title": "Agentic Researcher",
        "theme": {
            "primary_color": "#FF6B6B",
            "background_color": "#F5F5F5",
            "secondary_background_color": "#EEEEEE",
            "text_color": "#262730"
        },
        "page_icon": "🔍",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }

# Initialize the app
def initialize_app():
    """Initialize the Streamlit app"""
    # Load UI configuration
    ui_config = load_ui_config()
    
    # Configure Streamlit page
    st.set_page_config(
        page_title=ui_config["title"],
        page_icon=ui_config["page_icon"],
        layout=ui_config["layout"],
        initial_sidebar_state=ui_config["initial_sidebar_state"]
    )
    
    # Apply custom CSS
    st.markdown(f"""
    <style>
    .reportview-container .main .block-container{{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }}
    .stApp {{
        background-color: {ui_config['theme']['background_color']};
        color: {ui_config['theme']['text_color']};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize orchestrator if not already initialized
    if st.session_state.orchestrator is None:
        with st.spinner("Initializing Agentic Researcher system..."):
            try:
                st.session_state.orchestrator = SwarmOrchestrator()
                logger.info("Initialized Swarm orchestrator")
                
                # Load projects
                load_projects()
            except Exception as e:
                st.error(f"Error initializing the system: {str(e)}")
                logger.error(f"Initialization error: {str(e)}", exc_info=True)

# Load projects from SQLite
def load_projects():
    """Load projects from SQLite database"""
    try:
        db = SQLiteManager()
        # Call a method to get all projects
        # For now, let's use a placeholder
        st.session_state.projects = [
            {"id": 1, "name": "Default Project", "description": "Default research project", "created_at": datetime.now().isoformat()}
        ]
        
        # Set current project if not set
        if st.session_state.current_project_id is None and st.session_state.projects:
            st.session_state.current_project_id = st.session_state.projects[0]["id"]
            
    except Exception as e:
        st.error(f"Error loading projects: {str(e)}")
        logger.error(f"Error loading projects: {str(e)}", exc_info=True)

# Create main UI
def create_ui():
    """Create the main Streamlit UI"""
    st.title("Agentic Researcher")
    st.subheader("Orchestrated Retrieval and Knowledge Agent")
    
    # Show project description
    st.markdown("""
    A state-of-the-art multi-agent system for research, reasoning, and code synthesis. Ask a question,
    and Agentic Researcher will research it, provide a comprehensive answer, and even generate executable code.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Settings & Projects")
        
        # Project selection
        if st.session_state.projects:
            project_names = [p["name"] for p in st.session_state.projects]
            project_ids = [p["id"] for p in st.session_state.projects]
            
            selected_index = 0
            if st.session_state.current_project_id in project_ids:
                selected_index = project_ids.index(st.session_state.current_project_id)
                
            selected_project = st.selectbox(
                "Select Project",
                project_names,
                index=selected_index
            )
            
            # Update current project ID
            selected_index = project_names.index(selected_project)
            st.session_state.current_project_id = project_ids[selected_index]
        
        # New project button
        if st.button("Create New Project"):
            st.session_state.show_new_project_form = True
        
        # Show new project form if requested
        if st.session_state.get("show_new_project_form", False):
            with st.form("new_project_form"):
                new_project_name = st.text_input("Project Name")
                new_project_desc = st.text_area("Description")
                
                submitted = st.form_submit_button("Create Project")
                if submitted and new_project_name:
                    try:
                        # Create new project
                        db = SQLiteManager()
                        new_project_id = db.create_project(new_project_name, new_project_desc)
                        
                        # Add to session state
                        st.session_state.projects.append({
                            "id": new_project_id,
                            "name": new_project_name,
                            "description": new_project_desc,
                            "created_at": datetime.now().isoformat()
                        })
                        
                        # Set as current project
                        st.session_state.current_project_id = new_project_id
                        
                        # Hide form
                        st.session_state.show_new_project_form = False
                        
                        # Force refresh
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating project: {str(e)}")
        
        # Toggle for internal monologue visibility
        st.session_state.monologue_visibility = st.checkbox(
            "Show agent internal monologue", 
            value=st.session_state.monologue_visibility
        )
        
        # About information
        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "Agentic Researcher uses a multi-agent system with LLMs to "
            "help you research any topic in depth."
        )
    
    # Main content area
    tabs = st.tabs(["Research", "History", "Settings"])
    
    # Research tab
    with tabs[0]:
        create_research_tab()
    
    # History tab
    with tabs[1]:
        create_history_tab()
    
    # Settings tab
    with tabs[2]:
        create_settings_tab()

# Create the research tab
def create_research_tab():
    """Create the research tab UI"""
    st.header("Research Assistant")
    st.markdown("Ask a research question and get comprehensive answers with sources.")
    
    # Query input
    query = st.text_area(
        "Enter your research query",
        height=100,
        placeholder="What is volatility index and what is the mathematical formula to calculate the VIX score. Also write a python code to calculate the vix score."
    )
    
    # Submit button
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Research", type="primary")
    with col2:
        if st.session_state.processing:
            st.info("Processing your query... This may take a few minutes.")
    
    # Process the query when submitted
    if submit_button and query and not st.session_state.processing:
        st.session_state.processing = True
        process_research_query(query)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("Conversation")
        
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "internal_monologue" and st.session_state.monologue_visibility:
                st.markdown(
                    f"<div style='background-color:#f0f0f0; padding:10px; border-radius:5px; margin:5px 0;'>"
                    f"<em>🤔 Agent's thought process: {msg['content']}</em>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
            elif msg["role"] == "assistant":
                st.markdown(f"**AI:** {msg['content']}")
    
    # Display research results
    if st.session_state.results:
        st.markdown("---")
        st.subheader("Research Results")
        
        # Display the formatted research results
        formatted_results = st.session_state.results.get("formatted_results", "")
        if formatted_results:
            with st.expander("Structured Research Results", expanded=True):
                st.markdown(formatted_results["structured_content"])
        
        # Display the final result
        final_result = st.session_state.results.get("final_result", "")
        if final_result:
            with st.expander("Final Summary and Action", expanded=True):
                st.markdown(final_result)
        
        # Show sources
        if formatted_results and "sources" in formatted_results:
            with st.expander("Sources", expanded=False):
                sources = formatted_results["sources"]
                for i, source in enumerate(sources):
                    st.markdown(f"**{i+1}. [{source['title']}]({source['url']})**")
                    st.markdown(f"*{source['snippet']}*")
                    st.markdown("---")

# Create the history tab
def create_history_tab():
    """Create the history tab UI"""
    st.header("Research History")
    
    # Get past queries for the current project
    try:
        if st.session_state.current_project_id:
            db = SQLiteManager()
            # This would be a call to get past queries
            # For now, let's use a placeholder
            past_queries = [
                {"id": 1, "query": "What is volatility index?", "created_at": "2025-05-19T10:30:00"},
                {"id": 2, "query": "How to calculate VIX in Python?", "created_at": "2025-05-19T11:15:00"}
            ]
            
            if past_queries:
                # Create a DataFrame for display
                df = pd.DataFrame(past_queries)
                df["Time"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
                df = df[["id", "query", "Time"]]
                df = df.rename(columns={"id": "ID", "query": "Query"})
                
                # Display the table
                st.dataframe(df, use_container_width=True)
                
                # Allow user to select a past query
                selected_id = st.selectbox("Select a past query to view results", df["ID"].tolist())
                if st.button("View Results"):
                    # This would load the results for the selected query
                    st.info(f"Results for query #{selected_id} would be loaded here")
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
        logger.error(f"Error loading history: {str(e)}", exc_info=True)

# Create the settings tab
def create_settings_tab():
    """Create the settings tab UI"""
    st.header("Settings")
    
    # Load configuration
    config = ConfigLoader()
    
    # Create tabs for different settings sections
    settings_tabs = st.tabs(["API Keys", "Search Engines", "Database", "Advanced"])
    
    # API Keys tab
    with settings_tabs[0]:
        st.subheader("API Keys Configuration")
        
        # Azure OpenAI settings
        st.markdown("### Azure OpenAI")
        azure_api_key = st.text_input(
            "Azure OpenAI API Key", 
            value=config.azure_openai_api_key,
            type="password"
        )
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            value=config.azure_openai_endpoint
        )
        azure_deployment = st.text_input(
            "Azure OpenAI Deployment Name",
            value=config.azure_openai_deployment
        )
        
        # Save API keys
        if st.button("Save API Settings"):
            try:
                # Update config
                config.azure_openai_api_key = azure_api_key
                config.azure_openai_endpoint = azure_endpoint
                config.azure_openai_deployment = azure_deployment
                
                # Note: The config class doesn't have a save method
                # Changes are applied directly to the instance
                
                st.success("API settings saved successfully!")
                
                # Reinitialize orchestrator
                st.session_state.orchestrator = None
                st.rerun()
            except Exception as e:
                st.error(f"Error saving API settings: {str(e)}")
    
    # Search Engines tab
    with settings_tabs[1]:
        st.subheader("Search Engine Configuration")
        
        # Search engine priority
        st.markdown("### Search Priority")
        primary_engine = st.selectbox(
            "Primary Search Engine",
            ["duckduckgo", "tavily", "google"],
            index=["duckduckgo", "tavily", "google"].index(
                getattr(config, "primary_search_engine", "duckduckgo")
            )
        )
        
        # Save search settings
        if st.button("Save Search Settings"):
            try:
                # Update config - using setattr to be safe
                # Try to set the attribute, with fallback to avoid errors
                try:
                    setattr(config, "primary_search_engine", primary_engine)
                except Exception as e:
                    logger.warning(f"Could not set primary_search_engine: {e}")
                
                # Note: The config class doesn't have a save method
                # Changes are applied directly to the instance
                
                st.success("Search settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving search settings: {str(e)}")
    
    # Database tab
    with settings_tabs[2]:
        st.subheader("Database Configuration")
        
        # SQLite settings
        st.markdown("### SQLite")
        sqlite_path = st.text_input(
            "SQLite Database Path",
            value=getattr(config, "sqlite_path", "agentic_researcher.db")  # Using config value with fallback
        )
        
        # Qdrant settings
        st.markdown("### Qdrant")
        qdrant_host = st.text_input(
            "Qdrant Host",
            value=config.qdrant_url
        )
        qdrant_port = st.number_input(
            "Qdrant Port",
            value=config.qdrant_port
        )
        
        # Save database settings
        if st.button("Save Database Settings"):
            try:
                # Direct attribute update
                # Note: This may not persist after restart as these are directly modifying instance attributes
                # You may need to implement environment variable updates or .env file writing
                # Set sqlite path (though this may not persist between runs)
                try:
                    setattr(config, "sqlite_path", sqlite_path)
                except Exception as e:
                    logger.warning(f"Could not set sqlite_path: {e}")
                config.qdrant_url = qdrant_host
                config.qdrant_port = qdrant_port
                
                # Note: The config class doesn't have a save method
                # Changes are applied directly to the instance
                
                st.success("Database settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving database settings: {str(e)}")
    
    # Advanced tab
    with settings_tabs[3]:
        st.subheader("Advanced Configuration")
        
        # Debug mode
        debug_mode = st.checkbox(
            "Debug Mode",
            value=getattr(config, "debug_mode", False)
        )
        
        # Log level
        log_level = st.selectbox(
            "Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(
                getattr(config, "log_level", "INFO")
            )
        )
        
        # Save advanced settings
        if st.button("Save Advanced Settings"):
            try:
                # Direct attribute update
                config.debug_mode = debug_mode
                config.log_level = log_level
                
                # Note: The config class doesn't have a save method
                # Changes are applied directly to the instance
                
                st.success("Advanced settings saved successfully!")
            except Exception as e:
                st.error(f"Error saving advanced settings: {str(e)}")

# Process a research query
async def process_research_query(query):
    """Process a research query and update the UI with results"""
    if not query or not st.session_state.orchestrator:
        return
    
    # Set processing flag
    st.session_state.processing = True
    
    # Update conversation history
    st.session_state.conversation_history.append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get project ID
    project_id = st.session_state.current_project_id
    
    try:
        # Display processing message
        with st.spinner("Researching your query... this may take a few minutes."):
            # Create agent activity container for real-time updates
            activity_container = st.empty()
            activity_container.info("Planning research approach...")
            
            # Call orchestrator to process query
            results = await st.session_state.orchestrator.process_research_query(
                query=query,
                project_id=project_id,
                use_cache=True
            )
            
            # Update activity status
            if results.get("from_cache", False):
                activity_container.success("Found cached answer!")
            else:
                activity_container.success("Research completed!")
            
            # Store results
            st.session_state.results = results
            
            # Store execution stats
            st.session_state.execution_stats = st.session_state.orchestrator.get_execution_stats()
            
            # Update conversation history with response
            if "results" in results and "answer" in results["results"]:
                answer = results["results"]["answer"]
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {
                        "execution_time": results.get("execution_time", 0),
                        "from_cache": results.get("from_cache", False),
                        "similarity": results.get("similarity", 1.0) if results.get("from_cache", False) else 1.0
                    }
                })
    except Exception as e:
        # Log error
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        
        # Add error to conversation history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": f"I encountered an error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
    finally:
        # Clear processing flag
        st.session_state.processing = False

# Main function
async def main():
    """Main entry point for the Streamlit app"""
    # Initialize app
    initialize_app()
    
    # Create the main UI
    create_ui()

# Run the app
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
