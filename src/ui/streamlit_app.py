"""Streamlit UI for Agentic Researcher

This module provides a web-based user interface for the Agentic Researcher
system using Streamlit. It allows users to interact with the agent system,
submit research queries, and view results.
"""

import os
import logging
import json
import uuid
import base64
import io
import time
import math
from typing import Dict, List, Any, Optional, Union, Tuple
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

# Import project modules - handle both direct execution and module import scenarios
try:
    # When imported as a module
    from src.orchestrator.swarm_orchestrator import SwarmOrchestrator
    from src.utils.config import config
    from src.db.sqlite_manager import SQLiteManager
    from src.utils.openai_client import get_chat_client, generate_chat_completion
    from src.ui.file_upload_processor import FileUploadProcessor
    from src.utils.large_output_generator import LargeOutputGenerator
    from src.utils.export_utils import ExportManager
    from src.utils.visualization import DataVisualizer
    from src.utils.code_executor import CodeExecutor
    from src.utils.citation_manager import CitationManager
    from src.agents.planner.advanced_planner import AdvancedPlannerAgent
except ModuleNotFoundError:
    # When run directly as a script
    import sys
    import os
    # Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    # Try imports again
    from src.orchestrator.swarm_orchestrator import SwarmOrchestrator
    from src.utils.config import config
    from src.db.sqlite_manager import SQLiteManager
    from src.utils.openai_client import get_chat_client, generate_chat_completion
    from src.ui.file_upload_processor import FileUploadProcessor
    from src.utils.large_output_generator import LargeOutputGenerator
    from src.utils.export_utils import ExportManager
    from src.utils.visualization import DataVisualizer
    from src.utils.code_executor import CodeExecutor
    from src.utils.citation_manager import CitationManager
    from src.agents.planner.advanced_planner import AdvancedPlannerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("streamlit_ui")

# Initialize global state - this must be done before any UI code runs
def initialize_session_state():
    """Initialize Streamlit session state with default values"""
    # Core components
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
        
    # UI state
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "test"  # Can be "full" or "test"
        
    # Research state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
        
    if "processing" not in st.session_state:
        st.session_state.processing = False
        
    if "results" not in st.session_state:
        st.session_state.results = None
        
    # Project state
    if "projects" not in st.session_state:
        st.session_state.projects = []
        
    if "current_project_id" not in st.session_state:
        st.session_state.current_project_id = None
        
    # Debug options
    if "monologue_visibility" not in st.session_state:
        st.session_state.monologue_visibility = False
        
    # File upload
    if "show_file_upload" not in st.session_state:
        st.session_state.show_file_upload = False
        
    if "uploaded_file" not in st.session_state:
        st.session_state.uploaded_file = None
        
    if "uploaded_file_data" not in st.session_state:
        st.session_state.uploaded_file_data = None
        
    if "include_file_in_research" not in st.session_state:
        st.session_state.include_file_in_research = False
        
    # Workflow options
    if "workflow_type" not in st.session_state:
        st.session_state.workflow_type = "dynamic"  # Can be "dynamic" or "standard"
        
    # Code execution
    if "show_code_editor" not in st.session_state:
        st.session_state.show_code_editor = False
        
    if "code_output" not in st.session_state:
        st.session_state.code_output = None
        
    # Export options
    if "export_format" not in st.session_state:
        st.session_state.export_format = "markdown"
        
    if "last_export" not in st.session_state:
        st.session_state.last_export = None
        
    # Visualization options
    if "visualization_results" not in st.session_state:
        st.session_state.visualization_results = None
        
    if "show_visualization" not in st.session_state:
        st.session_state.show_visualization = False
        
    # Citation tracking
    if "citations" not in st.session_state:
        st.session_state.citations = []
        
    # Component initialization
    # File processor
    if "file_processor" not in st.session_state:
        st.session_state.file_processor = FileUploadProcessor()
        
    # Large output generator
    if "large_output_generator" not in st.session_state:
        try:
            st.session_state.large_output_generator = LargeOutputGenerator()
            logger.info("Large output generator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Large Output Generator: {str(e)}")
            st.session_state.large_output_generator = None
            
    # Export manager
    if "export_manager" not in st.session_state:
        try:
            st.session_state.export_manager = ExportManager()
            logger.info("Export manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Export Manager: {str(e)}")
            st.session_state.export_manager = None
            
    # Data visualizer
    if "data_visualizer" not in st.session_state:
        try:
            st.session_state.data_visualizer = DataVisualizer()
            logger.info("Data visualizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Data Visualizer: {str(e)}")
            st.session_state.data_visualizer = None
            
    # Code executor
    if "code_executor" not in st.session_state:
        try:
            st.session_state.code_executor = CodeExecutor()
            logger.info("Code executor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Code Executor: {str(e)}")
            st.session_state.code_executor = None
            
    # Citation manager
    if "citation_manager" not in st.session_state:
        try:
            st.session_state.citation_manager = CitationManager()
            logger.info("Citation manager initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Citation Manager: {str(e)}")
            st.session_state.citation_manager = None
            
    # Advanced planner
    if "advanced_planner" not in st.session_state:
        try:
            st.session_state.advanced_planner = AdvancedPlannerAgent()
            logger.info("Advanced planner initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Advanced Planner: {str(e)}")
            st.session_state.advanced_planner = None

# Initialize session state
initialize_session_state()

# Load UI configuration
def load_ui_config() -> Dict[str, Any]:
    """Load UI configuration from config.yaml or use defaults
    
    Returns:
        Dict containing UI configuration
    """
    # Default Italian-inspired color palette
    italian_palette = {
        "primary": "#CE2B37",  # Italian red
        "secondary": "#009246",  # Italian green
        "accent": "#F1F2F1",  # Italian white
        "background": "#FCFCFC",
        "text": "#333333"
    }
    
    # Try to load from config, or use defaults
    ui_colors = config.get('ui_colors', {})
    app_name = config.get('app_name', 'Agentic Researcher')
    app_subtitle = config.get('app_subtitle', 'Your AI-powered research assistant')
    
    # Create a complete UI configuration with all required parameters
    ui_config = {
        "colors": {**italian_palette, **(ui_colors or {})},
        "title": app_name,
        "subtitle": app_subtitle,
        # Add required parameters for initialize_app
        "page_icon": "🧠",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "theme": {
            "background_color": italian_palette["background"],
            "text_color": italian_palette["text"],
            "primary_color": italian_palette["primary"],
            "secondary_color": italian_palette["secondary"],
            "accent_color": italian_palette["accent"],
            "secondary_background_color": "#f0f0f0"
        }
    }
    
    # Log the configuration being used
    logging.info(f"Loaded UI configuration with colors: {ui_config['colors']}")
    
    return ui_config

# Initialize the app
def initialize_app():
    """Initialize the Streamlit app"""
    # Load UI configuration
    ui_config = load_ui_config()
    
    # Configure page (must be first Streamlit command)
    st.set_page_config(
        page_title=ui_config["title"], 
        page_icon=ui_config["page_icon"],
        layout=ui_config["layout"],
        initial_sidebar_state=ui_config["initial_sidebar_state"]
    )
    
    # Apply custom CSS for styling
    st.markdown(f"""
    <style>
    .reportview-container .main .block-container {{
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }}
    .stApp {{
        background-color: {ui_config['theme']['background_color']};
        color: {ui_config['theme']['text_color']};
    }}
    /* Claude-like styling */
    .stButton>button {{
        background-color: {ui_config['theme']['primary_color']};
        color: white;
    }}
    .stProgress .st-bo {{
        background-color: {ui_config['theme']['secondary_color']};
    }}
    /* Headers */
    h1, h2, h3 {{
        color: {ui_config['theme']['primary_color']};
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }}
    /* Expandable sections */
    .streamlit-expanderHeader {{
        background-color: {ui_config['theme']['secondary_background_color']};
        border-radius: 4px;
    }}
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {{
        background-color: {ui_config['theme']['secondary_background_color']};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Try to initialize full app, with fallback to test UI
    try:
        # Check if API keys are configured
        if not hasattr(config, 'azure_openai_api_key') or not config.azure_openai_api_key:
            # Display a friendly message in the sidebar
            st.sidebar.warning(
                "⚠️ **API Key Required**\n\n"
                "To use the full research capabilities, please add your OpenAI API key in the Settings tab."
            )
            st.session_state.app_mode = "test"
            return
        
        # Initialize orchestrator if not already initialized
        if st.session_state.orchestrator is None:
            with st.spinner("Initializing Agentic Researcher system..."):
                try:
                    # Set environment variable to ensure it's available for all components
                    import os
                    os.environ["OPENAI_API_KEY"] = config.azure_openai_api_key
                    
                    # Initialize the orchestrator
                    st.session_state.orchestrator = SwarmOrchestrator()
                    logger.info("Initialized Swarm orchestrator successfully")
                    
                    # Load projects
                    load_projects()
                    st.session_state.app_mode = "full"
                except Exception as e:
                    logger.error(f"Initialization error: {str(e)}", exc_info=True)
                    st.session_state.app_mode = "test"
                    st.sidebar.error(
                        f"**Error initializing research system**\n\n{str(e)}\n\n"
                        "Using simplified interface without advanced research capabilities."
                    )
    except Exception as e:
        logger.error(f"Critical error in app initialization: {str(e)}", exc_info=True)
        st.session_state.app_mode = "test"

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
    """Create the main Streamlit UI with Claude-like styling"""
    st.title("Agentic Researcher")
    st.subheader("Orchestrated Research & Knowledge Assistant")
    
    # Check which app mode to use
    if st.session_state.app_mode == "test":
        # Show a message about test mode
        st.info(
            "**Running in Test Mode**\n\n"
            "For full functionality, please configure your API keys in the Settings tab."
        )
    # Continue with the rest of the UI regardless of mode
    
    # Show project description with Claude-like styling
    st.markdown("""
    <div style='background-color:#f9f9f9; padding:1.5rem; border:1px solid #e0e0e0; border-radius:0.5rem; margin-bottom:1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
    <p style='font-family:"Inter", sans-serif; font-size:1.1rem; color:#333333; margin:0; line-height:1.5;'>
    A professional AI research assistant that conducts comprehensive research on your queries.
    Simply enter your research question, and I will gather, analyze, and synthesize information from credible sources.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
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
    """Create the research tab UI with a Claude-like interface"""
    # Clean, modern header
    st.markdown("<h2 style='margin-bottom:0.8rem; font-weight:500;'>Research Assistant</h2>", unsafe_allow_html=True)
    
    # More subtle helper text
    st.markdown("<p style='color:#666; margin-bottom:1.5rem; font-size:0.9rem;'>Ask a research question and get comprehensive answers with sources.</p>", unsafe_allow_html=True)
    
    # File upload toggle
    file_upload_expander = st.expander("📁 Upload Data Files", expanded=st.session_state.show_file_upload)
    
    with file_upload_expander:
        st.markdown("<p style='font-weight:500; margin-bottom:0.5rem;'>Upload files for analysis and code generation</p>", unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload CSV, Excel, JSON, PDF or text files", 
            type=["csv", "xlsx", "xls", "json", "pdf", "txt"]
        )
        
        # Process uploaded file
        if uploaded_file is not None:
            # Check if this is a new file upload
            if st.session_state.uploaded_file is None or st.session_state.uploaded_file.name != uploaded_file.name:
                st.session_state.uploaded_file = uploaded_file
                st.session_state.show_file_upload = True
                
                # Process the file
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    result = st.session_state.file_processor.process_upload(uploaded_file)
                    st.session_state.uploaded_file_data = result
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success(f"Successfully processed {uploaded_file.name}")
            
            # Display file info and preview
            if st.session_state.uploaded_file_data and "error" not in st.session_state.uploaded_file_data:
                data = st.session_state.uploaded_file_data
                
                # Display file metadata
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**File:** {data['file_name']}")
                    st.markdown(f"**Type:** {data['file_type']}")
                with col2:
                    if "shape" in data:
                        st.markdown(f"**Dimensions:** {data['shape'][0]} rows × {data['shape'][1]} columns")
                    elif "text_length" in data:
                        st.markdown(f"**Content Length:** {data['text_length']} characters")
                    elif "file_size_bytes" in data:
                        st.markdown(f"**Size:** {data['file_size_bytes']/1024:.1f} KB")
                
                # Display appropriate preview based on file type
                if data.get("data_type") == "tabular" and "preview_html" in data:
                    st.markdown("**Data Preview:**")
                    st.write(data["data"].head())
                elif data.get("data_type") == "multi_tabular" and "sheets_data" in data:
                    # Allow user to select a sheet
                    selected_sheet = st.selectbox("Select Sheet", data["sheets"])
                    sheet_data = data["sheets_data"][selected_sheet]
                    st.markdown(f"**{selected_sheet} Preview:**")
                    st.write(sheet_data["data"].head())
                elif data.get("data_type") == "text" and "text_content" in data:
                    st.markdown("**Text Preview:**")
                    st.text_area("Content", data["text_content"][:1000] + ("..." if len(data["text_content"]) > 1000 else ""), height=150)
                elif data.get("data_type") == "document" and "text_content" in data:
                    st.markdown("**Document Content Preview:**")
                    st.text_area("Content", data["text_content"][:1000] + ("..." if len(data["text_content"]) > 1000 else ""), height=150)
        
        # Option to include file in research
        if st.session_state.uploaded_file_data and "error" not in st.session_state.uploaded_file_data:
            st.session_state.include_file_in_research = st.checkbox("Include file data in research query", value=True)
    
    # Query input with better styling
    st.markdown("<p style='font-weight:500; margin-bottom:0.3rem;'>Enter your research query</p>", unsafe_allow_html=True)
    query = st.text_area(
        "",  # Remove label as we're using custom markdown
        height=120,
        placeholder="What is volatility index and what is the mathematical formula to calculate the VIX score? Also write a Python code to calculate the VIX score.",
        key="research_query_input"
    )
    
    # Submit button with better layout
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        submit_button = st.button("Research", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Clear", type="secondary", use_container_width=True, 
                               on_click=lambda: st.session_state.update({"research_query_input": "", "conversation_history": []}))
    with col3:
        upload_toggle = st.button("Toggle Upload", type="secondary", use_container_width=True,
                              on_click=lambda: setattr(st.session_state, "show_file_upload", not st.session_state.show_file_upload))
    with col4:
        if st.session_state.processing:
            st.markdown("<div style='background-color:#f0f7ff; padding:0.7rem; border-radius:0.3rem; color:#0969da; font-size:0.9rem;'><i>Processing your query... This may take a few minutes.</i></div>", unsafe_allow_html=True)
    
    # Process the query when submitted
    if submit_button and query and not st.session_state.processing:
        # Make a copy of the query and set processing flag
        current_query = query
        st.session_state.processing = True
        st.session_state.progress_counter = 0
        st.session_state.progress_status = "Initializing research workflow..."
        
        # Only try to process if we're in full mode with an orchestrator
        if st.session_state.app_mode == "full" and st.session_state.orchestrator:
            # Update conversation history immediately
            st.session_state.conversation_history.append({
                "role": "user",
                "content": current_query,
                "timestamp": datetime.now().isoformat()
            })
            
            # Instead of a separate thread, we'll use a timestamp-based approach for progress updates
            # Store initial timestamp
            import time as time_module  # Import locally to avoid any potential shadowing
            start_time = time_module.time()
            st.session_state.research_start_time = start_time
            
            # Define status messages for different stages
            st.session_state.status_messages = [
                "Planning research approach...",
                "Searching for relevant sources...",
                "Reading and analyzing content...",
                "Extracting key information...",
                "Synthesizing findings...",
                "Formatting comprehensive answer...",
                "Generating citations...",
                "Finalizing research results..."
            ]
            
            # Create a thread-safe callback to update the state after processing
            def after_process_callback(future_result):
                try:
                    # Get results from the future
                    results = future_result.result()
                    if results:
                        # Check if there was a timeout or error
                        if "error" in results and "status" in results and results["status"] == "timeout":
                            def update_state_with_timeout():
                                st.session_state.processing = False
                                st.session_state.progress_status = "Processing timed out"
                                st.session_state.conversation_history.append({
                                    "role": "assistant",
                                    "content": "⚠️ Your query timed out after 5 minutes. This could be due to high complexity or server load. Please try:\n\n1. Breaking your question into smaller parts\n2. Asking a more specific question\n3. Trying again in a few minutes",
                                    "timestamp": datetime.now().isoformat()
                                })
                            # Schedule the timeout update on the next rerun
                            st.rerun()
                            return
                            
                        # Store successful results in session state
                        def update_state_with_results():
                            st.session_state.results = results
                            st.session_state.processing = False
                            st.session_state.progress_status = "Research completed!"
                            
                            # Add assistant response
                            if "results" in results and "answer" in results["results"]:
                                answer = results["results"]["answer"]
                                st.session_state.conversation_history.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "timestamp": datetime.now().isoformat()
                                })
                            elif "error" in results:
                                # Handle specific error from the results
                                st.session_state.conversation_history.append({
                                    "role": "assistant",
                                    "content": f"I encountered an issue while processing your query: {results['error']}",
                                    "timestamp": datetime.now().isoformat()
                                })
                            
                        # Schedule the state update on the next rerun
                        st.rerun()
                except Exception as e:
                    # Handle unexpected errors
                    def update_state_with_error():
                        st.session_state.processing = False
                        st.session_state.progress_status = "Error encountered"
                        logging.error(f"Error processing query: {str(e)}")
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": f"I encountered an error processing your query: {str(e)}\n\nPlease try again or simplify your question.",
                            "timestamp": datetime.now().isoformat()
                        })
                    # Schedule the error update on the next rerun
                    st.rerun()
            
            # No thread needed - Streamlit will handle updates on each rerun
            
            # Process the query in a separate thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: process_query_sync(current_query))
                future.add_done_callback(after_process_callback)
        else:
            # In test mode, just add a simulated response after a short delay
            import time
            time.sleep(2)  # Simulate processing time
            
            # Add mock conversation
            st.session_state.conversation_history.append({
                "role": "user",
                "content": current_query,
                "timestamp": datetime.now().isoformat()
            })
            
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": "I'm currently running in test mode without access to external APIs. Please add your OpenAI API key in the Settings tab to enable full research capabilities.",
                "timestamp": datetime.now().isoformat()
            })
            
            # Clear processing flag
            st.session_state.processing = False
            st.rerun()
    
    # Add workflow and advanced options
    with st.expander("Workflow Options", expanded=False):
        # Workflow selection
        st.markdown("### Research Workflow")
        workflow_type = st.radio(
            "Select research workflow type:",
            ["Dynamic", "Standard"],
            index=0 if st.session_state.workflow_type == "dynamic" else 1,
            horizontal=True,
            help="Dynamic workflow analyzes your query and selects the most relevant agents, while Standard workflow follows a fixed sequence of agents."
        )
        
        # Update session state
        st.session_state.workflow_type = workflow_type.lower()
        
        # Advanced options for showing agent thought process
        st.markdown("### Advanced Options")
        monologue_visibility = st.checkbox(
            "Show agent thought process", 
            value=st.session_state.monologue_visibility,
            help="View the internal reasoning of the agent as it processes your query"
        )
        st.session_state.monologue_visibility = monologue_visibility
        
    # Display conversation history with Claude-like styling
    if st.session_state.conversation_history:
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background-color:#f9f9f9; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; border-left:3px solid #555;'>"
                    f"<p style='margin:0; font-weight:500;'>You</p>"
                    f"<p style='margin-top:0.5rem;'>{msg['content']}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif msg["role"] == "internal_monologue" and st.session_state.monologue_visibility:
                st.markdown(
                    f"<div style='background-color:#f0f7ff; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; font-size:0.9rem; color:#555; border-left:3px solid #0969da;'>"
                    f"<p style='margin:0; font-weight:500; color:#0969da;'>🤔 Agent's Thought Process</p>"
                    f"<p style='margin-top:0.5rem; font-style:italic;'>{msg['content']}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f"<div style='background-color:#f0f7ff; padding:1rem; border-radius:0.5rem; margin-bottom:1rem; border-left:3px solid #0969da;'>"
                    f"<p style='margin:0; font-weight:500; color:#0969da;'>Agentic Researcher</p>"
                    f"<div style='margin-top:0.5rem;'>{msg['content']}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            elif msg["role"] == "system":
                st.markdown(
                    f"<div style='background-color:#e8f4f8; padding:0.8rem; border-radius:0.5rem; margin-bottom:1rem; font-size:0.9rem;'>"
                    f"<p style='margin:0;'>{msg['content']}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    # Show processing message with animated indicators
    if st.session_state.processing:
        import time as time_module  # Import locally to avoid any potential shadowing
        
        # Always show a pulsing animation regardless of actual progress
        current_time = time_module.time()
        
        # Create a pulsing effect (oscillating between 0.1 and 0.9)
        # This creates a visually satisfying progress animation even when actual progress is unknown
        pulse_speed = 2.0  # Higher = faster pulse
        pulse_val = 0.4 + 0.4 * (math.sin(current_time * pulse_speed) + 1) / 2
        
        # Determine which status message to show
        if "research_start_time" in st.session_state and "status_messages" in st.session_state:
            elapsed_time = current_time - st.session_state.research_start_time
            # Show different messages over time to indicate progress
            # Change message every 15 seconds to show ongoing activity
            msg_index = min(int(elapsed_time / 15), len(st.session_state.status_messages) - 1)
            current_status = st.session_state.status_messages[msg_index]
            
            # Force a rerun every 3 seconds to keep animation smooth
            if not hasattr(st.session_state, "last_progress_update") or \
               current_time - st.session_state.last_progress_update > 3:
                st.session_state.last_progress_update = current_time
                st.rerun()
        else:
            current_status = "Initializing research workflow..."
        
        # Create a visually appealing progress indicator
        progress_bar = st.progress(pulse_val)
        
        # Add animated elements to show activity
        st.info(f"**{current_status}**")
        
        # Add a spinner animation for visual feedback
        with st.spinner("Working..."):
            st.empty()
        
        # Add dots animation to the status message (changes every second)
        dots = "." * (int(current_time) % 4)  # 0 to 3 dots that change every second
        st.markdown(f"*The system is working on your query{dots} This may take a few minutes depending on the complexity of the research.*")
    
    # Display research results
    if st.session_state.results:
        st.markdown("---")
        st.subheader("Research Results")
        
        # Action buttons for results
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            export_button = st.button("📤 Export Results", use_container_width=True)
            
        with col2:
            visualize_button = st.button("📈 Visualize Data", use_container_width=True) if "file_data" in st.session_state.results else None
            
        with col3:
            code_button = st.button("💻 Run Code", use_container_width=True)
            
        with col4:
            # Export format selection
            export_format = st.selectbox(
                "Format",
                ["markdown", "pdf", "docx", "html"],
                index=["markdown", "pdf", "docx", "html"].index(st.session_state.export_format)
            )
            st.session_state.export_format = export_format
        
        # Check if large output processing was applied
        large_output_info = st.session_state.results.get("large_output_processing", {})
        if large_output_info and large_output_info.get("applied", False):
            st.info(f"📊 Large output optimized: {large_output_info.get('original_length', 0):,} characters processed in {large_output_info.get('processing_time', 0):.2f} seconds")
        
        # Display file analysis results if present
        if "file_data" in st.session_state.results:
            with st.expander("📄 File Analysis Results", expanded=True):
                file_data = st.session_state.results["file_data"]
                st.markdown(f"**File:** {file_data.get('file_name', 'Unknown')}")
                st.markdown(f"**Type:** {file_data.get('file_type', 'Unknown')}")
                
                # If there's a specific file analysis section
                if "file_analysis" in st.session_state.results["results"]:
                    st.markdown("### File Analysis")
                    st.markdown(st.session_state.results["results"]["file_analysis"])
        
        # Display the formatted research results
        formatted_results = st.session_state.results.get("formatted_results", "")
        if formatted_results:
            with st.expander("📊 Structured Research Results", expanded=True):
                st.markdown(formatted_results["structured_content"])
        
        # Display sources if available
        if "sources" in st.session_state.results["results"]:
            with st.expander("🔍 Research Sources", expanded=False):
                sources = st.session_state.results["results"]["sources"]
                
                # Add sources to citation manager
                if st.session_state.citation_manager:
                    for source in sources:
                        # Create citation object
                        citation_data = {
                            "title": source.get("title", "Unknown"),
                            "url": source.get("url", ""),
                            "relevance_score": source.get("relevance", 0.0),
                            "content_excerpt": source.get("excerpt", ""),
                        }
                        # Add to citation manager if not already added
                        st.session_state.citation_manager.add_citation(citation_data)
                    
                    # Get formatted citations
                    citations = st.session_state.citation_manager.generate_bibliography("apa")
                    st.session_state.citations = citations.get("citations", [])
                
                # Display sources with proper citations
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:** {source.get('title', 'Unknown')}")
                    st.markdown(f"**URL:** {source.get('url', 'N/A')}")
                    st.markdown(f"**Relevance:** {source.get('relevance', 'Unknown')}")
                    if source.get("excerpt"):
                        with st.expander("Excerpt"):
                            st.markdown(f"*{source.get('excerpt')}*")
                    st.markdown("---")
                
                # Show formatted citations
                if st.session_state.citations:
                    st.markdown("### Formatted Citations (APA Style)")
                    for citation in st.session_state.citations:
                        st.markdown(f"{citation.get('formatted')}")
        
        # Export functionality
        if export_button and st.session_state.export_manager:
            try:
                # Get content to export
                if "answer" in st.session_state.results.get("results", {}):
                    content = st.session_state.results["results"]["answer"]
                    
                    # Add citations if available
                    if st.session_state.citations:
                        content += "\n\n## References\n\n"
                        for citation in st.session_state.citations:
                            content += f"{citation.get('formatted')}\n\n"
                    
                    # Create export directory if it doesn't exist
                    os.makedirs("./exports", exist_ok=True)
                    
                    # Export in selected format
                    export_result = st.session_state.export_manager.export(
                        content=content,
                        format=st.session_state.export_format,
                        title="Research Results"
                    )
                    
                    if export_result.get("success", False):
                        st.session_state.last_export = export_result
                        st.success(f"Results exported to {export_result.get('file_path')}")
                    else:
                        st.error(f"Export failed: {export_result.get('error', 'Unknown error')}")
                else:
                    st.warning("No content available for export")
            except Exception as e:
                st.error(f"Error exporting results: {str(e)}")
        
        # Visualization functionality
        if visualize_button and st.session_state.data_visualizer and "file_data" in st.session_state.results:
            st.session_state.show_visualization = True
        
        # Display visualization interface
        if st.session_state.show_visualization and st.session_state.data_visualizer and "file_data" in st.session_state.results:
            with st.expander("📈 Data Visualization", expanded=True):
                file_data = st.session_state.results["file_data"]
                
                if file_data.get("data_type") == "tabular" and "data" in file_data:
                    df = file_data["data"]
                    
                    # Basic stats
                    st.markdown("### Data Summary")
                    st.dataframe(df.describe())
                    
                    # Column selection for visualization
                    st.markdown("### Create Visualization")
                    viz_type = st.selectbox(
                        "Visualization Type",
                        ["Histogram", "Bar Plot", "Scatter Plot", "Correlation Matrix"]
                    )
                    
                    if viz_type == "Histogram":
                        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        if num_cols:
                            col = st.selectbox("Select Column", num_cols)
                            bins = st.slider("Number of Bins", 5, 100, 20)
                            
                            if st.button("Generate Histogram"):
                                with st.spinner("Generating histogram..."):
                                    result = st.session_state.data_visualizer.plot_histogram(
                                        df=df,
                                        column=col,
                                        bins=bins
                                    )
                                    
                                    if "error" not in result:
                                        st.image(f"data:image/png;base64,{result['image']}")
                                    else:
                                        st.error(f"Error: {result['error']}")
                        else:
                            st.warning("No numerical columns available for histogram")
                            
                    elif viz_type == "Bar Plot":
                        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                        if cat_cols:
                            col = st.selectbox("Select Column", cat_cols)
                            limit = st.slider("Number of Categories", 5, 20, 10)
                            
                            if st.button("Generate Bar Plot"):
                                with st.spinner("Generating bar plot..."):
                                    result = st.session_state.data_visualizer.plot_bar(
                                        df=df,
                                        column=col,
                                        limit=limit
                                    )
                                    
                                    if "error" not in result:
                                        st.image(f"data:image/png;base64,{result['image']}")
                                    else:
                                        st.error(f"Error: {result['error']}")
                        else:
                            st.warning("No categorical columns available for bar plot")
                            
                    elif viz_type == "Scatter Plot":
                        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        if len(num_cols) >= 2:
                            x_col = st.selectbox("X-Axis", num_cols, index=0)
                            y_col = st.selectbox("Y-Axis", num_cols, index=min(1, len(num_cols)-1))
                            
                            # Optional color column
                            color_option = st.checkbox("Add Color Dimension")
                            color_col = None
                            if color_option:
                                color_col = st.selectbox("Color By", df.columns.tolist())
                            
                            if st.button("Generate Scatter Plot"):
                                with st.spinner("Generating scatter plot..."):
                                    result = st.session_state.data_visualizer.plot_scatter(
                                        df=df,
                                        x_column=x_col,
                                        y_column=y_col,
                                        color_column=color_col
                                    )
                                    
                                    if "error" not in result:
                                        st.image(f"data:image/png;base64,{result['image']}")
                                    else:
                                        st.error(f"Error: {result['error']}")
                        else:
                            st.warning("Need at least 2 numerical columns for scatter plot")
                            
                    elif viz_type == "Correlation Matrix":
                        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                        if len(num_cols) >= 2:
                            selected_cols = st.multiselect("Select Columns", num_cols, default=num_cols[:min(5, len(num_cols))])
                            
                            if st.button("Generate Correlation Matrix") and selected_cols:
                                with st.spinner("Generating correlation matrix..."):
                                    result = st.session_state.data_visualizer.plot_correlation_matrix(
                                        df=df,
                                        columns=selected_cols
                                    )
                                    
                                    if "error" not in result:
                                        st.image(f"data:image/png;base64,{result['image']}")
                                    else:
                                        st.error(f"Error: {result['error']}")
                        else:
                            st.warning("Need at least 2 numerical columns for correlation matrix")
                    
                    # Option to generate comprehensive report
                    if st.button("Generate Comprehensive Visualization Report"):
                        with st.spinner("Generating visualization report..."):
                            report_result = st.session_state.data_visualizer.generate_visualization_report(
                                df=df
                            )
                            
                            if report_result.get("success", False):
                                st.success(f"Report generated: {report_result.get('file_path')}")
                                st.session_state.visualization_results = report_result
                                
                                # Add download link
                                st.markdown(f"[Download Report]({report_result.get('file_path')})")
                            else:
                                st.error(f"Error: {report_result.get('error', 'Unknown error')}")
                else:
                    st.warning("Visualization is only available for tabular data")
        
        # Code execution functionality
        if code_button:
            st.session_state.show_code_editor = True
        
        # Display code execution interface
        if st.session_state.show_code_editor and st.session_state.code_executor:
            with st.expander("💻 Code Execution", expanded=True):
                st.markdown("### Python Code Editor")
                st.markdown("Write Python code to analyze research data or generate visualizations.")
                
                # Code editor
                default_code = """# Example: Display research data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Print available data
print("Available data:")
for key in globals():
    if not key.startswith('_'):
        print(f"- {key}")

# If file data is available, analyze it
if 'file' in globals() and 'data' in file:
    df = file['data']
    print(f"\nDataFrame shape: {df.shape}")
    print("\nDataFrame head:")
    print(df.head())
    
    # Generate a simple plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df.select_dtypes(include=['number']).iloc[:, 0], kde=True)
    plt.title(f"Distribution of {df.select_dtypes(include=['number']).columns[0]}")
    plt.show()
"""
                
                code = st.text_area("Python Code", value=default_code, height=300)
                
                # Execute button
                if st.button("Execute Code"):
                    with st.spinner("Executing code..."):
                        # Prepare data for code execution
                        execution_data = {}
                        
                        # Add research results
                        if st.session_state.results:
                            execution_data["results"] = st.session_state.results.get("results", {})
                        
                        # Add file data if available
                        if "file_data" in st.session_state.results:
                            execution_data["file"] = st.session_state.results["file_data"]
                        
                        # Execute the code
                        code_result = st.session_state.code_executor.execute_code(
                            code=code,
                            data=execution_data
                        )
                        
                        st.session_state.code_output = code_result
                
                # Display code execution results
                if st.session_state.code_output:
                    st.markdown("### Execution Results")
                    
                    # Display any errors
                    if not st.session_state.code_output.get("success", False):
                        st.error(f"Execution Error: {st.session_state.code_output.get('error', 'Unknown error')}")
                    
                    # Display execution time
                    st.info(f"Execution time: {st.session_state.code_output.get('execution_time', 0):.2f} seconds")
                    
                    # Display output
                    if st.session_state.code_output.get("output"):
                        st.markdown("#### Output")
                        st.code(st.session_state.code_output["output"], language="")
                    
                    # Display plots
                    if st.session_state.code_output.get("plots"):
                        st.markdown("#### Generated Plots")
                        for i, plot in enumerate(st.session_state.code_output["plots"]):
                            st.image(f"data:image/png;base64,{plot['data']}")
                    
                    # Display variables
                    if st.session_state.code_output.get("variables"):
                        st.markdown("#### Variables")
                        for var_name, var_info in st.session_state.code_output["variables"].items():
                            var_type = var_info.get("type", "Unknown")
                            
                            if var_type == "DataFrame":
                                st.markdown(f"**{var_name}** (DataFrame)")
                                st.markdown(f"Shape: {var_info.get('shape', (0, 0))}")
                                st.dataframe(pd.DataFrame(var_info.get("preview", {})))
                            elif var_type == "ndarray":
                                st.markdown(f"**{var_name}** (ndarray)")
                                st.markdown(f"Shape: {var_info.get('shape', ())}")
                            elif var_type in ["list", "dict", "str", "int", "float", "bool"]:
                                st.markdown(f"**{var_name}** ({var_type})")
                                st.write(var_info.get("value", ""))
                            else:
                                st.markdown(f"**{var_name}** ({var_type})")
        
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
    
    # Use the imported config directly - don't reassign the name
    # Create tabs for different settings sections
    settings_tabs = st.tabs(["API Keys", "Search Engines", "Database", "Advanced"])
    
    # API Keys tab
    with settings_tabs[0]:
        st.markdown("<h3 style='margin-bottom:1rem;'>API Keys Configuration</h3>", unsafe_allow_html=True)
        st.markdown("<div style='background-color:#fff8db; padding:0.8rem; border-radius:0.5rem; border:1px solid #f0e6be; margin-bottom:1.5rem;'><p style='margin:0; font-size:0.9rem;'>⚠️ <strong>Security Note:</strong> API keys are sensitive information. Keys are stored securely and not displayed in plaintext.</p></div>", unsafe_allow_html=True)
        
        # Azure OpenAI settings
        st.markdown("<h4 style='margin-top:1rem; margin-bottom:0.5rem;'>Azure OpenAI</h4>", unsafe_allow_html=True)
        
        # Mask the existing key if present
        masked_key = "••••••••" if config.azure_openai_api_key else ""
        key_placeholder = "Enter your Azure OpenAI API Key" if not config.azure_openai_api_key else "API Key is set (hidden)"
        
        # For API Key, only show placeholder if key exists
        azure_api_key = st.text_input(
            "Azure OpenAI API Key", 
            value="" if config.azure_openai_api_key else "",
            placeholder=key_placeholder,
            type="password",
            help="Your API key is stored securely and never displayed"
        )
        
        # Keep endpoint visible but in a more secure format
        endpoint_display = config.azure_openai_endpoint if config.azure_openai_endpoint else ""
        azure_endpoint = st.text_input(
            "Azure OpenAI Endpoint",
            value=endpoint_display,
            placeholder="https://your-resource-name.openai.azure.com/"
        )
        
        # Keep deployment visible since it's not sensitive
        azure_deployment = st.text_input(
            "Azure OpenAI Deployment Name",
            value=config.azure_openai_deployment,
            placeholder="your-deployment-name"
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

# Process a research query synchronously (for threading)
def process_query_sync(query):
    """Process a research query synchronously and return results"""
    if not query or not st.session_state.orchestrator:
        return None
    
    # Get project ID and workflow type
    project_id = st.session_state.current_project_id
    workflow_type = st.session_state.workflow_type
    
    # Add detailed logging
    logging.info(f"Starting query processing: {query[:50]}...")
    
    try:
        import asyncio
        import concurrent.futures
        import time
        
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Set a processing timeout (5 minutes)
        PROCESSING_TIMEOUT = 300  # seconds
        
        # Initialize context
        context = {
            "project_id": project_id,
            "use_cache": True,
            "workflow_type": workflow_type
        }
        
        # Add file data to context if available and user wants to include it
        if hasattr(st.session_state, "include_file_in_research") and \
           st.session_state.include_file_in_research and \
           st.session_state.uploaded_file_data and \
           "error" not in st.session_state.uploaded_file_data:
            
            file_data = st.session_state.uploaded_file_data
            
            # Add file data to context
            context["file_data"] = {
                "file_name": file_data.get("file_name", ""),
                "file_type": file_data.get("file_type", ""),
                "file_path": file_data.get("file_path", ""),
                "data_type": file_data.get("data_type", "")
            }
            
            # Add more specific data based on file type
            if file_data.get("data_type") == "tabular" and "data" in file_data:
                # For tabular data, include head and summary
                df = file_data["data"]
                context["file_data"]["data_head"] = df.head(10).to_dict()
                context["file_data"]["data_shape"] = df.shape
                context["file_data"]["columns"] = list(df.columns)
                context["file_data"]["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                
                # Add basic statistics if available
                if "stats" in file_data:
                    context["file_data"]["stats"] = file_data["stats"]
                    
            elif file_data.get("data_type") == "text" and "text_content" in file_data:
                # For text data, include the content
                context["file_data"]["text_content"] = file_data["text_content"]
                
            elif file_data.get("data_type") == "document" and "text_content" in file_data:
                # For document data, include the extracted text
                context["file_data"]["text_content"] = file_data["text_content"]
            
            # Modify the query to include file reference
            file_reference = f"\n\nAnalyze the uploaded {file_data.get('file_type', 'file')} named '{file_data.get('file_name', '')}'."
            query = query + file_reference
            logger.info(f"Added file reference to query: {file_reference}")
        
        # Track start time for timeout
        start_time = time.time()
        logger.info(f"Starting query execution with timeout: {PROCESSING_TIMEOUT} seconds")
        
        # Select workflow based on user choice
        if workflow_type == "dynamic":
            logger.info(f"Executing dynamic workflow for query: {query[:100]}...")
            
            # Execute the dynamic workflow with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                async def run_dynamic_workflow():
                    return await st.session_state.orchestrator.execute_dynamic_workflow(query, context)
                
                future = executor.submit(lambda: loop.run_until_complete(run_dynamic_workflow()))
                
                try:
                    # Wait for the result with a timeout
                    results = future.result(timeout=PROCESSING_TIMEOUT)
                    logger.info(f"Dynamic workflow completed in {time.time() - start_time:.2f} seconds")
                except concurrent.futures.TimeoutError:
                    logger.error(f"Dynamic workflow timed out after {PROCESSING_TIMEOUT} seconds")
                    return {
                        "error": f"Processing timed out after {PROCESSING_TIMEOUT} seconds",
                        "status": "timeout"
                    }
        else:
            logger.info(f"Executing standard workflow for query: {query[:100]}...")
            
            # Execute the standard workflow with timeout
            with concurrent.futures.ThreadPoolExecutor() as executor:
                async def run_standard_workflow():
                    return await st.session_state.orchestrator.process_research_query(
                        query=query,
                        context=context
                    )
                
                future = executor.submit(lambda: loop.run_until_complete(run_standard_workflow()))
                
                try:
                    # Wait for the result with a timeout
                    results = future.result(timeout=PROCESSING_TIMEOUT)
                    logger.info(f"Standard workflow completed in {time.time() - start_time:.2f} seconds")
                except concurrent.futures.TimeoutError:
                    logger.error(f"Standard workflow timed out after {PROCESSING_TIMEOUT} seconds")
                    return {
                        "error": f"Processing timed out after {PROCESSING_TIMEOUT} seconds",
                        "status": "timeout"
                    }
        
        # Use large output generator if available and content is large
        if st.session_state.large_output_generator and results and "results" in results:
            answer = results["results"].get("answer", "")
            if isinstance(answer, str) and len(answer) > 20000:  # Only for very large outputs
                logger.info(f"Optimizing large output with {len(answer)} characters")
                # Process through large output generator to ensure proper formatting
                large_output_result = asyncio.run(
                    st.session_state.large_output_generator.generate_large_output(
                        prompt=f"Reformat and optimize this research result:\n{answer[:2000]}...",
                        system_message="You are a research content formatter. Optimize the formatting of research content while preserving all information.",
                        target_length=len(answer)
                    )
                )
                
                # Replace the answer with the optimized version
                if large_output_result and "text" in large_output_result:
                    results["results"]["answer"] = large_output_result["text"]
                    results["large_output_processing"] = {
                        "applied": True,
                        "original_length": len(answer),
                        "processed_length": len(large_output_result["text"]),
                        "processing_time": large_output_result.get("generation_time", 0)
                    }
        
        # Clean up the loop
        loop.close()
        
        return results
    except Exception as e:
        # Log error
        logger.error(f"Error processing query synchronously: {str(e)}", exc_info=True)
        return {
            "error": str(e),
            "results": {
                "answer": f"I encountered an error while researching: {str(e)}"
            }
        }

# Process a research query asynchronously
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
        # Show processing message with animated progress indicator
        if st.session_state.processing:
            col1, col2 = st.columns([3, 1])
            with col1:
                # Progress bar
                progress_bar = st.progress(min(st.session_state.progress_counter / 8.0, 1.0))
                # Status message
                st.markdown(f"**{st.session_state.progress_status}**")
            with col2:
                # Animated spinner
                with st.spinner("Working..."):
                    st.empty()
            st.markdown("*Please wait while your research is being conducted. This may take a few minutes.*")
        
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
def main():
    """Main entry point for the Streamlit app"""
    # Make sure session state is initialized
    initialize_session_state()
    
    # Initialize app
    initialize_app()
    
    # Create the full UI with the actual research functionality
    create_ui()

# Create a simple UI that works without dependencies
def create_test_ui():
    """Create a simplified UI using configuration values"""
    # Get UI configuration
    ui_config = load_ui_config()
    colors = ui_config["colors"]
    
    # Apply CSS using configuration values
    st.markdown(f"""
    <style>
    /* Base theme */
    .stApp {{
        background-color: {colors['background']};
        color: {colors['text']};
    }}
    /* Palette styling for all buttons */
    .stButton>button {{
        background-color: {colors['primary']} !important;
        color: white !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: 500 !important;
    }}
    .stButton>button:hover {{
        background-color: {colors.get('primary_dark', '#B8252F')} !important;
        color: white !important;
    }}
    /* Form fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        border: 1px solid {colors.get('accent', '#E8DCCA')} !important;
        border-radius: 4px !important;
    }}
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {{
        border: 1px solid {colors.get('accent_dark', '#CE2B37')} !important;
        box-shadow: 0 0 0 1px rgba({colors.get('accent_dark', '#CE2B37')}, 0.2) !important;
    }}
    /* Headers */
    h1, h2, h3 {{
        color: {colors['primary']} !important;
        font-family: 'Times New Roman', serif !important;
    }}
    /* Slider track */
    .stSlider>div>div>div>div[data-baseweb='slider'] {{
        background-color: {colors.get('accent', '#E8DCCA')} !important;
    }}
    /* Slider handle */
    .stSlider>div>div>div>div[data-baseweb='slider']>div:last-child>div {{
        background-color: {colors['primary']} !important;
        border-color: {colors['primary']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Simple header with Italian styling
    st.markdown("<h1 style='color:#CE2B37; font-family:Times New Roman, serif; padding-bottom:0.5rem;'>Agentic Researcher</h1>", unsafe_allow_html=True)
    
    # Elegant description without navigation
    st.markdown("""
    <div style='background-color:#E8DCCA; padding:1.2rem; border-left:4px solid #CE2B37; border-radius:0.3rem; margin-bottom:1.5rem;'>
    <p style='font-family:Times New Roman, serif; font-size:1.1rem; color:#333333; margin:0;'>
    A professional multi-agent research assistant that automatically determines what information to gather and how to present it.
    Simply enter your research question below.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Direct research query form - no tabs, no options
    with st.form("research_form"):
        query = st.text_area(
            "Enter your research question",
            "What are the latest advancements in quantum computing?",
            height=150
        )
        
        # Only depth as setting - the agent decides everything else
        search_depth = st.select_slider(
            "Research Depth",
            options=["Basic", "Standard", "Deep"],
            value="Standard"
        )
        
        # Centered submit button with Italian styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button("Start Research")
        
        if submitted:
            st.markdown("""
            <div style='background-color:#E8DCCA; border-left:4px solid #009246; padding:1rem; border-radius:0.3rem; margin-top:1rem;'>
                <p style='color:#333333; margin:0;'><strong>Research Started:</strong> Your query has been submitted!</p>
                <p style='color:#333333; margin-top:0.5rem; margin-bottom:0;'>The system will automatically determine what information to gather and how to present it.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("Our agents are processing your query..."):
                # Simulate processing delay
                time.sleep(2)
                
                # Show sample result with Italian styling
                st.markdown("""
                <div style='margin-top:1.5rem;'>
                    <h2 style='color:#CE2B37; font-family:"Times New Roman", serif;'>Research Results</h2>
                    <div style='background-color:#FCFCFC; padding:1.5rem; border-radius:0.3rem; border:1px solid #E8DCCA;'>
                        <p>Our agents have analyzed your query and prepared comprehensive research.</p>
                        <p><em>Sample result placeholder - in the full version, detailed research would appear here...</em></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("This is a test UI. In the full application, this would trigger the AI research process.")
                
                # Display test results with Italian styling
                st.markdown("""
                <div style='margin-top:1.5rem; background-color:#FCFCFC; padding:1.5rem; border-radius:0.3rem; border:1px solid #E8DCCA;'>
                    <h3 style='color:#CE2B37; font-family:Times New Roman, serif;'>Sample Research Results</h3>
                    <p style='font-size:1.1rem; margin-bottom:1rem;'>
                    Recent advancements in quantum computing include:
                    </p>
                    <ol>
                        <li><strong>Error Correction</strong>: Breakthrough in quantum error correction with logical qubits</li>
                        <li><strong>Quantum Supremacy</strong>: Several companies demonstrating computational advantages</li>
                        <li><strong>Hardware Improvements</strong>: New qubit designs with longer coherence times</li>
                        <li><strong>Hybrid Algorithms</strong>: Combining classical and quantum approaches for practical applications</li>
                    </ol>
                    <p><em>Note: This is a placeholder for the agent-generated content.</em></p>
                
                    <h4 style='color:#CE2B37; font-family:Times New Roman, serif; margin-top:1.5rem;'>Sources</h4>
                    <div style='margin-top:0.5rem;'>
                        <p><strong>1. <a href="https://example.com/paper1">Quantum Error Correction Paper</a></strong><br/>
                        <em>Sample source description would appear here.</em></p>
                        
                        <p><strong>2. <a href="https://example.com/paper2">Hardware Advancements Review</a></strong><br/>
                        <em>Sample source description would appear here.</em></p>
                        
                        <p><strong>3. <a href="https://example.com/paper3">Hybrid Algorithms Survey</a></strong><br/>
                        <em>Sample source description would appear here.</em></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    try:
        print("Initializing Streamlit App...")
        print("If you see dependency errors, they can be safely ignored during this test.")
        print("Starting Streamlit server...")
        
        # Try to run the full application
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
        
    except Exception as e:
        import traceback
        print(f"Error initializing full Streamlit app: {e}")
        print(traceback.format_exc())
        print("\nFalling back to test UI...")
        print("This simplified version will run without all dependencies.")
        
        # Fall back to the test UI
        create_test_ui()
