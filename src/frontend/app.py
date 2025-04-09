
import streamlit as st
from src.agents.orchestrator import TaskOrchestrator
from src.config.system_config import SystemConfig
from src.frontend.components.sidebar import render_sidebar
from src.frontend.components.header import render_header
from src.frontend.components.query_input import render_query_input
from src.frontend.components.results_display import render_results
from src.frontend.components.agent_status import render_agent_status
from src.frontend.components.swarm_visualizer import render_swarm_visualizer
from src.frontend.state import initialize_session_state

def run_app():
    # Set page config
    st.set_page_config(
        page_title="Agentic AI Research Assistant",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load system configuration
    config = SystemConfig()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar(config)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input section
        query = render_query_input()
        
        if st.session_state.get('submit_clicked', False):
            # Reset state
            st.session_state.submit_clicked = False
            
            # Create orchestrator
            orchestrator = TaskOrchestrator(config)
            
            # Start research process
            with st.spinner("🧠 Researching your query..."):
                result = orchestrator.process_query(query)
                st.session_state.current_result = result
    
    with col2:
        # Agent status display
        render_agent_status()
        
        # Swarm visualizer (if in debug mode)
        if st.session_state.get('debug_mode', False) and st.session_state.get('current_result'):
            if 'swarm_visualization' in st.session_state.current_result:
                render_swarm_visualizer(st.session_state.current_result['swarm_visualization'])
    
    # Display results
    if st.session_state.get('current_result'):
        render_results(st.session_state.current_result)
