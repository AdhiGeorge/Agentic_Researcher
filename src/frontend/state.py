
import streamlit as st

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = {
            "orchestrator": "idle",
            "search": "idle",
            "scraper": "idle",
            "rag": "idle",
            "code_generator": "idle",
            "validator": "idle",
            "report_generator": "idle"
        }
    
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
        
    if 'swarm_visualization' not in st.session_state:
        st.session_state.swarm_visualization = None
        
    if 'show_agent_communication' not in st.session_state:
        st.session_state.show_agent_communication = False
