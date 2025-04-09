
import streamlit as st
from src.config.system_config import SystemConfig

def render_sidebar(config: SystemConfig):
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # LLM Settings
        st.subheader("LLM Configuration")
        
        llm_option = st.selectbox(
            "LLM Provider",
            options=["Azure OpenAI", "OpenAI", "Mistral", "Local Model"],
            index=0
        )
        
        model_name = st.selectbox(
            "Model",
            options=["gpt-4o", "gpt-3.5-turbo", "mistral-large", "mistral-medium"],
            index=0
        )
        
        # Advanced settings collapsible
        with st.expander("Advanced Settings"):
            # Vector store
            vector_store = st.selectbox(
                "Vector Store",
                options=["Qdrant", "Chroma", "FAISS"],
                index=0
            )
            
            # Web scraping settings
            st.subheader("Web Scraping")
            use_proxy = st.checkbox("Use Proxy Rotation", value=True)
            respect_robots = st.checkbox("Respect robots.txt", value=True)
            
            # Execution settings
            st.subheader("Code Execution")
            sandbox_mode = st.selectbox(
                "Sandbox Mode",
                options=["Firejail", "Docker", "None"],
                index=0
            )
        
        # Debug mode
        st.divider()
        debug_mode = st.checkbox("Debug Mode", value=st.session_state.get('debug_mode', False))
        if debug_mode != st.session_state.get('debug_mode', False):
            st.session_state.debug_mode = debug_mode
            st.rerun()
        
        # About
        st.divider()
        st.markdown("### About")
        st.markdown("""
        **Agentic AI Research Assistant**  
        Version 0.1.0
        
        A modular, multi-agent system for automated research.
        """)
