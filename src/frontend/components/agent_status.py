
import streamlit as st
import time
import random

def render_agent_status():
    """Render the agent status panel"""
    st.subheader("Agent Status")
    
    # Display the status of each agent
    agents = {
        "Orchestrator": st.session_state.agent_status.get("orchestrator", "idle"),
        "Web Search": st.session_state.agent_status.get("search", "idle"),
        "Scraper": st.session_state.agent_status.get("scraper", "idle"),
        "RAG Engine": st.session_state.agent_status.get("rag", "idle"),
        "Code Generator": st.session_state.agent_status.get("code_generator", "idle"),
        "Validator": st.session_state.agent_status.get("validator", "idle"),
        "Report Generator": st.session_state.agent_status.get("report_generator", "idle")
    }
    
    # Status colors and emoji indicators
    status_colors = {
        "idle": "gray",
        "running": "blue",
        "completed": "green",
        "error": "red"
    }
    
    status_emoji = {
        "idle": "⏸️",
        "running": "⚙️",
        "completed": "✅",
        "error": "❌"
    }
    
    # Render each agent's status
    for agent, status in agents.items():
        emoji = status_emoji.get(status, "⏸️")
        color = status_colors.get(status, "gray")
        
        # Create a status indicator
        st.markdown(
            f"""
            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                padding: 8px;
                border-radius: 4px;
                background-color: {color}15;
            ">
                <div style="margin-right: 10px;">{emoji}</div>
                <div style="flex-grow: 1;">
                    <span style="font-weight: bold;">{agent}</span>
                </div>
                <div>
                    <span style="
                        color: {color};
                        text-transform: capitalize;
                        font-size: 0.8em;
                        font-weight: bold;
                    ">{status}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Add debug information if in debug mode
    if st.session_state.get('debug_mode', False):
        st.divider()
        st.subheader("Debug Information")
        
        # Memory usage
        st.markdown("**Memory Usage**")
        # This would be real data in a production app
        st.progress(0.7, "Memory: 70%")
        
        # Agent graph visualization placeholder
        st.markdown("**Agent Graph**")
        st.markdown("Agent graph visualization would appear here")
        
        # Log output
        st.markdown("**Log Output**")
        st.code("""
INFO: Orchestrator started
INFO: Web Search Agent processing query
INFO: Found 15 results from DuckDuckGo
INFO: Scraper processing top 5 results
INFO: RAG Engine embedding extracted content
...
        """)
