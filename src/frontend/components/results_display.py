
import streamlit as st
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

def render_results(result):
    """Render the research results"""
    if not result:
        return
    
    st.header("Research Results")
    
    # Create tabs for different views of the results
    tabs = st.tabs(["Summary", "Detailed Findings", "Generated Code", "Visualizations", "Sources"])
    
    with tabs[0]:  # Summary
        st.subheader("Research Summary")
        if "summary" in result:
            st.markdown(result["summary"])
        else:
            st.markdown("No summary available.")
        
        # Key insights
        if "key_insights" in result and result["key_insights"]:
            st.subheader("Key Insights")
            for i, insight in enumerate(result["key_insights"], 1):
                st.markdown(f"**{i}.** {insight}")
    
    with tabs[1]:  # Detailed Findings
        st.subheader("Detailed Findings")
        if "detailed_findings" in result and result["detailed_findings"]:
            for section in result["detailed_findings"]:
                st.markdown(f"### {section['title']}")
                st.markdown(section["content"])
        else:
            st.markdown("No detailed findings available.")
    
    with tabs[2]:  # Generated Code
        st.subheader("Generated Code")
        if "generated_code" in result and result["generated_code"]:
            for code_block in result["generated_code"]:
                st.markdown(f"### {code_block['title']}")
                st.markdown(code_block["description"])
                st.code(code_block["code"], language="python")
                
                if "execution_result" in code_block:
                    with st.expander("Execution Result"):
                        st.code(code_block["execution_result"])
        else:
            st.markdown("No code generated for this query.")
    
    with tabs[3]:  # Visualizations
        st.subheader("Visualizations")
        if "visualizations" in result and result["visualizations"]:
            for viz in result["visualizations"]:
                st.markdown(f"### {viz['title']}")
                st.markdown(viz["description"])
                
                # Check if it's a plotly figure or matplotlib
                if "plotly_fig" in viz:
                    try:
                        # This would be a JSON representation in a real scenario
                        fig_data = json.loads(viz["plotly_fig"]) 
                        fig = px.line(x=[1, 2, 3], y=[1, 4, 9])  # Placeholder
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error rendering plotly visualization: {e}")
                
                elif "matplotlib_fig" in viz:
                    try:
                        # This would be a base64 or saved figure in a real scenario
                        fig, ax = plt.subplots()
                        ax.plot([1, 2, 3], [1, 4, 9])
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error rendering matplotlib visualization: {e}")
        else:
            st.markdown("No visualizations available for this query.")
    
    with tabs[4]:  # Sources
        st.subheader("Sources")
        if "sources" in result and result["sources"]:
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"{i}. {source['title']}"):
                    st.markdown(f"**URL:** [{source['url']}]({source['url']})")
                    st.markdown(f"**Accessed:** {source.get('accessed_date', 'N/A')}")
                    st.markdown(f"**Relevance:** {source.get('relevance_score', 'N/A')}/10")
                    st.markdown(f"**Summary:** {source.get('summary', 'No summary available')}")
        else:
            st.markdown("No sources available.")
    
    # Save button
    st.divider()
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("Save Results", use_container_width=True):
            # In a real app, this would save to a file or database
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.search_history.append({
                "timestamp": timestamp,
                "query": result.get("query", "Unknown query"),
                "result": result
            })
            st.success("Results saved to history!")
