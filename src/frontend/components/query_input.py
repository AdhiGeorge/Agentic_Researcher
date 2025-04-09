
import streamlit as st

def render_query_input():
    """Render the query input section"""
    st.subheader("Research Query")
    
    # Research area selection
    research_area = st.selectbox(
        "Research Area",
        options=["General", "Computer Science", "Biology", "Finance", "Physics", "Chemistry"],
        index=0
    )
    
    # Query input
    query = st.text_area(
        "Enter your research query",
        placeholder="e.g., 'Explore the relationship between quantum computing and neural networks'",
        height=100
    )
    
    # Search depth and options
    col1, col2 = st.columns(2)
    
    with col1:
        search_depth = st.slider(
            "Search Depth",
            min_value=1,
            max_value=5,
            value=3,
            help="Higher values mean more thorough but slower research"
        )
    
    with col2:
        include_academic = st.checkbox("Include Academic Papers", value=True)
        include_code = st.checkbox("Generate Code", value=True)
    
    # Submit button
    submit = st.button("Start Research", type="primary", use_container_width=True)
    
    if submit:
        if not query:
            st.error("Please enter a research query")
            return None
        st.session_state.submit_clicked = True
        return {
            "query": query,
            "research_area": research_area,
            "search_depth": search_depth,
            "include_academic": include_academic,
            "include_code": include_code
        }
    
    return None
