"""
Streamlit Views for Agentic Researcher
Provides UI components for the web application
"""
import streamlit as st
import pandas as pd
import json
import time
from typing import Dict, List, Any, Optional

from ..db.sqlite_logger import SQLiteLogger
from ..db.qdrant_manager import QdrantManager

class StreamlitViews:
    """
    Streamlit UI views for Agentic Researcher
    Manages different pages and UI components
    """
    
    def __init__(self):
        # Initialize SQLite logger
        self.logger = SQLiteLogger()
        
        # Initialize Vector DB
        self.vector_db = QdrantManager(collection_name="research_data")
        
        # Set page configuration
        st.set_page_config(
            page_title="Agentic Researcher",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def sidebar(self) -> str:
        """
        Render sidebar with navigation
        
        Returns:
            str: Selected page
        """
        st.sidebar.title("🧠 Agentic Researcher")
        st.sidebar.markdown("---")
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            options=["Home", "Projects", "Vector Inspector", "Code Viewer", "Logs"],
            index=0
        )
        
        st.sidebar.markdown("---")
        
        # App info
        st.sidebar.markdown("### About")
        st.sidebar.info(
            "Agentic Researcher combines AI agents for research, "
            "web scraping, and code generation. Built with Azure OpenAI, "
            "Qdrant, and Playwright."
        )
        
        return page
    
    def home_page(self) -> None:
        """Render home page with prompt interface"""
        st.title("🧠 Agentic Researcher")
        st.markdown(
            "Enter your research query below. The system will use multiple AI agents "
            "to research, analyze, and generate insights on your topic."
        )
        
        # New project form
        with st.form("new_project_form"):
            project_name = st.text_input("Project Name", placeholder="My Research Project")
            
            col1, col2 = st.columns(2)
            with col1:
                query = st.text_area(
                    "Research Query",
                    placeholder="Enter your research question or topic here...",
                    height=150
                )
            
            with col2:
                st.markdown("#### Research Options")
                search_depth = st.select_slider(
                    "Search Depth",
                    options=["Basic", "Standard", "Deep"],
                    value="Standard"
                )
                
                output_format = st.radio(
                    "Output Format",
                    options=["Text Summary", "Code Generation", "Both"],
                    index=0
                )
                
                include_pdf = st.checkbox("Include PDF sources", value=True)
            
            submit_button = st.form_submit_button("Start Research")
            
            if submit_button:
                if not project_name or not query:
                    st.error("Please provide both a project name and research query.")
                else:
                    # Create new project
                    project_id = self.logger.create_project(
                        name=project_name,
                        description=query,
                        metadata={
                            "search_depth": search_depth,
                            "output_format": output_format,
                            "include_pdf": include_pdf,
                            "status": "created"
                        }
                    )
                    
                    # Show success message with project ID
                    st.success(f"Project created successfully! Project ID: {project_id}")
                    
                    # Display processing message
                    st.info("Your research request has been submitted. View progress in the Projects tab.")
                    
                    # Store query
                    self.logger.add_conversation_message(
                        project_id=project_id,
                        role="user",
                        content=query
                    )
    
    def projects_page(self) -> None:
        """Render projects page with list of projects and details"""
        st.title("📋 Projects")
        
        # Get all projects
        projects = self.logger.list_projects()
        
        if not projects:
            st.info("No projects found. Create a new project on the Home page.")
            return
        
        # Display projects in a table
        project_data = []
        for project in projects:
            project_data.append({
                "ID": project.get("id"),
                "Name": project.get("name"),
                "Created": project.get("created_at"),
                "Status": project.get("metadata", {}).get("status", "unknown")
            })
        
        df = pd.DataFrame(project_data)
        st.dataframe(df, use_container_width=True)
        
        # Project details section
        st.markdown("---")
        st.subheader("Project Details")
        
        selected_project_id = st.selectbox(
            "Select Project",
            options=[p.get("id") for p in projects],
            format_func=lambda x: f"{x}: {next((p.get('name') for p in projects if p.get('id') == x), '')}"
        )
        
        if selected_project_id:
            self._show_project_details(selected_project_id)
    
    def _show_project_details(self, project_id: int) -> None:
        """
        Show details for a specific project
        
        Args:
            project_id: Project ID to display
        """
        # Get project data
        project = self.logger.get_project(project_id)
        if not project:
            st.error(f"Project with ID {project_id} not found.")
            return
        
        # Get conversation history
        conversation = self.logger.get_conversation_history(project_id)
        
        # Display project info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### {project.get('name')}")
            st.markdown(f"**Created:** {project.get('created_at')}")
            st.markdown(f"**Status:** {project.get('metadata', {}).get('status', 'unknown')}")
        
        with col2:
            st.markdown("### Actions")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                if st.button("View Results", key=f"view_results_{project_id}"):
                    st.session_state.view_results_project_id = project_id
            
            with col2_2:
                if st.button("Export Results", key=f"export_results_{project_id}"):
                    st.session_state.export_results_project_id = project_id
        
        # Display description/query
        st.markdown("### Query")
        st.info(project.get("description", "No query provided"))
        
        # Display conversation
        if conversation:
            st.markdown("### Conversation History")
            
            for msg in conversation:
                role = msg.get("role", "").lower()
                
                if role == "user":
                    st.markdown(f"**User:** {msg.get('content', '')}")
                elif role == "assistant":
                    st.markdown(f"**Assistant:** {msg.get('content', '')}")
                elif role == "system":
                    st.markdown(f"**System:** {msg.get('content', '')}")
        
        # View results section
        if hasattr(st.session_state, "view_results_project_id") and st.session_state.view_results_project_id == project_id:
            self._show_project_results(project_id)
    
    def _show_project_results(self, project_id: int) -> None:
        """
        Show results for a specific project
        
        Args:
            project_id: Project ID to display results for
        """
        st.markdown("---")
        st.subheader("Research Results")
        
        # Get the latest answer agent state
        answer_state = self.logger.get_latest_agent_state(project_id, "answer")
        
        if not answer_state:
            st.info("No research results available yet.")
            return
        
        state_data = answer_state.get("state_data", {})
        
        if state_data.get("status") == "completed":
            # Display answer if available
            if "answer" in state_data:
                st.markdown(state_data["answer"])
            else:
                st.info("Research completed but no answer generated.")
            
            # Display sources if available
            if "sources" in state_data and state_data["sources"]:
                st.markdown("### Sources")
                for i, source in enumerate(state_data["sources"]):
                    st.markdown(f"{i+1}. [{source}]({source})")
        
        # Get the latest coder agent state
        coder_state = self.logger.get_latest_agent_state(project_id, "coder")
        
        if coder_state and coder_state.get("state_data", {}).get("status") == "completed":
            st.markdown("---")
            st.subheader("Generated Code")
            
            code_result = coder_state.get("state_data", {}).get("value", {})
            
            if code_result:
                # Display explanation
                st.markdown(f"### Explanation")
                st.markdown(code_result.get("explanation", "No explanation provided"))
                
                # Display usage instructions
                st.markdown(f"### Usage Instructions")
                st.markdown(code_result.get("usage_instructions", "No usage instructions provided"))
                
                # Display code files
                st.markdown(f"### Code Files")
                
                files = code_result.get("files", [])
                if files:
                    for file in files:
                        file_name = file.get("file_name", "unnamed.txt")
                        content = file.get("content", "")
                        
                        st.markdown(f"**{file_name}**")
                        st.code(content, language=self._get_language_from_filename(file_name))
                else:
                    st.info("No code files generated.")
        
        # Agent execution history
        st.markdown("---")
        st.subheader("Agent Execution History")
        
        # Get all agent states for this project
        planner_state = self.logger.get_latest_agent_state(project_id, "planner")
        researcher_state = self.logger.get_latest_agent_state(project_id, "researcher")
        chunker_state = self.logger.get_latest_agent_state(project_id, "chunker")
        
        agent_states = [
            ("Planner", planner_state),
            ("Researcher", researcher_state),
            ("Chunker", chunker_state),
            ("Answer", answer_state),
            ("Coder", coder_state)
        ]
        
        for agent_name, state in agent_states:
            if state:
                status = state.get("state_data", {}).get("status", "unknown")
                created_at = state.get("created_at", "")
                
                st.markdown(f"**{agent_name}:** {status} ({created_at})")
    
    def _get_language_from_filename(self, filename: str) -> str:
        """
        Get language for syntax highlighting based on file extension
        
        Args:
            filename: Filename to parse
            
        Returns:
            str: Language for syntax highlighting
        """
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "csharp",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".md": "markdown",
            ".sql": "sql",
            ".sh": "bash",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".php": "php"
        }
        
        for ext, lang in ext_map.items():
            if filename.lower().endswith(ext):
                return lang
        
        return "text"
    
    def vector_inspector_page(self) -> None:
        """Render vector inspector page for viewing embedded chunks"""
        st.title("🔍 Vector Inspector")
        
        # Get all projects for filtering
        projects = self.logger.list_projects()
        
        if not projects:
            st.info("No projects found. Create a new project on the Home page.")
            return
        
        # Project selector
        selected_project_id = st.selectbox(
            "Select Project",
            options=[0] + [p.get("id") for p in projects],
            format_func=lambda x: f"{x}: {next((p.get('name') for p in projects if p.get('id') == x), '')}" if x > 0 else "All Projects"
        )
        
        # Search interface
        st.markdown("### Search Vector Database")
        
        query = st.text_input("Search Query", placeholder="Enter search terms...")
        limit = st.slider("Results Limit", min_value=5, max_value=50, value=10)
        
        search_button = st.button("Search")
        
        if search_button and query:
            # Prepare filter condition
            filter_condition = {}
            if selected_project_id > 0:
                filter_condition["project_id"] = str(selected_project_id)
            
            # Search vector database
            with st.spinner("Searching vectors..."):
                project_id = selected_project_id if selected_project_id > 0 else None
                results = self.vector_db.search_similar(query, project_id=project_id, limit=limit)
            
            # Display results
            if results:
                st.markdown(f"### Search Results ({len(results)} chunks)")
                
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} - Score: {result.get('score', 0):.4f}"):
                        # Display metadata
                        metadata = result.get("metadata", {})
                        source = metadata.get("url", "Unknown source")
                        
                        st.markdown(f"**Source:** {source}")
                        
                        # Display content
                        st.markdown("**Content:**")
                        st.text(result.get("content", "No content"))
            else:
                st.info("No results found.")
    
    def code_viewer_page(self) -> None:
        """Render code viewer page for viewing generated code"""
        st.title("💻 Code Viewer")
        
        # Get projects with code
        projects = self.logger.list_projects()
        
        if not projects:
            st.info("No projects found. Create a new project on the Home page.")
            return
        
        # Find projects with code results
        projects_with_code = []
        for project in projects:
            project_id = project.get("id")
            coder_state = self.logger.get_latest_agent_state(project_id, "coder")
            
            if coder_state and coder_state.get("state_data", {}).get("status") == "completed":
                projects_with_code.append(project)
        
        if not projects_with_code:
            st.info("No projects with generated code found.")
            return
        
        # Project selector
        selected_project_id = st.selectbox(
            "Select Project",
            options=[p.get("id") for p in projects_with_code],
            format_func=lambda x: f"{x}: {next((p.get('name') for p in projects_with_code if p.get('id') == x), '')}"
        )
        
        if selected_project_id:
            # Get coder agent state
            coder_state = self.logger.get_latest_agent_state(selected_project_id, "coder")
            
            if not coder_state:
                st.error("No code found for this project.")
                return
            
            code_result = coder_state.get("state_data", {}).get("value", {})
            
            if not code_result:
                st.error("No code result found.")
                return
            
            # Display explanation
            st.markdown(f"### Explanation")
            st.markdown(code_result.get("explanation", "No explanation provided"))
            
            # Display usage instructions
            st.markdown(f"### Usage Instructions")
            st.markdown(code_result.get("usage_instructions", "No usage instructions provided"))
            
            # Display code files
            files = code_result.get("files", [])
            
            if not files:
                st.info("No code files found.")
                return
            
            # File selector
            file_names = [f.get("file_name", f"File {i+1}") for i, f in enumerate(files)]
            selected_file = st.selectbox("Select File", options=file_names)
            
            # Display selected file
            for file in files:
                if file.get("file_name") == selected_file:
                    content = file.get("content", "")
                    
                    st.markdown(f"### {selected_file}")
                    st.code(content, language=self._get_language_from_filename(selected_file))
                    
                    # Download button
                    file_content = content.encode()
                    st.download_button(
                        label="Download File",
                        data=file_content,
                        file_name=selected_file,
                        mime="text/plain"
                    )
                    break
    
    def logs_page(self) -> None:
        """Render logs page for viewing agent execution logs"""
        st.title("📊 Logs")
        
        # Get all projects
        projects = self.logger.list_projects()
        
        if not projects:
            st.info("No projects found. Create a new project on the Home page.")
            return
        
        # Project selector
        selected_project_id = st.selectbox(
            "Select Project",
            options=[p.get("id") for p in projects],
            format_func=lambda x: f"{x}: {next((p.get('name') for p in projects if p.get('id') == x), '')}"
        )
        
        if selected_project_id:
            # Agent selector
            agent_types = ["orchestrator", "planner", "researcher", "chunker", "coder", "answer", "action"]
            selected_agent = st.selectbox("Select Agent", options=agent_types)
            
            if selected_agent:
                # Get agent states for this project and agent type
                agent_state = self.logger.get_latest_agent_state(selected_project_id, selected_agent)
                
                if agent_state:
                    # Display state data
                    st.json(agent_state.get("state_data", {}))
                else:
                    st.info(f"No logs found for {selected_agent} agent in this project.")
    
    def run(self) -> None:
        """Run the Streamlit application"""
        # Render sidebar and get selected page
        page = self.sidebar()
        
        # Render selected page
        if page == "Home":
            self.home_page()
        elif page == "Projects":
            self.projects_page()
        elif page == "Vector Inspector":
            self.vector_inspector_page()
        elif page == "Code Viewer":
            self.code_viewer_page()
        elif page == "Logs":
            self.logs_page()
