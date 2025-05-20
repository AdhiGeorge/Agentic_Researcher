# Agentic Researcher

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-GPT--4o-orange)
![Framework](https://img.shields.io/badge/Framework-OpenAI_Swarm-yellow)

A state-of-the-art multi-agent system for research, reasoning, and code synthesis powered by Azure OpenAI's GPT-4o and the OpenAI Swarm framework.

## Overview

Agentic Researcher is a powerful, modular, and extensible research and code synthesis system. It combines specialized agents through a sophisticated Swarm orchestration framework using Azure OpenAI's GPT-4o and text-embedding-3-small models:

1. **Planner Agent**: Creates detailed research plans with self-revisioning chain of thought
2. **Researcher Agent**: Conducts web research with multi-source verification
3. **Formatter Agent**: Processes content with entity extraction and knowledge graph integration
4. **Answer Agent**: Synthesizes comprehensive answers with citation integration
5. **Coder Agent**: Generates robust, reusable code libraries
6. **Runner Agent**: Executes code in isolated environments with error handling
7. **Feature Agent**: Implements new capabilities with architectural integration
8. **Patcher Agent**: Fixes bugs through test-driven fixing approaches
9. **Reporter Agent**: Creates structured documentation and visual representations
10. **Decision Agent**: Routes ambiguous prompts to the appropriate specialized agent

## Key Features

- **Self-Revisioning Chain of Thought**: Agents evaluate and improve their outputs through dynamic feedback loops
- **Entity-Based Contextual Graphing**: Named entity recognition and knowledge graph construction for contextual understanding
- **Context-Aware Prompt Caching**: Hybrid hash-based and semantic similarity caching (0.85-0.95 threshold)
- **Advanced Playwright Web Scraping**: Anti-detection mechanisms, proxy rotation, and robots.txt compliance
- **Semantic Chunking**: Content processing with entity preservation and optimal overlap
- **Azure OpenAI Integration**: Leverages GPT-4o for reasoning and text-embedding-3-small for embeddings
- **SwarmOS Architecture**: Full implementation of OpenAI's Swarm framework for agent orchestration
- **Dual Database Strategy**: Qdrant for vector storage and SQLite for state management
- **Streamlit UI**: Developer-friendly interface with real-time agent activity visualization

## Getting Started

### Prerequisites

- Python 3.8+
- Azure OpenAI API key and endpoint
- Qdrant for vector storage (optional, built-in SQLite-based fallback)
- Streamlit for the web interface (optional)

### Project Structure

The project follows a clean, modular architecture designed for extensibility and maintainability:

```
src/                        # Main source code directory
├── agents/                 # Specialized agent implementations
│   ├── __init__.py
│   ├── base_agent.py       # Base agent class
│   ├── planner/            # Planning with self-revision
│   ├── researcher/         # Web research and scraping
│   ├── formatter/          # Content processing and structuring
│   ├── answer/             # Answer synthesis
│   ├── coder/              # Code generation
│   ├── runner/             # Code execution
│   ├── feature/            # Feature implementation
│   ├── patcher/            # Bug fixing
│   ├── reporter/           # Documentation generation
│   ├── decision/           # Query routing
│   └── internal_monologue/ # Thought verbalization
├── orchestrator/           # Orchestration framework
│   ├── __init__.py
│   ├── swarm_orchestrator.py       # Swarm implementation
│   └── swarm_manager.py    # Workflow management
├── db/                     # Database systems
│   ├── __init__.py
│   ├── sqlite_manager.py   # State and raw data storage
│   └── qdrant_manager.py   # Vector embeddings
├── search/                 # Search and scraping
│   ├── __init__.py
│   └── unified_scraper.py  # Comprehensive web scraper
├── ui/                     # User interfaces
│   ├── __init__.py
│   └── streamlit_app.py    # Streamlit UI
└── utils/                  # Utility modules
    ├── __init__.py
    ├── azure_client.py     # Azure OpenAI client
    ├── config.py           # Configuration management
    ├── graph_builder.py    # Knowledge graph construction
    ├── semantic_cache.py   # Context-aware caching
    ├── semantic_chunker.py # Advanced text chunking
    └── query_hash.py       # Query hashing and management

app.py                      # Streamlit entry point
main.py                     # CLI entry point
requirements.txt            # Dependencies
.env                        # Environment variables
```

## Installation

1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Playwright for advanced scraping:
   ```bash
   playwright install
   ```
4. Install Qdrant for vector storage:
   ```bash
   pip install qdrant-client
   # Optional: run Qdrant locally
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```
5. Set up your environment variables in a `.env` file:
   ```
   # Azure OpenAI configuration
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint_here
   AZURE_API_VERSION_CHAT=2023-07-01-preview
   AZURE_API_VERSION_EMBEDDING=2023-05-15
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
   
   # Qdrant configuration
   QDRANT_URL=localhost
   QDRANT_PORT=6333
   
   # Search configuration (optional)
   TAVILY_API_KEY=your_tavily_key
   SERPER_API_KEY=your_serper_key
   GOOGLE_CSE_ID=your_google_cse_id
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

Agentic Researcher can be used through both a command line interface and a Streamlit web UI:

### Command Line Interface

```bash
# Run a research query
python main.py research "What is volatility index and what is the mathematical formula to calculate the VIX score?"

# Generate code for a specific task
python main.py code "Write a Python code to calculate the VIX score."

# List all research projects
python main.py list

# View project details
python main.py details 1

# Ask a follow-up question
python main.py followup "Can you explain the implications of a high VIX score?" --project-id 1 --action answer

# Get system statistics
python main.py stats
```

### Example Research Query

Try this example to see Agentic Researcher in action:

```bash
python main.py research "What is the Volatility Index (VIX) and how is it calculated? Also provide a Python implementation of the VIX calculation."
```

The system will:
1. Create a detailed research plan with self-revisioning
2. Extract keywords for precise searching
3. Conduct web research from multiple sources
4. Process and structure the information
5. Generate a comprehensive answer
6. Create executable Python code for VIX calculation

### Advanced Knowledge Processing

Agentic Researcher implements a sophisticated knowledge processing pipeline:

1. **Web Scraping**: Using Playwright with advanced anti-detection
   ```python
   from src.search.unified_scraper import UnifiedScraper
   
   # Create scraper with proxy rotation
   scraper = UnifiedScraper(use_proxies=True, use_stealth_mode=True)
   
   # Scrape with stealth browser
   results = scraper.scrape_multiple(["https://www.cboe.com/tradable_products/vix/"])
   ```

2. **Knowledge Graph Construction**: Entity extraction and relationship modeling
   ```python
   from src.utils.graph_builder import KnowledgeGraphBuilder
   
   # Create graph builder
   graph_builder = KnowledgeGraphBuilder()
   
   # Process text and extract entities
   entities = graph_builder.process_text(content, source_id="vix_article")
   
   # Get entities relevant to a query
   relevant = graph_builder.get_relevant_entities("VIX calculation")
   ```

3. **Semantic Chunking**: Content processing with entity preservation
   ```python
   from src.utils.semantic_chunker import SemanticChunker
   
   # Create chunker
   chunker = SemanticChunker(chunk_size=1000, chunk_overlap=200)
   
   # Create chunks with entity awareness
   chunks = chunker.chunk_with_entities(text, entities, metadata={"source": "cboe"})
   ```

4. **Vector Storage**: Optimized embedding and retrieval
   ```python
   from src.db.qdrant_manager import QdrantManager
   
   # Store processed knowledge
   qdrant = QdrantManager()
   point_ids = qdrant.store_document(document, project_id=1)
   
   # Search for relevant content
   results = qdrant.search("VIX calculation formula", limit=5)
   ```

### Streamlit Web Interface

Launch the Streamlit UI for an interactive experience:

```bash
streamlit run app.py
```

The Streamlit UI provides a user-friendly interface with:

1. **Research Dashboard**: Submit queries and view results in real-time
2. **Agent Monitoring**: Observe agent activities and reasoning processes
3. **Project Management**: Create, view, and manage research projects
4. **Follow-up Handling**: Ask follow-up questions with context preserved
5. **Code Execution**: Run generated code directly from the interface
6. **Knowledge Exploration**: Navigate the knowledge graph visually

### Self-Revisioning Chain of Thought

Agentic Researcher implements a sophisticated self-revisioning mechanism:

```python
# From src/orchestrator/swarm_orchestrator.py
async def execute_with_self_revision(self, agent_name, prompt, context=None, max_revisions=2):
    """Execute an agent with self-revision capability."""
    # Initial execution with the agent
    response = await self.swarm.arun(agent=agent, thread=thread)
    initial_response = response.messages[-1]["content"]
    
    # Initialize revisions
    revisions = [{"revision": 0, "response": initial_response}]
    
    # Perform self-revisions
    for revision in range(1, max_revisions + 1):
        revision_prompt = {
            "role": "user",
            "content": "Review your previous response and improve it if needed..."
        }
        
        # Get revision
        revision_response = await self.swarm.arun(agent=agent, thread=thread)
        revised_response = revision_response.messages[-1]["content"]
        
        # Store revision if changes were made
        if "No revision needed" not in revised_response:
            revisions.append({"revision": revision, "response": revised_response})
```

## Architecture Deep Dive

### Agent Hierarchy

Agentic Researcher implements a hierarchical agent structure with specialized responsibilities:

```
SwarmOrchestrator (Orchestrator)
├── PlannerAgent
│   └── Self-Revisioning Planning
├── KeywordAgent
│   └── Context-Specific Extraction
├── ResearcherAgent
│   └── Multi-Source Verification
├── FormatterAgent
│   └── Entity Extraction & Knowledge Graphs
├── AnswerAgent
│   └── Multi-Perspective Synthesis
├── CodeAgent
│   └── Library-Oriented Generation
├── RunnerAgent
│   └── Isolated Execution Environment
├── FeatureAgent
│   └── Modular Component Design
├── PatcherAgent
│   └── Test-Driven Fixing
├── ReporterAgent
│   └── Template-Based Documentation
└── DecisionAgent
    └── Confidence-Weighted Routing
```

### Storage Architecture

The dual-database design separates concerns for optimal performance:

#### SQLite Manager
- Agent states and execution metrics
- Raw scraped content storage
- Query cache with hash and semantic retrieval
- Conversation history tracking
- Project and metadata management

#### Qdrant Manager
- Vector embeddings for semantic search
- Clean, processed content chunks
- Entity relationship metadata
- Knowledge graph connections
- Contextual retrieval with threshold tuning

## Strategic Innovations

Agentic Researcher introduces several strategic innovations in AI agent design:

### 1. Self-Revisioning Chain of Thought
- Dynamic plan evaluation and revision
- Quality assessment feedback loops
- Execution metric-driven refinement
- Confidence scoring and improvement tracking

### 2. Entity-Based Contextual Graphing
- Named entity recognition from research content
- Semantic network construction between concepts
- Contextual graph pruning for query relevance
- Multi-perspective synthesis across sources

### 3. Context-Aware Prompt Caching
- Hybrid exact/semantic matching strategy
- Adaptive threshold control (0.85-0.95)
- Temporal relevance for cache invalidation
- Execution-aware response enhancement

### 4. Advanced Playwright Web Scraping
- Headless browser detection evasion
- User agent and proxy rotation
- Session management and cookie persistence
- Rate limiting awareness and robots.txt compliance

## Future Directions

Future enhancements planned for Agentic Researcher include:

1. **Graph-of-Thought Reasoning**: Moving beyond chains to reasoning graphs
2. **Conversational Memory**: Enhanced long-term context retention
3. **Domain Adaptation**: Specialized knowledge for finance, science, engineering
4. **Multi-Modal Research**: Integrating image and video understanding
5. **Citation Validation**: Factual accuracy verification of sources

## Extending the System

The modular architecture makes it easy to extend with new capabilities:

1. **Add a new search engine**: Create a new class in `src/search/` that implements the base search interface
2. **Add a new agent**: Create a new agent class in `src/agents/` and register it with the agent orchestrator
3. **Enhance the UI**: Modify the Streamlit app in `src/ui/app.py` to add new features
4. **Add a new data source**: Extend the vector database in `src/db/qdrant_manager.py`
5. **Improve keyword extraction**: Add new methods to `src/utils/keyword_extractor.py`

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


# netstat -ano | findstr :8502
# taskkill /PID 26560 /F 