
# Agentic AI Research Assistant

## Overview
The Agentic AI Research Assistant is a fully autonomous, modular, multi-agent research system designed for deep knowledge discovery, secure code generation, validation, and structured reporting. The system is optimized for performance, extensibility, and transparency, making it suitable for researchers, developers, analysts, and AI experimenters.

## Features
- Swarm-style multi-agent coordination
- Retrieval-augmented generation (RAG) with persistent memory
- Dynamic tool generation and registration
- Secure, sandboxed code execution
- Domain-agnostic validation
- Structured report generation with visualizations
- Modular and scalable framework

## Setup and Installation

### Prerequisites
- Python 3.10+ recommended
- Pip package manager

### Installation
1. Clone this repository:
```
git clone https://github.com/yourusername/agentic-ai-research.git
cd agentic-ai-research
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_api_key_here
```

## Running the Application
Start the Streamlit web interface:
```
python src/main.py
```

This will launch the web interface at `http://localhost:8501`.

## Usage
1. Enter your research query in the main input field
2. Select relevant research options (depth, academic sources, etc.)
3. Click "Start Research" to initiate the research process
4. View results in the various tabs (Summary, Detailed Findings, Generated Code, etc.)

## Project Structure
```
agentic-ai-research/
├── src/
│   ├── agents/            # Agent implementations
│   ├── config/            # System configuration
│   ├── frontend/          # Streamlit UI components
│   ├── llm/               # LLM management
│   ├── memory/            # Persistent memory handling
│   └── main.py            # Entry point
├── memory/                # Stored embeddings and history
├── outputs/               # Generated reports, code, etc.
│   ├── code/              # Generated code outputs
│   └── reports/           # Research reports
└── tools/                 # Dynamic tool registry
```

## Configuration
System behavior can be configured through the web interface or by editing the configuration files in `src/config/`.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
