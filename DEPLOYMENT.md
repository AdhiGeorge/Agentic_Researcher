# Deployment Guide for Agentic Researcher

This document provides detailed instructions for deploying and running the Agentic Researcher system, an AI-powered research platform using Azure OpenAI and Swarm orchestration.

## Prerequisites

- Python 3.8+ installed
- Azure OpenAI API access (with GPT-4o and text-embedding-3-small models)
- Qdrant vector database (optional, built-in SQLite fallback available)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Agentic-Researcher.git
cd Agentic-Researcher
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Unix/MacOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and edit it with your credentials:

```bash
cp .env.example .env
```

Open the `.env` file and set your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_API_VERSION_CHAT=2023-07-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 5. Initialize Directories

Create necessary directories for data storage:

```bash
mkdir -p logs scraped_data processed_data
```

## Running the Application

### CLI Mode

For a simple research query:

```bash
python main.py --query "What is the Volatility Index (VIX) and how is it calculated?"
```

For more options:

```bash
python main.py --help
```

### Streamlit UI (Recommended)

Launch the Streamlit web interface:

```bash
python main.py --ui
```

Or directly:

```bash
streamlit run src/ui/app.py
```

The UI will be available at http://localhost:8501

### Minimal Example

For a simplified example with fewer dependencies:

```bash
python main_minimal.py
```

## Project Structure

```
.
├── config.yaml          # Application configuration
├── .env                 # Environment variables (create from .env.example)
├── main.py              # Main application entry point
├── main_minimal.py      # Simplified example
├── src/
│   ├── agents/          # Specialized agent implementations
│   ├── db/              # Database interfaces (SQLite, Qdrant)
│   ├── orchestrator/    # Swarm orchestration framework
│   ├── search/          # Search engine integrations
│   ├── ui/              # Streamlit web interface
│   └── utils/           # Utility functions
├── logs/                # Application logs
├── scraped_data/        # Storage for scraped web content
└── processed_data/      # Storage for processed research data
```

## Troubleshooting

### Common Issues

1. **Azure OpenAI API Connection Failures**
   - Verify your Azure OpenAI credentials in the `.env` file
   - Ensure the specified models (GPT-4o and text-embedding-3-small) are deployed in your Azure OpenAI resource

2. **Database Connection Issues**
   - Check the SQLite database path in the config
   - For Qdrant, ensure the service is running and accessible

3. **Module Import Errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Ensure you're running from the project root directory

### Logs

Application logs are stored in the `logs/` directory. Check these logs for detailed error information.

## Advanced Configuration

For advanced configuration options, edit the `config.yaml` file in the project root directory. This file contains settings for:

- Model parameters
- Search engines
- Database connections
- File paths
- Log settings

## Deployment Options

### Docker Deployment

You can deploy the application using Docker:

```bash
docker build -t agentic-researcher .
docker run -p 8501:8501 agentic-researcher
```

### Cloud Deployment

The application can be deployed to cloud platforms like Azure Web Apps, Google Cloud Run, or AWS Elastic Beanstalk. Follow the respective platform's Python web application deployment guides.
