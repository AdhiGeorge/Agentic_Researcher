"""Agents package for Agentic Researcher

This package contains various agent implementations for the research system:
- BaseAgent: Foundation class for all agents
- CoderAgent: Generates code based on research findings
- DecisionAgent: Analyzes options and makes decisions
- FeatureAgent: Adds new features to the project
- PatcherAgent: Fixes bugs in generated code
- ReporterAgent: Creates reports from research results
- RunnerAgent: Executes generated code and captures outputs
- ResearcherAgent: Performs web searches and content extraction
- PlannerAgent: Creates structured research plans (legacy)
- FormatterAgent: Processes and structures research content (legacy)
- ActionAgent: Handles actions based on research results (legacy)
- AnswerAgent: Generates answers to research queries (legacy)
- InternalMonologueAgent: Provides human-like reasoning (legacy)
"""
import os
import sys

# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import only the base agent - we'll avoid importing all specific agents here to prevent circular imports
from src.agents.base_agent import BaseAgent

# Define what should be accessible from this package
# Users will need to import specific agents directly as needed
__all__ = ['BaseAgent']

# Note: We're avoiding importing all agent modules directly here to prevent circular imports.
# Instead, consumers should import specific agents directly as needed:
# from src.agents.coder.coder_agent import CoderAgent
# from src.agents.decision.decision_agent import DecisionAgent
# etc.