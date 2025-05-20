"""Decision Agent package for Agentic Researcher

This package contains the implementation of the Decision Agent that analyzes options and makes decisions.
"""

from src.agents.decision_agent import DecisionAgent
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ['DecisionAgent']
