"""Coder Agent package for Agentic Researcher

This package contains the implementation of the Coder Agent that generates code based on research results.
"""

from src.agents.coder.coder_agent import CoderAgent
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ['CoderAgent']
