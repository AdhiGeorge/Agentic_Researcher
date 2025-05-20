"""Patcher Agent package for Agentic Researcher

This package contains the implementation of the Patcher Agent that fixes bugs in code.
"""

from src.agents.patcher_agent import PatcherAgent
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ['PatcherAgent']
