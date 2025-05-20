"""Feature Agent package for Agentic Researcher

This package contains the implementation of the Feature Agent that adds new features to the system.
"""

from src.agents.feature_agent import FeatureAgent
import os
import sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ['FeatureAgent']
