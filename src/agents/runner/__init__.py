"""Runner Agent package for Agentic Researcher

This package contains the implementation of the Runner Agent that executes code based on research results.
"""

from src.agents.runner_agent import RunnerAgentimport osimport sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ['RunnerAgent']
