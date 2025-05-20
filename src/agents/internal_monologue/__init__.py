"""Internal Monologue agent package

This package provides the Internal Monologue agent implementation
for the Agentic Researcher system.
"""

from src.agents.internal_monologue_agent import InternalMonologueAgentimport osimport sys


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ['InternalMonologueAgent']
