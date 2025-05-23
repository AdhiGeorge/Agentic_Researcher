import os
import sys

# Import the ResearcherAgent from the local module
from src.agents.researcher.researcher_agent import ResearcherAgent


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ["ResearcherAgent"]
