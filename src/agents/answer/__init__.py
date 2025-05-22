"""Answer agent package

This package provides the Answer agent implementation
for the Agentic Researcher system.
"""

import os
import sys

# Import the AnswerAgent from the local module
from src.agents.answer.answer_agent import AnswerAgent


# Add project root to the Python path to enable imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


__all__ = ['AnswerAgent']
