#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for Agentic Researcher
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="agentic_researcher",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Agent-based research system using Swarm for orchestration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agentic-researcher",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "agentic-researcher=main_app:main",
        ],
    },
)
