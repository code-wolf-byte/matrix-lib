#!/usr/bin/env python3
"""
Setup script for matrix-lib-python
Supports installation from GitHub and PyPI
"""

import os
import sys
from setuptools import setup, find_packages

# Read the README file
def read_readme():
    with open("PYTHON_README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Check if we're building with maturin
if os.path.exists("pyproject.toml"):
    # Use pyproject.toml configuration
    setup()
else:
    # Fallback setup.py for direct installation
    setup(
        name="matrix-lib-python",
        version="0.1.0",
        author="Your Name",
        author_email="your.email@example.com",
        description="High-performance matrix and n-dimensional array library with Python bindings",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/code-wolf-byte/matrix-lib",
        project_urls={
            "Bug Tracker": "https://github.com/code-wolf-byte/matrix-lib/issues",
            "Documentation": "https://github.com/code-wolf-byte/matrix-lib/blob/main/PYTHON_README.md",
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "License :: OSI Approved :: Apache Software License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
        packages=find_packages(),
        install_requires=[],
        extras_require={
            "dev": [
                "pytest>=6.0",
                "numpy>=1.20",
                "black>=22.0",
                "flake8>=4.0",
                "mypy>=0.950",
            ],
        },
        keywords=["matrix", "linear-algebra", "math", "mathematics", "numpy", "rust"],
    ) 