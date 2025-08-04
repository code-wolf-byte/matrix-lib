#!/bin/bash

# Build and test Python bindings for Rust matrix library

set -e  # Exit on any error

echo "Building Python bindings for Rust matrix library..."
echo "=================================================="

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Maturin not found. Installing..."
    pip install maturin
fi

# Check if pytest is installed
if ! python -c "import pytest" &> /dev/null; then
    echo "Pytest not found. Installing..."
    pip install pytest
fi

# Clean previous builds
echo "Cleaning previous builds..."
cargo clean

# Build and install in development mode
echo "Building and installing Python package..."
maturin develop --release

# Run Rust tests
echo "Running Rust tests..."
cargo test

# Run Python tests
echo "Running Python tests..."
python -m pytest tests/test_python_bindings.py -v

# Run example
echo "Running Python example..."
python examples/python_example.py

echo ""
echo "Build and test completed successfully!"
echo ""
echo "You can now use the library in Python:"
echo "  import matrix_lib_python as ml"
echo "  matrix = ml.PyMatrix.from_list([[1.0, 2.0], [3.0, 4.0]])"
echo ""
echo "To build a distributable wheel:"
echo "  maturin build --release" 