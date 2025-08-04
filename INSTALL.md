# Installation Guide for Python Bindings

This guide will help you install and use the Python bindings for the Rust matrix library.

## Prerequisites

1. **Python 3.8 or higher**
   ```bash
   python --version
   ```

2. **Rust toolchain**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

3. **pip** (usually comes with Python)

## Quick Installation

### Option 1: Automated Build Script

```bash
# Clone the repository
git clone <your-repo-url>
cd self-LLM

# Run the automated build script
./build_python.sh
```

### Option 2: Manual Installation

```bash
# Install maturin (build tool)
pip install maturin

# Install development dependencies
pip install -r requirements.txt

# Build and install in development mode
maturin develop --release

# Test the installation
python -c "import matrix_lib_python as ml; print('Installation successful!')"
```

## Verification

Run the example to verify everything works:

```bash
python examples/python_example.py
```

Run the tests:

```bash
python -m pytest tests/test_python_bindings.py -v
```

## Usage

Once installed, you can use the library in Python:

```python
import matrix_lib_python as ml

# Create a matrix
matrix = ml.PyMatrix.from_list([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

# Create an array
array = ml.PyNDArray.zeros([2, 3, 4])

print(f"Matrix shape: {matrix.dimensions()}")
print(f"Array shape: {array.shape()}")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you ran `maturin develop` successfully
2. **Build Errors**: Ensure you have Rust installed and up to date
3. **Permission Errors**: Try running with `sudo` if needed (not recommended for development)

### Getting Help

- Check the [PYTHON_README.md](PYTHON_README.md) for detailed documentation
- Run `cargo test` to test the Rust code
- Run `python -m pytest tests/` to test the Python bindings

## Development

For development, install in editable mode:

```bash
maturin develop
```

This will rebuild the library automatically when you make changes to the Rust code. 