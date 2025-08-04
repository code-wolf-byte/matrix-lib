# Installing from GitHub

This guide shows you how to install the matrix-lib-python package from GitHub using different methods.

## Prerequisites

- Python 3.8 or higher
- pip (usually comes with Python)
- Rust toolchain (for source installation)

## Method 1: Direct GitHub Installation (Recommended)

### From GitHub Repository

```bash
# Install directly from GitHub
pip install git+https://github.com/code-wolf-byte/matrix-lib.git

# Or with specific branch/tag
pip install git+https://github.com/code-wolf-byte/matrix-lib.git@main
pip install git+https://github.com/code-wolf-byte/matrix-lib.git@v0.1.0
```

### From GitHub with Subdirectory

If the Python package is in a subdirectory:

```bash
pip install git+https://github.com/code-wolf-byte/matrix-lib.git#subdirectory=python
```

## Method 2: Using Requirements.txt

Add to your `requirements.txt`:

```txt
# Install from GitHub
git+https://github.com/code-wolf-byte/matrix-lib.git

# Or with specific version
git+https://github.com/code-wolf-byte/matrix-lib.git@v0.1.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Method 3: Development Installation

For development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/code-wolf-byte/matrix-lib.git
cd matrix-lib

# Install in editable mode
pip install -e .
```

## Method 4: Using Pre-built Wheels (When Available)

When we release pre-built wheels on GitHub:

```bash
# Download and install specific wheel
pip install https://github.com/code-wolf-byte/matrix-lib/releases/download/v0.1.0/matrix_lib_python-0.1.0-cp39-cp39-linux_x86_64.whl
```

## Method 5: Using Poetry

If you're using Poetry:

```bash
# Add to pyproject.toml
poetry add git+https://github.com/code-wolf-byte/matrix-lib.git

# Or with specific version
poetry add git+https://github.com/code-wolf-byte/matrix-lib.git#v0.1.0
```

## Method 6: Using Pipenv

If you're using Pipenv:

```bash
pipenv install git+https://github.com/code-wolf-byte/matrix-lib.git#egg=matrix-lib-python
```

## Verification

After installation, verify it works:

```python
import matrix_lib_python as ml

# Create a simple matrix
matrix = ml.PyMatrix.new(2, 2)
print(f"Matrix created: {matrix}")

# Create an array
array = ml.PyNDArray.zeros([3, 3])
print(f"Array shape: {array.shape()}")
```

## Troubleshooting

### Common Issues

1. **Rust not found**: Install Rust from https://rustup.rs/
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env
   ```

2. **Build tools missing**: Install build essentials
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential
   
   # macOS
   xcode-select --install
   
   # Windows
   # Install Visual Studio Build Tools
   ```

3. **Permission errors**: Use virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install git+https://github.com/code-wolf-byte/matrix-lib.git
   ```

4. **Network issues**: Use SSH instead of HTTPS
   ```bash
   pip install git+ssh://git@github.com/code-wolf-byte/matrix-lib.git
   ```

### Platform-Specific Notes

#### Linux
- Usually works out of the box
- May need `build-essential` package

#### macOS
- Requires Xcode Command Line Tools
- May need to install Rust separately

#### Windows
- Requires Visual Studio Build Tools
- May need to install Rust separately
- Consider using WSL for easier development

## Development Setup

For contributing to the project:

```bash
# Clone the repository
git clone https://github.com/code-wolf-byte/matrix-lib.git
cd matrix-lib

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .

# Run tests
python -m pytest tests/

# Run examples
python examples/python_example.py
```

## Updating

To update to the latest version:

```bash
# If installed from GitHub
pip install --upgrade git+https://github.com/code-wolf-byte/matrix-lib.git

# If installed in editable mode
cd matrix-lib
git pull
pip install -e .
```

## Uninstalling

```bash
pip uninstall matrix-lib-python
```

## Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Look at [GitHub Issues](https://github.com/code-wolf-byte/matrix-lib/issues)
3. Create a new issue with:
   - Your operating system
   - Python version
   - Rust version
   - Full error message
   - Steps to reproduce 