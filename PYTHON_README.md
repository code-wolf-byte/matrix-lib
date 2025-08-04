# Python Bindings for Rust Matrix Library

This project provides Python bindings for the Rust matrix library, allowing you to use high-performance matrix and n-dimensional array operations directly from Python.

## Features

- **High Performance**: Rust implementation provides significant speedup over pure Python
- **Matrix Operations**: 2D matrix creation, manipulation, and arithmetic
- **N-Dimensional Arrays**: NumPy-like arrays with arbitrary dimensions
- **Type Safety**: Rust's type system ensures memory safety and prevents runtime errors
- **Easy Integration**: Simple Python API that feels natural to use

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (install via [rustup](https://rustup.rs/))
- Maturin (for building Python packages)

### Install Maturin

```bash
pip install maturin
```

### Build and Install

From the project root directory:

```bash
# Build and install in development mode
maturin develop

# Or build a wheel for distribution
maturin build
```

## Quick Start

```python
import matrix_lib_python as ml

# Create a matrix from Python list
data = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]
matrix = ml.PyMatrix.from_list(data)
print(f"Matrix shape: {matrix.dimensions()}")
print(matrix)

# Create n-dimensional arrays
arr_2d = ml.PyNDArray.zeros([3, 4])
arr_3d = ml.PyNDArray.ones([2, 3, 4])

# Matrix operations
transposed = matrix.transpose()
identity = ml.identity_matrix(3)

# Arithmetic operations
result = ml.matrix_add(matrix1, matrix2)
product = ml.matrix_mul(matrix1, matrix2)
```

## API Reference

### PyMatrix

#### Constructors

- `PyMatrix.new(rows: int, cols: int)` - Create empty matrix
- `PyMatrix.from_list(data: List[List[float]])` - Create from Python list

#### Methods

- `rows() -> int` - Get number of rows
- `cols() -> int` - Get number of columns
- `dimensions() -> Tuple[int, int]` - Get dimensions as tuple
- `get(row: int, col: int) -> float` - Get element at position
- `set(row: int, col: int, value: float)` - Set element at position
- `transpose() -> PyMatrix` - Return transposed matrix
- `to_list() -> List[List[float]]` - Convert to Python list

### PyNDArray

#### Constructors

- `PyNDArray.new(shape: List[int])` - Create empty array
- `PyNDArray.from_list(data: List, shape: Optional[List[int]] = None)` - Create from Python list
- `PyNDArray.zeros(shape: List[int])` - Create array filled with zeros
- `PyNDArray.ones(shape: List[int])` - Create array filled with ones

#### Methods

- `shape() -> List[int]` - Get array shape
- `ndim() -> int` - Get number of dimensions
- `size() -> int` - Get total number of elements
- `get(indices: List[int]) -> float` - Get element at indices
- `set(indices: List[int], value: float)` - Set element at indices
- `reshape(new_shape: List[int]) -> PyNDArray` - Reshape array
- `flatten() -> PyNDArray` - Flatten to 1D array
- `to_list() -> List` - Convert to Python list

### Utility Functions

- `matrix_add(a: PyMatrix, b: PyMatrix) -> PyMatrix` - Matrix addition
- `matrix_mul(a: PyMatrix, b: PyMatrix) -> PyMatrix` - Matrix multiplication
- `identity_matrix(size: int) -> PyMatrix` - Create identity matrix
- `matrix_to_ndarray(matrix: PyMatrix) -> PyNDArray` - Convert matrix to array
- `ndarray_to_matrix(array: PyNDArray) -> PyMatrix` - Convert array to matrix (2D only)

## Examples

### Basic Matrix Operations

```python
import matrix_lib_python as ml

# Create matrices
matrix1 = ml.PyMatrix.from_list([
    [1.0, 2.0],
    [3.0, 4.0]
])

matrix2 = ml.PyMatrix.from_list([
    [5.0, 6.0],
    [7.0, 8.0]
])

# Arithmetic operations
sum_matrix = ml.matrix_add(matrix1, matrix2)
product_matrix = ml.matrix_mul(matrix1, matrix2)

# Transpose
transposed = matrix1.transpose()

# Identity matrix
identity = ml.identity_matrix(3)
```

### N-Dimensional Arrays

```python
import matrix_lib_python as ml

# Create arrays of different dimensions
arr_1d = ml.PyNDArray.zeros([5])
arr_2d = ml.PyNDArray.ones([3, 4])
arr_3d = ml.PyNDArray.new([2, 3, 4])

# Create from nested lists
data = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]
arr_from_list = ml.PyNDArray.from_list(data)

# Access and modify elements
value = arr_from_list.get([1, 2])  # Get element at [1, 2]
arr_from_list.set([0, 1], 99.0)    # Set element at [0, 1]

# Reshape operations
reshaped = arr_from_list.reshape([3, 2])
flattened = arr_from_list.flatten()
```

### Conversion Between Types

```python
import matrix_lib_python as ml

# Matrix to NDArray
matrix = ml.PyMatrix.from_list([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])
ndarray = ml.matrix_to_ndarray(matrix)

# NDArray to Matrix (only for 2D arrays)
matrix_back = ml.ndarray_to_matrix(ndarray)

# This will fail for 3D arrays
arr_3d = ml.PyNDArray.zeros([2, 2, 3])
try:
    matrix_from_3d = ml.ndarray_to_matrix(arr_3d)
except ValueError as e:
    print(f"Conversion failed: {e}")
```

## Performance Comparison

The Rust implementation provides significant performance improvements over pure Python:

```python
import time
import matrix_lib_python as ml
import numpy as np

size = 1000

# Matrix creation performance
start = time.time()
rust_matrix = ml.PyMatrix.new(size, size)
rust_time = time.time() - start

start = time.time()
python_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
python_time = time.time() - start

print(f"Rust: {rust_time:.4f}s")
print(f"Python: {python_time:.4f}s")
print(f"Speedup: {python_time/rust_time:.2f}x")
```

## Development

### Building from Source

```bash
# Clone the repository
git clone <repository-url>
cd self-LLM

# Install in development mode
maturin develop

# Run tests
python -m pytest tests/
```

### Running Examples

```bash
# Run the Python example
python examples/python_example.py
```

### Testing

```bash
# Run Rust tests
cargo test

# Run Python tests
python -m pytest tests/
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you've built and installed the package with `maturin develop`
2. **Type Errors**: The library expects `float` values (64-bit). Convert integers to floats if needed
3. **Shape Mismatch**: Ensure matrix dimensions are compatible for operations
4. **Index Out of Bounds**: Check that indices are within valid ranges

### Debug Mode

For debugging, you can build with debug symbols:

```bash
maturin develop --release
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- [PyO3](https://pyo3.rs/) for Python bindings
- [Maturin](https://github.com/PyO3/maturin) for build system
- [NumPy](https://numpy.org/) for inspiration on array operations 