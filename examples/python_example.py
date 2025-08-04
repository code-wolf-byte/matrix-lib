#!/usr/bin/env python3
"""
Python example demonstrating usage of the Rust matrix library.

This example shows how to use the Matrix and NDArray classes from Python.
"""

import matrix_lib_python as ml

def matrix_example():
    """Demonstrate Matrix operations."""
    print("=== Matrix Operations ===")
    
    # Create a matrix from Python list
    data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ]
    
    matrix = ml.PyMatrix.from_list(data)
    print(f"Matrix shape: {matrix.dimensions()}")
    print(f"Matrix:\n{matrix}")
    
    # Access elements
    print(f"Element at (1, 2): {matrix.get(1, 2)}")
    
    # Modify elements
    matrix.set(0, 0, 10.0)
    print(f"After setting (0, 0) to 10.0:\n{matrix}")
    
    # Transpose
    transposed = matrix.transpose()
    print(f"Transposed matrix:\n{transposed}")
    
    # Create identity matrix
    identity = ml.identity_matrix(3)
    print(f"3x3 Identity matrix:\n{identity}")
    
    return matrix

def ndarray_example():
    """Demonstrate NDArray operations."""
    print("\n=== NDArray Operations ===")
    
    # Create arrays with different shapes
    arr_2d = ml.PyNDArray.zeros([2, 3])
    print(f"2D zeros array shape: {arr_2d.shape()}")
    print(f"2D array:\n{arr_2d}")
    
    arr_3d = ml.PyNDArray.ones([2, 2, 3])
    print(f"3D ones array shape: {arr_3d.shape()}")
    print(f"3D array size: {arr_3d.size()}")
    
    # Create from Python list
    data_2d = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
    arr_from_list = ml.PyNDArray.from_list(data_2d)
    print(f"Array from list:\n{arr_from_list}")
    
    # Access elements
    print(f"Element at [1, 2]: {arr_from_list.get([1, 2])}")
    
    # Modify elements
    arr_from_list.set([0, 1], 99.0)
    print(f"After setting [0, 1] to 99.0:\n{arr_from_list}")
    
    # Reshape
    reshaped = arr_from_list.reshape([3, 2])
    print(f"Reshaped to [3, 2]:\n{reshaped}")
    
    # Flatten
    flattened = arr_from_list.flatten()
    print(f"Flattened array shape: {flattened.shape()}")
    print(f"Flattened array: {flattened}")
    
    return arr_from_list

def matrix_operations_example():
    """Demonstrate matrix arithmetic operations."""
    print("\n=== Matrix Arithmetic ===")
    
    # Create two matrices
    data1 = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
    data2 = [
        [5.0, 6.0],
        [7.0, 8.0]
    ]
    
    matrix1 = ml.PyMatrix.from_list(data1)
    matrix2 = ml.PyMatrix.from_list(data2)
    
    print(f"Matrix 1:\n{matrix1}")
    print(f"Matrix 2:\n{matrix2}")
    
    # Addition
    try:
        result_add = ml.matrix_add(matrix1, matrix2)
        print(f"Matrix addition:\n{result_add}")
    except Exception as e:
        print(f"Addition error: {e}")
    
    # Multiplication
    try:
        result_mul = ml.matrix_mul(matrix1, matrix2)
        print(f"Matrix multiplication:\n{result_mul}")
    except Exception as e:
        print(f"Multiplication error: {e}")

def conversion_example():
    """Demonstrate conversion between Matrix and NDArray."""
    print("\n=== Matrix-NDArray Conversion ===")
    
    # Create a matrix
    matrix_data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
    matrix = ml.PyMatrix.from_list(matrix_data)
    print(f"Original matrix:\n{matrix}")
    
    # Convert to NDArray
    ndarray = ml.matrix_to_ndarray(matrix)
    print(f"Converted to NDArray shape: {ndarray.shape()}")
    print(f"NDArray:\n{ndarray}")
    
    # Convert back to Matrix
    matrix_back = ml.ndarray_to_matrix(ndarray)
    print(f"Converted back to Matrix:\n{matrix_back}")
    
    # Try to convert 3D array to matrix (should fail)
    arr_3d = ml.PyNDArray.zeros([2, 2, 3])
    try:
        matrix_from_3d = ml.ndarray_to_matrix(arr_3d)
        print(f"3D to Matrix: {matrix_from_3d}")
    except Exception as e:
        print(f"3D to Matrix conversion failed (expected): {e}")

def performance_comparison():
    """Compare performance with Python lists."""
    print("\n=== Performance Comparison ===")
    
    import time
    import numpy as np
    
    size = 100
    
    # Test matrix creation
    start = time.time()
    matrix = ml.PyMatrix.new(size, size)
    rust_time = time.time() - start
    print(f"Rust Matrix creation ({size}x{size}): {rust_time:.4f}s")
    
    # Test Python list creation
    start = time.time()
    python_matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    python_time = time.time() - start
    print(f"Python list creation ({size}x{size}): {python_time:.4f}s")
    
    # Test NumPy creation
    start = time.time()
    numpy_matrix = np.zeros((size, size))
    numpy_time = time.time() - start
    print(f"NumPy creation ({size}x{size}): {numpy_time:.4f}s")
    
    print(f"Speedup vs Python: {python_time/rust_time:.2f}x")
    print(f"Speedup vs NumPy: {numpy_time/rust_time:.2f}x")

def main():
    """Run all examples."""
    print("Rust Matrix Library Python Bindings Demo")
    print("=" * 50)
    
    try:
        matrix_example()
        ndarray_example()
        matrix_operations_example()
        conversion_example()
        performance_comparison()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 