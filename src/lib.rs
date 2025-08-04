//! # Matrix Library
//!
//! A generic matrix library for Rust providing basic linear algebra operations and n-dimensional arrays.
//!
//! ## Features
//!
//! - **2D Matrix**: Optimized 2D matrix operations with intuitive API
//! - **N-dimensional Arrays**: NumPy-like nDArray for arbitrary dimensions
//! - **Generic Implementation**: Works with any type implementing required traits
//! - **Basic Operations**: Addition, multiplication, transpose, and more
//! - **Safe Indexing**: Bounds checking with panic-free alternatives
//! - **Interoperability**: Seamless conversion between Matrix and NDArray
//!
//! ## Examples
//!
//! ### Matrix Operations
//! ```
//! use matrix_lib::Matrix;
//!
//! // Create a matrix from a vector
//! let data = vec![
//!     vec![1, 2, 3],
//!     vec![4, 5, 6],
//! ];
//! let matrix = Matrix::from_vec(data).unwrap();
//!
//! // Access elements
//! assert_eq!(matrix[(0, 1)], 2);
//!
//! // Matrix operations
//! let transposed = matrix.transpose();
//! ```
//!
//! ### N-Dimensional Arrays
//! ```
//! use matrix_lib::NDArray;
//!
//! // Create a 3D array
//! let arr = NDArray::<i32>::zeros(&[2, 3, 4]);
//!
//! // Access elements
//! let value = arr.get(&[1, 2, 3]).unwrap();
//!
//! // Reshape
//! let reshaped = arr.reshape(&[6, 4]).unwrap();
//! ```
//!
//! ### Interoperability
//! ```
//! use matrix_lib::{Matrix, NDArray};
//!
//! // Convert Matrix to NDArray
//! let matrix = Matrix::new(2, 3);
//! let nd_array = NDArray::from(matrix);
//!
//! // Convert NDArray back to Matrix (for 2D arrays)
//! let matrix_back: Matrix<i32> = nd_array.try_into().unwrap();
//! ```

pub mod utils;
pub mod python_bindings;

// Re-export the main types for easier access
pub use utils::{Matrix, NDArray, NDArrayError};

// Re-export commonly used traits
pub use std::ops::{Add, Mul, Index, IndexMut};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_matrix_ndarray_interoperability() {
        // Create a matrix
        let matrix = Matrix::from_vec(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]).unwrap();

        // Convert to NDArray
        let nd_array = NDArray::from(matrix);
        assert_eq!(nd_array.shape(), &[2, 3]);
        assert_eq!(nd_array.get(&[0, 0]), Some(&1));
        assert_eq!(nd_array.get(&[1, 2]), Some(&6));

        // Convert back to Matrix
        let matrix_back: Matrix<i32> = nd_array.try_into().unwrap();
        assert_eq!(matrix_back.dimensions(), (2, 3));
        assert_eq!(matrix_back[(0, 0)], 1);
        assert_eq!(matrix_back[(1, 2)], 6);
    }

    #[test]
    fn test_ndarray_operations() {
        // Create 3D array
        let arr = NDArray::<i32>::zeros(&[2, 3, 4]);
        assert_eq!(arr.shape(), &[2, 3, 4]);
        assert_eq!(arr.ndim(), 3);
        assert_eq!(arr.size(), 24);

        // Test reshaping
        let reshaped = arr.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(reshaped.size(), 24);

        // Test flattening
        let flattened = arr.flatten();
        assert_eq!(flattened.shape(), &[24]);
        assert_eq!(flattened.ndim(), 1);
    }

    #[test]
    fn test_ndarray_creation_methods() {
        // Test zeros
        let zeros: NDArray<i32> = NDArray::zeros(&[2, 3]);
        assert_eq!(zeros.get(&[0, 0]), Some(&0));
        assert_eq!(zeros.get(&[1, 2]), Some(&0));

        // Test ones
        let ones: NDArray<i32> = NDArray::ones(&[2, 3]);
        assert_eq!(ones.get(&[0, 0]), Some(&1));
        assert_eq!(ones.get(&[1, 2]), Some(&1));

        // Test from_vec
        let data = vec![1, 2, 3, 4, 5, 6];
        let arr = NDArray::from_vec(data, &[2, 3]).unwrap();
        assert_eq!(arr.get(&[0, 0]), Some(&1));
        assert_eq!(arr.get(&[1, 2]), Some(&6));
    }

    #[test]
    fn test_error_handling() {
        // Test shape mismatch
        let data = vec![1, 2, 3, 4];
        let result = NDArray::from_vec(data, &[2, 3]);
        assert!(result.is_err());

        // Test invalid reshape
        let arr = NDArray::from_vec(vec![1, 2, 3, 4], &[2, 2]).unwrap();
        let result = arr.reshape(&[2, 3]);
        assert!(result.is_err());

        // Test dimension error for Matrix conversion
        let arr = NDArray::<i32>::zeros(&[2, 3, 4]);
        let result: Result<Matrix<i32>, _> = arr.try_into();
        assert!(result.is_err());
    }

    #[test]
    fn test_library_usage_patterns() {
        // Test typical usage patterns
        let matrix: Matrix<i32> = Matrix::new(3, 3);
        let nd_array = NDArray::from(matrix);
        
        // Test that we can work with both types
        assert_eq!(nd_array.ndim(), 2);
        
        // Test reshaping to different dimensions
        let reshaped = nd_array.reshape(&[9]).unwrap();
        assert_eq!(reshaped.ndim(), 1);
        assert_eq!(reshaped.size(), 9);
    }
} 