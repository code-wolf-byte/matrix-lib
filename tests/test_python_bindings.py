#!/usr/bin/env python3
"""
Tests for Python bindings of the Rust matrix library.
"""

import pytest
import matrix_lib_python as ml


class TestPyMatrix:
    """Test PyMatrix functionality."""
    
    def test_matrix_creation(self):
        """Test matrix creation methods."""
        # Test new matrix
        matrix = ml.PyMatrix.new(2, 3)
        assert matrix.rows() == 2
        assert matrix.cols() == 3
        assert matrix.dimensions() == (2, 3)
        
        # Test from list
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix = ml.PyMatrix.from_list(data)
        assert matrix.rows() == 2
        assert matrix.cols() == 3
        assert matrix.get(0, 0) == 1.0
        assert matrix.get(1, 2) == 6.0
    
    def test_matrix_access(self):
        """Test matrix element access and modification."""
        matrix = ml.PyMatrix.new(2, 2)
        
        # Test setting and getting elements
        matrix.set(0, 0, 1.0)
        matrix.set(0, 1, 2.0)
        matrix.set(1, 0, 3.0)
        matrix.set(1, 1, 4.0)
        
        assert matrix.get(0, 0) == 1.0
        assert matrix.get(0, 1) == 2.0
        assert matrix.get(1, 0) == 3.0
        assert matrix.get(1, 1) == 4.0
    
    def test_matrix_transpose(self):
        """Test matrix transpose operation."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix = ml.PyMatrix.from_list(data)
        transposed = matrix.transpose()
        
        assert transposed.rows() == 3
        assert transposed.cols() == 2
        assert transposed.get(0, 0) == 1.0
        assert transposed.get(0, 1) == 4.0
        assert transposed.get(1, 0) == 2.0
        assert transposed.get(1, 1) == 5.0
    
    def test_identity_matrix(self):
        """Test identity matrix creation."""
        identity = ml.identity_matrix(3)
        assert identity.rows() == 3
        assert identity.cols() == 3
        assert identity.get(0, 0) == 1.0
        assert identity.get(1, 1) == 1.0
        assert identity.get(2, 2) == 1.0
        assert identity.get(0, 1) == 0.0
        assert identity.get(1, 0) == 0.0
    
    def test_matrix_to_list(self):
        """Test conversion to Python list."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        matrix = ml.PyMatrix.from_list(data)
        result = matrix.to_list()
        
        assert result == [[1.0, 2.0], [3.0, 4.0]]
    
    def test_matrix_string_representation(self):
        """Test string representation of matrices."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        matrix = ml.PyMatrix.from_list(data)
        
        str_repr = str(matrix)
        assert "1" in str_repr
        assert "2" in str_repr
        assert "3" in str_repr
        assert "4" in str_repr


class TestPyNDArray:
    """Test PyNDArray functionality."""
    
    def test_ndarray_creation(self):
        """Test NDArray creation methods."""
        # Test new array
        arr = ml.PyNDArray.new([2, 3])
        assert arr.shape() == [2, 3]
        assert arr.ndim() == 2
        assert arr.size() == 6
        
        # Test zeros
        arr = ml.PyNDArray.zeros([2, 3])
        assert arr.get([0, 0]) == 0.0
        assert arr.get([1, 2]) == 0.0
        
        # Test ones
        arr = ml.PyNDArray.ones([2, 3])
        assert arr.get([0, 0]) == 1.0
        assert arr.get([1, 2]) == 1.0
    
    def test_ndarray_from_list(self):
        """Test NDArray creation from Python list."""
        # 2D array
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        arr = ml.PyNDArray.from_list(data)
        assert arr.shape() == [2, 3]
        assert arr.get([0, 0]) == 1.0
        assert arr.get([1, 2]) == 6.0
        
        # 1D array
        data_1d = [1.0, 2.0, 3.0, 4.0]
        arr_1d = ml.PyNDArray.from_list(data_1d)
        assert arr_1d.shape() == [4]
        assert arr_1d.get([0]) == 1.0
        assert arr_1d.get([3]) == 4.0
    
    def test_ndarray_access(self):
        """Test NDArray element access and modification."""
        arr = ml.PyNDArray.zeros([2, 3])
        
        # Test setting and getting elements
        arr.set([0, 0], 1.0)
        arr.set([0, 1], 2.0)
        arr.set([1, 2], 3.0)
        
        assert arr.get([0, 0]) == 1.0
        assert arr.get([0, 1]) == 2.0
        assert arr.get([1, 2]) == 3.0
    
    def test_ndarray_reshape(self):
        """Test NDArray reshape operation."""
        arr = ml.PyNDArray.ones([2, 3])
        reshaped = arr.reshape([3, 2])
        
        assert reshaped.shape() == [3, 2]
        assert reshaped.size() == 6
        assert reshaped.get([0, 0]) == 1.0
    
    def test_ndarray_flatten(self):
        """Test NDArray flatten operation."""
        arr = ml.PyNDArray.ones([2, 3])
        flattened = arr.flatten()
        
        assert flattened.shape() == [6]
        assert flattened.ndim() == 1
        assert flattened.size() == 6
        assert flattened.get([0]) == 1.0
        assert flattened.get([5]) == 1.0
    
    def test_ndarray_to_list(self):
        """Test conversion to Python list."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        arr = ml.PyNDArray.from_list(data)
        result = arr.to_list()
        
        assert result == [[1.0, 2.0], [3.0, 4.0]]


class TestMatrixOperations:
    """Test matrix arithmetic operations."""
    
    def test_matrix_addition(self):
        """Test matrix addition."""
        data1 = [[1.0, 2.0], [3.0, 4.0]]
        data2 = [[5.0, 6.0], [7.0, 8.0]]
        
        matrix1 = ml.PyMatrix.from_list(data1)
        matrix2 = ml.PyMatrix.from_list(data2)
        
        result = ml.matrix_add(matrix1, matrix2)
        
        assert result.get(0, 0) == 6.0
        assert result.get(0, 1) == 8.0
        assert result.get(1, 0) == 10.0
        assert result.get(1, 1) == 12.0
    
    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        data1 = [[1.0, 2.0], [3.0, 4.0]]
        data2 = [[5.0, 6.0], [7.0, 8.0]]
        
        matrix1 = ml.PyMatrix.from_list(data1)
        matrix2 = ml.PyMatrix.from_list(data2)
        
        result = ml.matrix_mul(matrix1, matrix2)
        
        assert result.get(0, 0) == 19.0
        assert result.get(0, 1) == 22.0
        assert result.get(1, 0) == 43.0
        assert result.get(1, 1) == 50.0


class TestConversions:
    """Test conversions between Matrix and NDArray."""
    
    def test_matrix_to_ndarray(self):
        """Test Matrix to NDArray conversion."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        matrix = ml.PyMatrix.from_list(data)
        
        ndarray = ml.matrix_to_ndarray(matrix)
        
        assert ndarray.shape() == [2, 3]
        assert ndarray.get([0, 0]) == 1.0
        assert ndarray.get([1, 2]) == 6.0
    
    def test_ndarray_to_matrix(self):
        """Test NDArray to Matrix conversion."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ndarray = ml.PyNDArray.from_list(data)
        
        matrix = ml.ndarray_to_matrix(ndarray)
        
        assert matrix.dimensions() == (2, 3)
        assert matrix.get(0, 0) == 1.0
        assert matrix.get(1, 2) == 6.0
    
    def test_ndarray_to_matrix_3d_error(self):
        """Test that 3D NDArray to Matrix conversion fails."""
        arr_3d = ml.PyNDArray.zeros([2, 2, 3])
        
        with pytest.raises(ValueError):
            ml.ndarray_to_matrix(arr_3d)


class TestErrorHandling:
    """Test error handling."""
    
    def test_matrix_index_out_of_bounds(self):
        """Test matrix index out of bounds error."""
        matrix = ml.PyMatrix.new(2, 2)
        
        with pytest.raises(Exception):
            matrix.get(2, 0)  # Row out of bounds
        
        with pytest.raises(Exception):
            matrix.get(0, 2)  # Column out of bounds
    
    def test_ndarray_index_out_of_bounds(self):
        """Test NDArray index out of bounds error."""
        arr = ml.PyNDArray.zeros([2, 3])
        
        with pytest.raises(Exception):
            arr.get([2, 0])  # First index out of bounds
        
        with pytest.raises(Exception):
            arr.get([0, 3])  # Second index out of bounds
    
    def test_invalid_reshape(self):
        """Test invalid reshape operation."""
        arr = ml.PyNDArray.zeros([2, 3])  # Size = 6
        
        with pytest.raises(ValueError):
            arr.reshape([2, 4])  # Size = 8, should fail


if __name__ == "__main__":
    pytest.main([__file__]) 