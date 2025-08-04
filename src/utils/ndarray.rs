use std::fmt::{self, Display, Formatter};
use super::matrix::Matrix;

#[derive(Debug, PartialEq)]
pub enum NDArrayError {
    ShapeMismatch,
    IndexOutOfBounds,
    InvalidShape,
    BroadcastError,
    DimensionError,
}

type Result<T> = std::result::Result<T, NDArrayError>;

#[derive(Debug, Clone, PartialEq)]
pub struct NDArray<T> {
    data: Vec<T>,           // Flat storage for all elements
    shape: Vec<usize>,      // Dimensions [d0, d1, d2, ...]
    strides: Vec<usize>,    // Memory strides for indexing
}

impl<T> NDArray<T> 
where 
    T: Clone + Default,
{
    /// Create a new nDArray with specified shape, filled with default values
    pub fn new(shape: &[usize]) -> Self {
        Self::with_value(shape, T::default())
    }

    /// Create a new nDArray with specified shape, filled with a specific value
    pub fn with_value(shape: &[usize], value: T) -> Self {
        let total_size = shape.iter().product();
        let data = vec![value; total_size];
        let strides = Self::calculate_strides(shape);
        
        NDArray {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Create nDArray from flat data and shape
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(NDArrayError::ShapeMismatch);
        }
        
        let strides = Self::calculate_strides(shape);
        Ok(NDArray {
            data,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// Calculate strides for given shape (row-major order)
    fn calculate_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        
        for &dim in shape.iter().rev() {
            strides.push(stride);
            stride *= dim;
        }
        
        strides.reverse();
        strides
    }

    /// Convert multi-dimensional indices to flat index
    fn flat_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }
        
        let mut flat_idx = 0;
        for (i, (&idx, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
            flat_idx += idx * stride;
        }
        Some(flat_idx)
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Check if array is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at specified indices
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        self.flat_index(indices).and_then(|idx| self.data.get(idx))
    }

    /// Get mutable element at specified indices
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        self.flat_index(indices).and_then(move |idx| self.data.get_mut(idx))
    }

    /// Set element at specified indices
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<()> {
        if let Some(idx) = self.flat_index(indices) {
            self.data[idx] = value;
            Ok(())
        } else {
            Err(NDArrayError::IndexOutOfBounds)
        }
    }

    /// Reshape the array to new shape
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(NDArrayError::ShapeMismatch);
        }
        
        let new_strides = Self::calculate_strides(new_shape);
        Ok(NDArray {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
            strides: new_strides,
        })
    }

    /// Flatten the array to 1D
    pub fn flatten(&self) -> Self {
        self.reshape(&[self.size()]).unwrap()
    }

    /// Get a specific axis (dimension) size
    pub fn axis_size(&self, axis: usize) -> Result<usize> {
        self.shape.get(axis).copied().ok_or(NDArrayError::DimensionError)
    }
}

impl<T> NDArray<T> 
where 
    T: Clone + Default + From<i32>,
{
    /// Create array filled with zeros
    pub fn zeros(shape: &[usize]) -> Self {
        Self::with_value(shape, T::from(0))
    }

    /// Create array filled with ones
    pub fn ones(shape: &[usize]) -> Self {
        Self::with_value(shape, T::from(1))
    }
}

// Conversion from Matrix to NDArray
impl<T> From<Matrix<T>> for NDArray<T> 
where 
    T: Clone + Default,
{
    fn from(matrix: Matrix<T>) -> Self {
        let rows = matrix.rows();
        let cols = matrix.cols();
        let shape = vec![rows, cols];
        
        // Flatten matrix data
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(matrix[(r, c)].clone());
            }
        }
        
        Self::from_vec(data, &shape).unwrap()
    }
}

// Conversion from NDArray to Matrix (only for 2D arrays)
impl<T> TryFrom<NDArray<T>> for Matrix<T> 
where 
    T: Clone + Default,
{
    type Error = NDArrayError;
    
    fn try_from(array: NDArray<T>) -> Result<Matrix<T>> {
        if array.ndim() != 2 {
            return Err(NDArrayError::DimensionError);
        }
        
        let rows = array.shape[0];
        let cols = array.shape[1];
        
        // Convert flat data to 2D structure
        let mut matrix_data = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(array.get(&[r, c]).unwrap().clone());
            }
            matrix_data.push(row);
        }
        
        Matrix::from_vec(matrix_data).map_err(|_| NDArrayError::ShapeMismatch)
    }
}

// Display implementation
impl<T> Display for NDArray<T> 
where 
    T: Display + Clone + Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.ndim() {
            1 => self.fmt_1d(f),
            2 => self.fmt_2d(f),
            _ => write!(f, "NDArray with shape {:?}", self.shape),
        }
    }
}

impl<T> NDArray<T> 
where 
    T: Display + Clone + Default,
{
    fn fmt_1d(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", item)?;
        }
        write!(f, "]")
    }

    fn fmt_2d(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let rows = self.shape[0];
        let cols = self.shape[1];
        
        for r in 0..rows {
            write!(f, "[")?;
            for c in 0..cols {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(&[r, c]).unwrap())?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// Iterator support
impl<T> NDArray<T> {
    /// Create an iterator over all elements
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }

    /// Create a mutable iterator over all elements
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.data.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_creation() {
        let arr: NDArray<i32> = NDArray::new(&[2, 3, 4]);
        assert_eq!(arr.shape(), &[2, 3, 4]);
        assert_eq!(arr.ndim(), 3);
        assert_eq!(arr.size(), 24);
    }

    #[test]
    fn test_stride_calculation() {
        let arr: NDArray<i32> = NDArray::new(&[2, 3, 4]);
        assert_eq!(arr.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_flat_index() {
        let arr: NDArray<i32> = NDArray::new(&[2, 3, 4]);
        assert_eq!(arr.flat_index(&[1, 2, 3]), Some(23));
        assert_eq!(arr.flat_index(&[0, 0, 0]), Some(0));
        assert_eq!(arr.flat_index(&[2, 0, 0]), None); // Out of bounds
    }

    #[test]
    fn test_get_set() {
        let mut arr: NDArray<i32> = NDArray::new(&[2, 3]);
        arr.set(&[1, 2], 42).unwrap();
        assert_eq!(arr.get(&[1, 2]), Some(&42));
    }

    #[test]
    fn test_reshape() {
        let arr: NDArray<i32> = NDArray::new(&[2, 3, 4]);
        let reshaped = arr.reshape(&[6, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4]);
        assert_eq!(reshaped.size(), 24);
    }

    #[test]
    fn test_from_matrix() {
        let matrix = Matrix::from_vec(vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ]).unwrap();
        
        let arr = NDArray::from(matrix);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.get(&[0, 0]), Some(&1));
        assert_eq!(arr.get(&[1, 2]), Some(&6));
    }

    #[test]
    fn test_to_matrix() {
        let arr = NDArray::from_vec(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let matrix: Matrix<i32> = arr.try_into().unwrap();
        assert_eq!(matrix.dimensions(), (2, 3));
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    fn test_zeros_ones() {
        let zeros: NDArray<i32> = NDArray::zeros(&[2, 3]);
        assert_eq!(zeros.get(&[0, 0]), Some(&0));
        
        let ones: NDArray<i32> = NDArray::ones(&[2, 3]);
        assert_eq!(ones.get(&[0, 0]), Some(&1));
    }

    #[test]
    fn test_flatten() {
        let arr = NDArray::from_vec(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
        let flattened = arr.flatten();
        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.get(&[0]), Some(&1));
        assert_eq!(flattened.get(&[5]), Some(&6));
    }
}

