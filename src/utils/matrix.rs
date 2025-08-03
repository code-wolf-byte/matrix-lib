use std::ops::{Index, IndexMut, Add, Mul};
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    data: Vec<Vec<T>>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T> 
where 
    T: Clone + Default,
{
    /// Create a new matrix with specified dimensions, filled with default values
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![vec![T::default(); cols]; rows];
        Matrix { data, rows, cols }
    }

    /// Create a matrix from a 2D vector
    pub fn from_vec(data: Vec<Vec<T>>) -> Result<Self, &'static str> {
        if data.is_empty() {
            return Err("Matrix cannot be empty");
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Check if all rows have the same length
        for row in &data {
            if row.len() != cols {
                return Err("All rows must have the same length");
            }
        }
        
        Ok(Matrix { data, rows, cols })
    }

    /// Get the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Get the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Get the dimensions as a tuple (rows, cols)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get a reference to the element at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.data.get(row)?.get(col)
    }

    /// Get a mutable reference to the element at (row, col)
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.data.get_mut(row)?.get_mut(col)
    }

    /// Set the value at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<(), &'static str> {
        if row >= self.rows || col >= self.cols {
            return Err("Index out of bounds");
        }
        self.data[row][col] = value;
        Ok(())
    }

    /// Get a reference to a specific row
    pub fn row(&self, row: usize) -> Option<&Vec<T>> {
        self.data.get(row)
    }

    /// Get a column as a vector
    pub fn col(&self, col: usize) -> Option<Vec<T>> {
        if col >= self.cols {
            return None;
        }
        Some(self.data.iter().map(|row| row[col].clone()).collect())
    }
}

impl<T> Matrix<T> 
where 
    T: Clone + Default + Add<Output = T>,
{
    /// Create an identity matrix (only works for square matrices with numeric types)
    pub fn identity(size: usize) -> Self 
    where 
        T: From<i32>,
    {
        let mut matrix = Self::new(size, size);
        for i in 0..size {
            matrix.data[i][i] = T::from(1);
        }
        matrix
    }

    /// Transpose the matrix
    pub fn transpose(&self) -> Matrix<T> {
        let mut transposed = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                transposed.data[j][i] = self.data[i][j].clone();
            }
        }
        transposed
    }
}

// Implement indexing
impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0][index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.data[index.0][index.1]
    }
}

// Implement Display trait for pretty printing
impl<T> Display for Matrix<T> 
where 
    T: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            write!(f, "[")?;
            for (i, item) in row.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", item)?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// Implement matrix addition
impl<T> Add for Matrix<T> 
where 
    T: Clone + Default + Add<Output = T>,
{
    type Output = Result<Matrix<T>, &'static str>;

    fn add(self, other: Matrix<T>) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err("Matrices must have the same dimensions for addition");
        }

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j].clone() + other.data[i][j].clone();
            }
        }
        Ok(result)
    }
}

// Implement matrix multiplication
impl<T> Mul for Matrix<T> 
where 
    T: Clone + Default + Add<Output = T> + Mul<Output = T>,
{
    type Output = Result<Matrix<T>, &'static str>;

    fn mul(self, other: Matrix<T>) -> Self::Output {
        if self.cols != other.rows {
            return Err("Number of columns in first matrix must equal number of rows in second matrix");
        }

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = T::default();
                for k in 0..self.cols {
                    sum = sum + self.data[i][k].clone() * other.data[k][j].clone();
                }
                result.data[i][j] = sum;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix: Matrix<i32> = Matrix::new(3, 3);
        assert_eq!(matrix.rows(), 3);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix.dimensions(), (3, 3));
    }

    #[test]
    fn test_matrix_from_vec() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ];
        let matrix = Matrix::from_vec(data).unwrap();
        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 3);
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(1, 2)], 6);
    }

    #[test]
    fn test_matrix_indexing() {
        let mut matrix: Matrix<i32> = Matrix::new(2, 2);
        matrix[(0, 0)] = 1;
        matrix[(0, 1)] = 2;
        matrix[(1, 0)] = 3;
        matrix[(1, 1)] = 4;
        
        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
    }

    #[test]
    fn test_matrix_addition() {
        let data1 = vec![
            vec![1, 2],
            vec![3, 4],
        ];
        let data2 = vec![
            vec![5, 6],
            vec![7, 8],
        ];
        
        let matrix1 = Matrix::from_vec(data1).unwrap();
        let matrix2 = Matrix::from_vec(data2).unwrap();
        let result = (matrix1 + matrix2).unwrap();
        
        assert_eq!(result[(0, 0)], 6);
        assert_eq!(result[(0, 1)], 8);
        assert_eq!(result[(1, 0)], 10);
        assert_eq!(result[(1, 1)], 12);
    }

    #[test]
    fn test_matrix_multiplication() {
        let data1 = vec![
            vec![1, 2],
            vec![3, 4],
        ];
        let data2 = vec![
            vec![5, 6],
            vec![7, 8],
        ];
        
        let matrix1 = Matrix::from_vec(data1).unwrap();
        let matrix2 = Matrix::from_vec(data2).unwrap();
        let result = (matrix1 * matrix2).unwrap();
        
        assert_eq!(result[(0, 0)], 19);
        assert_eq!(result[(0, 1)], 22);
        assert_eq!(result[(1, 0)], 43);
        assert_eq!(result[(1, 1)], 50);
    }

    #[test]
    fn test_matrix_transpose() {
        let data = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
        ];
        let matrix = Matrix::from_vec(data).unwrap();
        let transposed = matrix.transpose();
        
        assert_eq!(transposed.rows(), 3);
        assert_eq!(transposed.cols(), 2);
        assert_eq!(transposed[(0, 0)], 1);
        assert_eq!(transposed[(0, 1)], 4);
        assert_eq!(transposed[(1, 0)], 2);
        assert_eq!(transposed[(1, 1)], 5);
    }

    #[test]
    fn test_identity_matrix() {
        let identity: Matrix<i32> = Matrix::identity(3);
        assert_eq!(identity[(0, 0)], 1);
        assert_eq!(identity[(1, 1)], 1);
        assert_eq!(identity[(2, 2)], 1);
        assert_eq!(identity[(0, 1)], 0);
        assert_eq!(identity[(1, 0)], 0);
    }
} 