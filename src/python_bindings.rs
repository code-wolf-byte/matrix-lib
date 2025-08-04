use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::PyList;

use crate::utils::matrix::Matrix;
use crate::utils::ndarray::NDArray;

/// Python wrapper for Matrix
#[pyclass]
pub struct PyMatrix {
    inner: Matrix<f64>,
}

#[pymethods]
impl PyMatrix {
    #[new]
    fn new(rows: usize, cols: usize) -> Self {
        PyMatrix {
            inner: Matrix::new(rows, cols),
        }
    }

    #[staticmethod]
    fn from_list(data: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut matrix_data = Vec::new();
        
        for row in data.iter() {
            let row_list: &Bound<'_, PyList> = row.downcast()?;
            let mut row_vec = Vec::new();
            
            for item in row_list.iter() {
                let value: f64 = item.extract()?;
                row_vec.push(value);
            }
            matrix_data.push(row_vec);
        }
        
        let matrix = Matrix::from_vec(matrix_data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        
        Ok(PyMatrix { inner: matrix })
    }

    fn rows(&self) -> usize {
        self.inner.rows()
    }

    fn cols(&self) -> usize {
        self.inner.cols()
    }

    fn dimensions(&self) -> (usize, usize) {
        self.inner.dimensions()
    }

    fn get(&self, row: usize, col: usize) -> PyResult<f64> {
        self.inner.get(row, col)
            .copied()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"))
    }

    fn set(&mut self, row: usize, col: usize, value: f64) -> PyResult<()> {
        self.inner.set(row, col, value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    fn transpose(&self) -> PyMatrix {
        PyMatrix {
            inner: self.inner.transpose(),
        }
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = PyList::empty(py);
        
        for row in 0..self.inner.rows() {
            let row_list = PyList::empty(py);
            for col in 0..self.inner.cols() {
                row_list.append(self.inner.get(row, col).unwrap())?;
            }
            result.append(row_list)?;
        }
        
        Ok(result.into())
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Matrix({}x{})", self.inner.rows(), self.inner.cols()))
    }
}

impl PyMatrix {
    fn get_inner(&self) -> &Matrix<f64> {
        &self.inner
    }
}

/// Python wrapper for NDArray
#[pyclass]
pub struct PyNDArray {
    inner: NDArray<f64>,
}

#[pymethods]
impl PyNDArray {
    #[new]
    fn new(shape: Vec<usize>) -> Self {
        PyNDArray {
            inner: NDArray::new(&shape),
        }
    }

    #[staticmethod]
    fn from_list(data: &Bound<'_, PyList>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let mut flat_data = Vec::new();
        
        fn flatten_list(list: &Bound<'_, PyList>, result: &mut Vec<f64>) -> PyResult<()> {
            for item in list.iter() {
                if let Ok(sublist) = item.downcast::<PyList>() {
                    flatten_list(&sublist, result)?;
                } else {
                    let value: f64 = item.extract()?;
                    result.push(value);
                }
            }
            Ok(())
        }
        
        flatten_list(data, &mut flat_data)?;
        
        let final_shape = shape.unwrap_or_else(|| vec![flat_data.len()]);
        
        let array = NDArray::from_vec(flat_data, &final_shape)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
        
        Ok(PyNDArray { inner: array })
    }

    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        PyNDArray {
            inner: NDArray::zeros(&shape),
        }
    }

    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        PyNDArray {
            inner: NDArray::ones(&shape),
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn size(&self) -> usize {
        self.inner.size()
    }

    fn get(&self, indices: Vec<usize>) -> PyResult<f64> {
        self.inner.get(&indices)
            .copied()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"))
    }

    fn set(&mut self, indices: Vec<usize>, value: f64) -> PyResult<()> {
        self.inner.set(&indices, value)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))
    }

    fn reshape(&self, new_shape: Vec<usize>) -> PyResult<PyNDArray> {
        let reshaped = self.inner.reshape(&new_shape)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
        
        Ok(PyNDArray { inner: reshaped })
    }

    fn flatten(&self) -> PyNDArray {
        PyNDArray {
            inner: self.inner.flatten(),
        }
    }

    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner.ndim() {
            1 => {
                let result = PyList::empty(py);
                for i in 0..self.inner.size() {
                    result.append(self.inner.get(&[i]).unwrap())?;
                }
                Ok(result.into())
            },
            2 => {
                let rows = self.inner.shape()[0];
                let cols = self.inner.shape()[1];
                let result = PyList::empty(py);
                
                for r in 0..rows {
                    let row_list = PyList::empty(py);
                    for c in 0..cols {
                        row_list.append(self.inner.get(&[r, c]).unwrap())?;
                    }
                    result.append(row_list)?;
                }
                Ok(result.into())
            },
            _ => {
                // For higher dimensions, return a simple representation
                let result = PyList::empty(py);
                for i in 0..self.inner.size() {
                    result.append(self.inner.get(&[i]).unwrap())?;
                }
                Ok(result.into())
            }
        }
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("NDArray(shape={:?})", self.inner.shape()))
    }
}

/// Matrix addition
#[pyfunction]
fn matrix_add(a: &PyMatrix, b: &PyMatrix) -> PyResult<PyMatrix> {
    let result = (a.get_inner().clone() + b.get_inner().clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    Ok(PyMatrix { inner: result })
}

/// Matrix multiplication
#[pyfunction]
fn matrix_mul(a: &PyMatrix, b: &PyMatrix) -> PyResult<PyMatrix> {
    let result = (a.get_inner().clone() * b.get_inner().clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
    
    Ok(PyMatrix { inner: result })
}

/// Create identity matrix
#[pyfunction]
fn identity_matrix(size: usize) -> PyMatrix {
    PyMatrix {
        inner: Matrix::identity(size),
    }
}

/// Convert NDArray to Matrix (for 2D arrays)
#[pyfunction]
fn ndarray_to_matrix(array: &PyNDArray) -> PyResult<PyMatrix> {
    let matrix: Matrix<f64> = array.inner.clone().try_into()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{:?}", e)))?;
    
    Ok(PyMatrix { inner: matrix })
}

/// Convert Matrix to NDArray
#[pyfunction]
fn matrix_to_ndarray(matrix: &PyMatrix) -> PyNDArray {
    PyNDArray {
        inner: matrix.get_inner().clone().into(),
    }
}

/// Python module definition
#[pymodule]
fn matrix_lib_python(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatrix>()?;
    m.add_class::<PyNDArray>()?;
    m.add_function(wrap_pyfunction!(matrix_add, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_mul, m)?)?;
    m.add_function(wrap_pyfunction!(identity_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_to_ndarray, m)?)?;
    
    Ok(())
} 