mod utils;

use utils::Matrix;

fn main() {
    println!("Matrix Example");
    
    // Create a 3x3 matrix with default values (0 for integers)
    let mut matrix: Matrix<i32> = Matrix::new(3, 3);
    
    // Set some values
    matrix[(0, 0)] = 1;
    matrix[(0, 1)] = 2;
    matrix[(0, 2)] = 3;
    matrix[(1, 0)] = 4;
    matrix[(1, 1)] = 5;
    matrix[(1, 2)] = 6;
    matrix[(2, 0)] = 7;
    matrix[(2, 1)] = 8;
    matrix[(2, 2)] = 9;
    
    println!("Original matrix:");
    println!("{}", matrix);
    
    // Create a matrix from a vector
    let data = vec![
        vec![1, 2],
        vec![3, 4],
    ];
    let matrix2 = Matrix::from_vec(data).unwrap();
    println!("Matrix from vector:");
    println!("{}", matrix2);
    
    // Transpose the matrix
    let transposed = matrix2.transpose();
    println!("Transposed matrix:");
    println!("{}", transposed);
    
    // Create an identity matrix
    let identity: Matrix<i32> = Matrix::identity(3);
    println!("Identity matrix:");
    println!("{}", identity);
    
    // Matrix addition example
    let data1 = vec![
        vec![1, 2],
        vec![3, 4],
    ];
    let data2 = vec![
        vec![5, 6],
        vec![7, 8],
    ];
    
    let matrix_a = Matrix::from_vec(data1).unwrap();
    let matrix_b = Matrix::from_vec(data2).unwrap();
    
    match matrix_a + matrix_b {
        Ok(result) => {
            println!("Matrix addition result:");
            println!("{}", result);
        }
        Err(e) => println!("Error: {}", e),
    }
}
