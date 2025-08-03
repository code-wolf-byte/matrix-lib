use matrix_lib::{Matrix, NDArray};

fn main() {
    println!("=== nDArray Demonstration ===");
    
    // 1. Create different dimensional arrays
    println!("1. Creating arrays of different dimensions:");
    
    // 1D array (vector)
    let vec1d = NDArray::from_vec(vec![1, 2, 3, 4, 5], &[5]).unwrap();
    println!("1D array: {}", vec1d);
    
    // 2D array (matrix-like)
    let array2d = NDArray::from_vec(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    println!("2D array:\n{}", array2d);
    
    // 3D array (tensor)
    let array3d: NDArray<i32> = NDArray::zeros(&[2, 2, 3]);
    println!("3D array shape: {:?}", array3d.shape());
    println!("3D array dimensions: {}", array3d.ndim());
    println!("3D array total size: {}", array3d.size());
    
    // 2. Array creation methods
    println!("\n2. Array creation methods:");
    
    let zeros: NDArray<i32> = NDArray::zeros(&[2, 4]);
    println!("Zeros array:\n{}", zeros);
    
    let ones: NDArray<i32> = NDArray::ones(&[3, 2]);
    println!("Ones array:\n{}", ones);
    
    // 3. Accessing and modifying elements
    println!("\n3. Element access and modification:");
    
    let mut mutable_array = NDArray::from_vec(vec![1, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    println!("Original array:\n{}", mutable_array);
    
    // Access element
    println!("Element at [1, 2]: {:?}", mutable_array.get(&[1, 2]));
    
    // Modify element
    mutable_array.set(&[1, 2], 99).unwrap();
    println!("After modifying [1, 2] to 99:\n{}", mutable_array);
    
    // 4. Reshaping operations
    println!("\n4. Reshaping operations:");
    
    let original = NDArray::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], &[2, 4]).unwrap();
    println!("Original [2, 4] array:\n{}", original);
    
    let reshaped = original.reshape(&[4, 2]).unwrap();
    println!("Reshaped to [4, 2]:\n{}", reshaped);
    
    let flattened = original.flatten();
    println!("Flattened to 1D: {}", flattened);
    
    // 5. Interoperability with Matrix
    println!("\n5. Matrix-NDArray interoperability:");
    
    // Create a matrix
    let matrix = Matrix::from_vec(vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ]).unwrap();
    println!("Original Matrix:\n{}", matrix);
    
    // Convert to NDArray
    let nd_from_matrix = NDArray::from(matrix);
    println!("Matrix as NDArray:\n{}", nd_from_matrix);
    
    // Convert back to Matrix
    let matrix_back: Matrix<i32> = nd_from_matrix.try_into().unwrap();
    println!("Back to Matrix:\n{}", matrix_back);
    
    // 6. Working with higher dimensions
    println!("\n6. Higher dimensional operations:");
    
    let tensor4d: NDArray<f64> = NDArray::zeros(&[2, 3, 4, 5]);
    println!("4D tensor shape: {:?}", tensor4d.shape());
    println!("4D tensor dimensions: {}", tensor4d.ndim());
    println!("4D tensor total elements: {}", tensor4d.size());
    
    // Reshape to different dimensions
    let reshaped_4d = tensor4d.reshape(&[6, 10, 2]).unwrap();
    println!("4D tensor reshaped to [6, 10, 2]: {:?}", reshaped_4d.shape());
    
    // 7. Error handling demonstration
    println!("\n7. Error handling:");
    
    // Try to create array with mismatched data and shape
    let result = NDArray::from_vec(vec![1, 2, 3], &[2, 2]);
    match result {
        Ok(_) => println!("Array created successfully"),
        Err(e) => println!("Error creating array: {:?}", e),
    }
    
    // Try to convert 3D array to Matrix
    let array3d_sample = NDArray::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8], &[2, 2, 2]).unwrap();
    let matrix_result: Result<Matrix<i32>, _> = array3d_sample.try_into();
    match matrix_result {
        Ok(_) => println!("Conversion successful"),
        Err(e) => println!("Error converting 3D array to Matrix: {:?}", e),
    }
    
    println!("\n=== Demo Complete ===");
} 