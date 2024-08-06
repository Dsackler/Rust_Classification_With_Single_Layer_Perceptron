use ndarray::Array2;
use rand::Rng;

pub fn create_dataset() -> (Array2<u8>, Array2<u8>) {
    let m = 30; // Change this to your desired value
    let mut rng = rand::thread_rng();

    // Step 1: Generate a 2xM array of random integers (0 or 1)
    let mut x = Array2::<u8>::zeros((2, m));
    for i in 0..2 {
        for j in 0..m {
            x[(i, j)] = rng.gen_range(0..2);
        }
    }

    // Step 2: Perform a logical AND operation
    let mut y = vec![0; m];
    for j in 0..m {
        if x[(0, j)] == 0 && x[(1, j)] == 1 {
            y[j] = 1;
        }
    }

    // Convert y to an ndarray and reshape it to 1xM
    let y = ndarray::Array::from_shape_vec((1, m), y).unwrap();

    // Print the results for verification
    // println!("X: \n{}", x);
    // println!("Y: \n{}", y);
    return (x, y);
}
