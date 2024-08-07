use forward_propagation::forward_propagation;
use ndarray::array;
use ndarray::Array2;
use rand::Rng;

mod forward_propagation;
mod layer_size;
mod parameters_struct;

fn main() {
    let (X, Y) = create_dataset();
    // Length of X = 2, Length of Y = 1.

    let params = initialize_params();
    // println!("{:?}", params.W);
    let A = forward_propagation(X, params);
    println!("{:?}", A);
}

//Create dataset
pub fn create_dataset() -> (Array2<f32>, Array2<f32>) {
    /*
    Returns:
    (X, Y) -- tuple containing parameters:
                    X -- Training dataset with independent values 0 and 1
                    Y -- Training dataset with dependent values 1 and 0
    Creates m = 30 data points (x1, x2) where x1 and x2 are either 0 or 1 and save them in an array X with shape (2 x m).
    The labels (0: blue, 1: red) will be calculated such that y = 1 if x1 = 0 and x2 = 1. The rest of the cases, y = 0. The labels will be saves in array Y with shape (1 x m).
     */

    let m = 30; // Change this to your desired value
    let mut rng = rand::thread_rng();

    // Step 1: Generate a 2xM array of random integers (0 or 1)
    let mut x = Array2::<f32>::zeros((2, m));
    for i in 0..2 {
        for j in 0..m {
            x[(i, j)] = rng.gen_range(0..2) as f32;
        }
    }

    // Step 2: Perform a logical AND operation
    let mut y: Vec<f32> = vec![0.0; m];
    for j in 0..m {
        if x[(0, j)] == 0.0 && x[(1, j)] == 1.0 {
            y[j] = 1.0;
        }
    }

    // Convert y to an ndarray and reshape it to 1xM
    let y = ndarray::Array::from_shape_vec((1, m), y).unwrap();

    // Print the results for verification
    // println!("X: \n{}", x);
    // println!("Y: \n{}", y);
    return (x, y);
}

// struct Parameters {
//     W: Array2<f64>,
//     b: Array2<f64>,
// }

fn initialize_params() -> parameters_struct::Parameters {
    /*
    Returns:
    params -- struct containing parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
     */
    let mut rng = rand::thread_rng();
    let W = array![[rng.gen(), rng.gen()]];
    let b = array![[0.0]];

    let parameters = parameters_struct::Parameters { W: W, b: b };
    parameters
}
