use backward_propagation::backward_propagation;
use compute_cost::compute_cost;
use forward_propagation::forward_propagation;
use ndarray::array;
use ndarray::Array2;
use rand::Rng;
use structs::Parameters;
use update_parameters::update_parameters;

mod backward_propagation;
mod compute_cost;
mod forward_propagation;
mod layer_size;
mod structs;
mod update_parameters;

fn main() {
    let (X, Y) = create_dataset();
    // Length of X = 2, Length of Y = 1.

    // let params = initialize_params();
    // // println!("{:?}", params.W);
    // let A = forward_propagation(&X, &params);
    // // println!("{:?}", A);

    // let cost = compute_cost::compute_cost(&A, &Y);
    // // println!("Cost = {cost}");

    // let backward_propagation = backward_propagation(&A, &X, &Y);
    // // println!(
    // //     "{:?}, {:?}",
    // //     backward_propagation.dW, backward_propagation.dB
    // // );
    // let update = update_parameters(&params, backward_propagation, 1.2);
    // println!("Original W: {:?}, B: {:?}", params.W, params.b);
    // println!("Updated W: {:?}, B: {:?}", update.W, update.b);
    let result = nn_model(&X, &Y, 50, 1.2, true);
    println!("{:?}, {:?}", result.W, result.b);
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

    return (x, y);
}

fn initialize_params() -> structs::Parameters {
    /*
    Returns:
    params -- struct containing parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
     */
    let mut rng = rand::thread_rng();
    let mut W = array![[rng.gen(), rng.gen()]];
    let mut b = array![[0.0]];

    let parameters = structs::Parameters { W: W, b: b };
    parameters
}

fn nn_model(
    X: &Array2<f32>,
    Y: &Array2<f32>,
    num_iterations: i32,
    learning_rate: f32,
    print_cost: bool,
) -> structs::Parameters {
    let mut parameters = initialize_params();

    for n in 0..num_iterations {
        let A = forward_propagation(X, &parameters);
        let cost = compute_cost(&A, Y);
        let grads = backward_propagation(&A, X, Y);
        parameters = update_parameters(&parameters, grads, learning_rate);

        if print_cost {
            println!("Cost after iteration {n}: {cost}");
        }
    }

    parameters
}
