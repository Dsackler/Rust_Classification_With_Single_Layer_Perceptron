use ndarray::array;
use ndarray::Array2;
use rand::Rng;

mod create_dataset;
// mod initialize_params;
mod layer_size;

fn main() {
    let (X, Y) = create_dataset::create_dataset();
    println!("{:?}", X.len());
    // Length of X = 2, Length of Y = 1.

    let params = initialize_params();
    println!("{:?}", params.W);
}

struct Parameters {
    W: Array2<f64>,
    b: Array2<f64>,
}

fn initialize_params() -> Parameters {
    /*
    Returns:
    params -- struct containing parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
     */
    let mut rng = rand::thread_rng();
    let W = array![[rng.gen(), rng.gen()]];
    let b = array![[0.0]];

    let parameters = Parameters { W: W, b: b };
    parameters
}
