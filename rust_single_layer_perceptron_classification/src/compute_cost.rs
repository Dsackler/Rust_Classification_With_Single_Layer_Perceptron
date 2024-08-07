use ndarray::Array2;
use num_traits::Float;

pub fn compute_cost(A: Array2<f32>, Y: &Array2<f32>) -> f32 {
    // Number of examples.
    let m = Y.shape()[1] as f32;

    // Compute the cost function.
    let logprobs = -A.mapv(|x| x.ln()) * Y - (A.mapv(|x| (1.0 - x).ln()) * (&(1.0 - Y)));
    let cost = (1.0 / m) * logprobs.sum();
    cost
}
