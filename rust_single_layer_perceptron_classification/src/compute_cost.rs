use ndarray::Array2;

pub fn compute_cost(A: &Array2<f32>, Y: &Array2<f32>) -> f32 {
    /*
    Returns:
    The log loss of the yhat that went through the sigmoid function.
     */

    // Number of examples.
    let m = Y.shape()[1] as f32;

    // Compute the cost function.
    let logprobs = -(A.mapv(f32::ln) * Y + (1.0 - A).mapv(f32::ln) * (1.0 - Y));
    logprobs.sum() / m
    // cost
}
