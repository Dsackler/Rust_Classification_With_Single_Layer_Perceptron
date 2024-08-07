use ndarray::Array2;
use num_traits::Float;

use crate::structs;

pub fn update_parameters(
    parameters: &structs::Parameters,
    grads: structs::Gradients,
    learning_rate: f32,
) -> structs::Parameters {
    let dub = &parameters.W;
    let be = &parameters.b;

    let dW = grads.dW;
    let db = grads.dB;

    let W = dub - learning_rate * dW;
    let b = be - learning_rate * db;

    let parameters = structs::Parameters { W: W, b: b };
    return parameters;
}
