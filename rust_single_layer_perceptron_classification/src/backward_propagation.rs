use ndarray::{Array2, Axis};

use crate::structs;

pub fn backward_propagation(
    A: &Array2<f32>,
    X: &Array2<f32>,
    Y: &Array2<f32>,
) -> structs::Gradients {
    let m = Y.shape()[1];

    let dZ = A - Y;

    let dW = (1.0 / m as f32) * dZ.dot(&X.t());
    let sum_axis1 = dZ.sum_axis(Axis(1));

    let result = sum_axis1.into_shape((1, 1)).unwrap();
    let dB = (1.0 / m as f32) * result;

    let grads = structs::Gradients { dW: dW, dB: dB };
    return grads;
}
