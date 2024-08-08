use ndarray::{Array2, Axis};

use crate::structs;

pub fn backward_propagation(
    A: &Array2<f32>,
    X: &Array2<f32>,
    Y: &Array2<f32>,
) -> structs::Gradients {
    /*
    Returns:
    Gradient -- struct containing a gradient with the partial derivaties dW and dB.
     */

    let m = Y.shape()[1] as f32;

    let dZ = A - Y;
    // Partial derivative with respect to W
    let dW = (1.0 / m) * dZ.dot(&X.t());

    // Partial derivative with respect to B
    let dB = (1.0 / m) * dZ.sum_axis(Axis(1)).into_shape((1, 1)).unwrap();

    structs::Gradients { dW: dW, dB: dB }
}
