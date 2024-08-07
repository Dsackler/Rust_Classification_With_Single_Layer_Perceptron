use ndarray::Array2;

use crate::structs;

pub fn forward_propagation(X: &Array2<f32>, parameters: &structs::Parameters) -> Array2<f32> {
    /*
        Argument:
        X -- input data of size (n_x, m)
        parameters -- parameters struct containing W and b (output of initialization function)

        Returns:
        A -- The sigmoid function applied to Z (Z is the linear equation WX + b)
    */

    // let W = &parameters.W;
    // let b = &parameters.b;

    // let z = W.dot(X) + b;
    // let A = sigmoid(&z);

    sigmoid(&parameters.W.dot(X) + &parameters.b)
    // return A;
}

fn sigmoid(z: Array2<f32>) -> Array2<f32> {
    return z.mapv(|x| 1.0 / (1.0 + (-x).exp()));
}
