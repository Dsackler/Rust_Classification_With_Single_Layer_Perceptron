use crate::structs;

pub fn update_parameters(
    parameters: &structs::Parameters,
    grads: structs::Gradients,
    learning_rate: f32,
) -> structs::Parameters {
    /*
    Returns:
    Parameters -- struct containing updated parameters
        --The parameters are updated by performing gradient descent. (Taking the original value and subtracting the respective partial derivative times the learning rate)
     */

    // let dub = &parameters.W;
    // let be = &parameters.b;

    // let dW = grads.dW;
    // let db = grads.dB;

    // let W = dub - learning_rate * dW;
    // let b = be - learning_rate * db;

    // let parameters = structs::Parameters { W: W, b: b };

    structs::Parameters {
        W: &parameters.W - &(grads.dW * learning_rate),
        b: &parameters.b - &(grads.dB * learning_rate),
    }
    // return parameters;
}
