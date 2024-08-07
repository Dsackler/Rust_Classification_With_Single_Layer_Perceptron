use ndarray::Array2;

pub struct Parameters {
    pub W: Array2<f32>,
    pub b: Array2<f32>,
}

pub struct Gradients {
    pub dW: Array2<f32>,
    pub dB: Array2<f32>,
}
