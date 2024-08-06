use ndarray::Array2;

mod create_dataset;
mod layer_size;

fn main() {
    let data = create_dataset::create_dataset();
    println!("{:?}", data.1);
    // let size = layer_sizes(X, Y)
}
