use crate::maths::matrix::{Matrix};
use crate::utils::types::Dimension;
use rand::Rng;
pub fn randn(n_input: Dimension, n_output: Dimension) -> Matrix<f64> {
    Matrix::<f64>::from_fn(rand, n_input, n_output)
}
pub fn rand() -> f64 {
    rand::thread_rng().gen::<f64>()
}
pub fn randin(from: f64, to: f64) -> f64 {
    rand::thread_rng().gen_range(from..to)
}
