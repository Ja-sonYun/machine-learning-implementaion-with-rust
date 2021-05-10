use crate::utils::types::Error;

pub fn MSE(target: f64, output: f64) -> Error {
    (target - output).powi(2) / 2.
}
pub fn d_MSE_v(vec: &Vec<f64>) -> Error {
    (vec[0]).powi(2) / 2.
}
pub fn d_MSE(target: f64, output: f64) -> Error {
    -(target - output)
}
