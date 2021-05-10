use crate::utils::types::{Error, Weight};

pub trait CostFunction {
    fn feed_forward(&self, target: Weight, output: Weight) -> Error;
    fn back_propagation(&self, target: Weight, output: Weight) -> Error;
}

pub struct MSE;
impl CostFunction for MSE {
    fn feed_forward(&self, target: Weight, output: Weight) -> Error {
        (target - output).powi(2) / 2.
    }
    fn back_propagation(&self, target: Weight, output: Weight) -> Error {
        -(target - output)
    }
}
