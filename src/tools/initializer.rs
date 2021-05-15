use crate::utils::types::{Dimension, Weight};
use crate::maths::rand::*;
// Sigmoid / Tanh
pub fn xavier_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight {
    // rand()
        1.
}

pub trait WeightInitializer {
    fn initializer(&self, fan_in: Dimension, fan_out: Dimension) -> Weight;
}

#[allow(non_camel_case_types)]
pub struct he_initializer;
impl WeightInitializer for he_initializer {
    // ReLU
    fn initializer(&self, fan_in: Dimension, fan_out: Dimension) -> Weight {
        rand()
    }
}
