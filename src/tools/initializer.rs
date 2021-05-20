use crate::utils::types::{Dimension, Weight};
use rand::Rng;
// Sigmoid / Tanh

pub trait WeightInitializer {
    fn initializer(&self, fan_in: Dimension, fan_out: Dimension) -> Weight;
}

#[allow(non_camel_case_types)]
pub struct he_initializer;
impl WeightInitializer for he_initializer {
    fn initializer(&self, fan_in: Dimension, fan_out: Dimension) -> Weight {
        let mut rng = rand::thread_rng();
        rng.gen_range((fan_in as f64)..(fan_out as f64)) as f64 / ((fan_in / 2) as f64).sqrt()
    }
}

#[allow(non_camel_case_types)]
pub struct xavier_initializer;
impl WeightInitializer for xavier_initializer {
    fn initializer(&self, fan_in: Dimension, fan_out: Dimension) -> Weight {
        let mut rng = rand::thread_rng();
        rng.gen_range((fan_in as f64)..(fan_out as f64)) as f64 / (fan_in as f64).sqrt()
    }
}
