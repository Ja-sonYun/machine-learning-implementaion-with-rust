pub mod initializer {
    use crate::utils::types::{Dimension, Weight};
    use crate::maths::rand::*;
    // Sigmoid / Tanh
    pub fn xavier_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight {
        // rand()
            1.
    }

    // ReLU
    pub fn he_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight {
        rand()
    }
}
pub mod activations {
    pub trait Activation {
        fn feed_forward(&self, x: f64) -> f64;
        fn back_propagation(&self,x: f64) -> f64;
    }

    #[allow(non_camel_case_types)]
    pub struct Sigmoid;
    impl Activation for Sigmoid {
        fn feed_forward(&self, x: f64) -> f64 {
            1. / (1. + (-x).exp())
        }
        fn back_propagation(&self, x: f64) -> f64 {
            x * (1. - x)
        }
    }

    #[allow(non_camel_case_types)]
    pub struct ReLU;
    impl Activation for ReLU {
        fn feed_forward(&self, x: f64) -> f64 {
            if 0. < x {
                x
            } else {
                0.
            }
        }
        fn back_propagation(&self, x: f64) -> f64 {
            if 0. < x {
                1.
            } else {
                0.
            }
        }
    }

    #[allow(non_camel_case_types)]
    pub struct Leaky_ReLU;
    impl Activation for Leaky_ReLU {
        fn feed_forward(&self, x: f64) -> f64 {
            if 0. < x {
                x
            } else {
                x * 0.001
            }
        }
        fn back_propagation(&self, x: f64) -> f64 {
            if 0. < x {
                1.
            } else {
                0.001
            }
        }
    }

    #[allow(non_camel_case_types)]
    pub struct END;
    impl Activation for END {
        fn feed_forward(&self, _: f64) -> f64 {
            panic!("This won't be called normally. something is wrong!")
        }
        fn back_propagation(&self, _: f64) -> f64 {
            panic!("This won't be called normally. something is wrong!")
        }
    }
}

pub mod cost {
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
}
