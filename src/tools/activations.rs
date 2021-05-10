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
