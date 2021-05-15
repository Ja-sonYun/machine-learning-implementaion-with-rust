pub trait Activation<'lng> {
    fn feed_forward(&self, x: f64) -> f64;
    fn back_propagation(&self,x: f64) -> f64;
}

#[allow(non_camel_case_types)]
pub struct Sigmoid;
impl<'lng> Activation<'lng> for Sigmoid {
    fn feed_forward(&self, x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }
    fn back_propagation(&self, x: f64) -> f64 {
        x * (1. - x)
    }
}

#[allow(non_camel_case_types)]
pub struct ReLU;
impl<'lng> Activation<'lng> for ReLU {
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
impl<'lng> Activation<'lng> for Leaky_ReLU {
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
pub struct A_END;
impl<'lng> Activation<'lng> for A_END {
    fn feed_forward(&self, _: f64) -> f64 {
        panic!("This won't be called normally. something is wrong!")
    }
    fn back_propagation(&self, _: f64) -> f64 {
        panic!("This won't be called normally. something is wrong!")
    }
}
