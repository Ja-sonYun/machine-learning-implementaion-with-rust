use crate::maths::NumOrZeroF64;
use crate::utils::types::*;
//TODO: Remove initializer
use crate::tools::initializer::WeightInitializer;

#[derive(Clone, Debug)]
pub struct Neuron {
    pub input: NumOrZeroF64,
    pub z: Weight,              // summed of weights
    pub o: Weight,              // value that wrapped by activation func, output
    pub weights: Vec<Weight>,   // weights where routing from origins
    pub bias: Bias,             // bias, TODO: toggle this
    pub loss: Error,            // local loss
    pub local_loss: Vec<Weight>,
}

impl Neuron {
    pub fn new<'lng>(x: NumOrZeroF64, bias: Bias, fan_in: Dimension, fan_out: Dimension, initializer: &(dyn WeightInitializer + 'lng)) -> Self {
        Neuron {
            input: x,
            z: 0.,
            o: x.reveal(),
            weights: (0..fan_out).map(|_| initializer.initializer(fan_in, fan_out)).collect(),
            bias,
            loss: 0.,
            local_loss: (0..fan_out).map(|_| 0.).collect(),
        }
    }

    // W * x
    pub fn get_wx(&self, index: usize) -> Weight {
        self.weights[index] * self.o
    }

    pub fn set_loss(&mut self, loss: Weight) -> Error {
        self.loss = loss;
        self.loss
    }

    pub fn update_o(&mut self, z: Weight, o: Weight) {
        self.z = z;
        self.o = o;
    }
}
