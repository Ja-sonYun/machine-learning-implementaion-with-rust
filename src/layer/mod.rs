pub mod neuron;

use crate::utils::types::*;
use crate::maths::*;
use crate::layer::neuron::Neuron;
use crate::tools::activations::Activation;
use crate::tools::initializer::WeightInitializer;

#[derive(Debug)]
pub enum LAYER {
    IN(Vec<Weight>),
    OUT(Vec<Weight>),
    HIDDEN,
}
impl LAYER {
    pub fn unwrap(self) -> Option<Vec<Weight>> {
        match self {
            LAYER::IN(vw) | LAYER::OUT(vw) => Some(vw),
            _ => None
        }
    }
    pub fn unwrap_with_index(&self, i: usize) -> NumOrZeroF64 {
        match &self {
            LAYER::IN(vw) | LAYER::OUT(vw) => {
                NumOrZeroF64::Num(vw[i])
            },
            _ => NumOrZeroF64::Zero
        }
    }
    pub fn get_dim(&self, fan_in: Dimension, fan_out: Dimension) -> Dimension {
        match &self {
            LAYER::IN(_) | LAYER::HIDDEN => fan_in,
            LAYER::OUT(_) => fan_out
        }
    }
    pub fn get_len(&self) -> usize {
        match &self {
            LAYER::IN(val) | LAYER::OUT(val) => val.len(),
            _ => 0
        }
    }
    pub fn check_size(&self, dim: Dimension) -> bool {
        match &self {
            LAYER::IN(val) | LAYER::OUT(val) => (val.len() == dim as usize),
            _ => true
        }
    }
}
pub struct LayerObj<'lng>
{
    pub name: &'static str,
    pub neurons: Vec<Neuron>,
    pub fan_in: Dimension,
    pub fan_out: Dimension,
    pub activation: Box<dyn Activation + 'lng>,
    pub _type: LAYER
}

fn init_neurons<'lng>(fan_in: Dimension, fan_out: Dimension, layer: &LAYER, initializer: &Box<dyn WeightInitializer + 'lng>) -> Vec<Neuron> {
    let mut neurons: Vec<Neuron>;
    // if this layer is input or hidden,
    neurons = Vec::with_capacity(fan_in as usize);
    for i in 0..fan_in as usize {
        neurons.push(Neuron::new(layer.unwrap_with_index(i), 0., fan_in, fan_out, initializer));
    }
    // if this layer is output, which is doens't have nuerons
    neurons
}

pub fn new_layer_obj<'lng>(fan_in: Dimension, fan_out: Dimension, layer: LAYER, activation: impl Activation + 'lng, initializer: &Box<dyn WeightInitializer + 'lng>, name: Option<&'static str>) -> LayerObj<'lng> {
    if !layer.check_size(fan_in) {
        panic!("\n| vector and fan_in length mismatch!\n| at layer '{}', got {} size vector, expect {}\n", name.unwrap(), layer.get_len(), fan_in);
    }
    LayerObj {
        name: name.unwrap(),
        neurons: init_neurons(fan_in, fan_out, &layer, initializer),
        fan_in: fan_in,
        fan_out: fan_out,
        activation: Box::new(activation),
        _type: layer
    }
}

pub trait Layer {
}

pub struct DummyLayer;
impl Layer for DummyLayer {
}
