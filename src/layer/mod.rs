pub mod neuron;

use crate::utils::types::*;
use crate::maths::*;
use crate::layer::neuron::Neuron;
use crate::tools::activations::Activation;
use crate::tools::cost::CostFunction;
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
pub struct LayerObj
{
    pub name: &'static str,
    pub neurons: Vec<Neuron>,
    pub fan_in: Dimension,
    pub fan_out: Dimension,
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

pub fn new_layer_obj<'lng>(fan_in: Dimension, fan_out: Dimension, layer: LAYER, initializer: &Box<dyn WeightInitializer + 'lng>, name: Option<&'static str>) -> LayerObj {
    if !layer.check_size(fan_in) {
        panic!("\n| vector and fan_in length mismatch!\n| at layer '{}', got {} size vector, expect {}\n", name.unwrap(), layer.get_len(), fan_in);
    }
    LayerObj {
        name: name.unwrap(),
        neurons: init_neurons(fan_in, fan_out, &layer, initializer),
        fan_in: fan_in,
        fan_out: fan_out,
        _type: layer
    }
}

pub trait Layer {
    fn feed_forward<'lng>(&self, layers: &mut Vec<LayerObj>, i: usize, cost: &mut Weight, cost_f: &Box<dyn CostFunction + 'lng>, act_f: &Box<dyn Activation + 'lng>);
    fn back_propagation<'lng>(&self, layers: &mut Vec<LayerObj>, i: usize, lr: Weight, cost_f: &Box<dyn CostFunction + 'lng>, act_f: &Box<dyn Activation + 'lng>);
}

pub struct SGD;
impl Layer for SGD {
    #[inline]
    fn feed_forward<'lng>(&self, layers: &mut Vec<LayerObj>, i: usize, cost: &mut Weight, cost_f: &Box<dyn CostFunction + 'lng>, act_f: &Box<dyn Activation + 'lng>) {
        //-------If the layer is last, calculate loss without feed forward-----
        if let LAYER::OUT(_) = layers[i]._type {

            //--clear cost--
            *cost = 0.;
            //--------------

            //--------Sum all local loss----------------------
            for n in 0..layers[i].neurons.len() {
                let loss = cost_f.feed_forward(layers[i].neurons[n].input.reveal(), layers[i].neurons[n].o);
                *cost = *cost + layers[i].neurons[n].set_loss(loss);
            }
            //------------------------------------------------

        //---------------------------------------------------------------------

        //--------Do feed forward----------------------------------------------
        } else {

            //--------Validate current fan_in, and next fan_out----------------
            if layers[i].fan_out != layers[i+1].fan_in {
                panic!("layer dimension mismatch!");
            }
            //-----------------------------------------------------------------

            //-------feed forward, calculate each neruons from next layer------
            for k in 0..layers[i+1].neurons.len() as usize {

                //------Get current layer's sum of each neuron's weights * input-//
                let mut weight_sum: Weight = 0.;
                for j in 0..layers[i].neurons.len() as usize {
                    // get_wx(k) will get weight * input, on current layer, that the neuron at
                    //  index i that heading to next neuron which is indexed at k
                    weight_sum = layers[i].neurons[j].get_wx(k) + weight_sum;
                }
                //-------------------------------------------------------------

                //-------update output, activation(weight_sum + bias)----------
                // get bias of each neurons from next layer, and sum
                let z = layers[i+1].neurons[k].bias + weight_sum;
                // use current layer's activation function
                let o = act_f.feed_forward(z);
                // update next layers output which is same as this neuron's output
                layers[i+1].neurons[k].update_o(z, o);
                //-------------------------------------------------------------
            }
        }
        //---------------------------------------------------------------------

    }
    #[inline]
    fn back_propagation<'lng>(&self, layers: &mut Vec<LayerObj>, i: usize, lr: Weight, cost_f: &Box<dyn CostFunction + 'lng>, act_f: &Box<dyn Activation + 'lng>) {
        //--------Back propagation---------------------------------------------
        //--------When next layer is output------------------------------------
        if let LAYER::OUT(_) = layers[i+1]._type {

            //--------Looping length of next neurons---------------------------
            for n in 0..layers[i+1].neurons.len() {


                // calculate (dE_total / dh)
                let E_total = cost_f.back_propagation(layers[i+1].neurons[n].input.reveal(), layers[i+1].neurons[n].o);

                // calculate (do / dz)
                let o = act_f.back_propagation(layers[i+1].neurons[n].o);

                let E_total__o = E_total * o;

                // update neurons weights of current layer using next layer's Errors
                for w in 0..layers[i].neurons.len() {
                    // calculate (dz / dW)
                    layers[i].neurons[w].local_loss[n] = E_total__o * layers[i].neurons[w].o;

                    // update weight
                    layers[i].neurons[w].weights[n] = layers[i].neurons[w].weights[n] - (lr * layers[i].neurons[w].local_loss[n]);
                }

            }
        } else {
            for n in 0..layers[i].neurons.len() {
                for w in 0..layers[i+1].neurons.len() {

                    // sum sum of errors from next layer
                    let E_total:f64 = layers[i+1].neurons[w].local_loss.iter().sum();

                    // do back propagation
                    let h = act_f.back_propagation(layers[i+1].neurons[w].o);

                    // get local error, dE_total / dWn
                    layers[i].neurons[n].local_loss[w] = E_total * layers[i].neurons[n].o * h;

                    // and update weights, ( Wn - (lr * E_Wn) )
                    layers[i].neurons[n].weights[w] = layers[i].neurons[n].weights[w] - (lr * layers[i].neurons[n].local_loss[w]);
                }
            }
        }
    }
}
