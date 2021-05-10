use crate::*;
use tools::*;
use utils::types::*;
use activations::*;
use initializer::WeightInitializer;
use cost::*;
use layer::*;

pub struct Model<'lng>
{
    pub name: &'static str,
    pub layers: Vec<(Box<dyn Layer + 'lng>, LayerObj<'lng>)>,
    pub layer_size: usize,
    pub lr: f64,                // learning rate
    pub cost: Error,            // total loss
    pub cost_function: Box<dyn CostFunction + 'lng>,
    pub w_initializer: Box<dyn WeightInitializer + 'lng>,
}

impl<'lng> Model<'lng> {
    pub fn new(name: &'static str, cost_function: impl CostFunction + 'lng, w_initializer: impl WeightInitializer + 'lng) -> Self {
        Model { name: name, layers: Vec::new(), layer_size: 0, lr: 0., cost: 0., cost_function: Box::new(cost_function), w_initializer: Box::new(w_initializer) }
    }

    fn push_layer(&mut self, layer: impl Layer + 'lng, layer_obj: LayerObj<'lng>) {
        self.layers.push((Box::new(layer), layer_obj));
        self.layer_size = self.layer_size + 1;
    }

    pub fn add_layer(&mut self, layer_t: impl Layer + 'lng, fan_in: Dimension, fan_out: Dimension, layer: LAYER, activation: impl Activation + 'lng, name: Option<&'static str>) {
        self.push_layer(layer_t, new_layer_obj(fan_in, fan_out, layer, activation, &self.w_initializer, name));
    }

    // param: epoch, learning rate, log, log interval
    pub fn train(&mut self, epoch: usize, lr: f64, log: bool, log_interval: usize) {
        self.lr = lr;
        for e in 0..epoch {
            self.forward_propagation();
            self.back_propagation();
            if log && e % log_interval == 0 {
                println!("{:0} epoch, loss: {}", e, self.cost);
            }
        }
    }

    pub fn debug(&self) {
        println!("=================================");
        println!("| model {} -> {} layers |", self.name, self.layer_size);
        println!("=================================");
        for layer in &self.layers {
            println!(",-----{}-----, fan_in:{}, fan_out:{}, type:{:?}", layer.1.name, layer.1.fan_in, layer.1.fan_out, layer.1._type);
            println!("| -({} Neurons) => {} out", layer.1.neurons.len(), layer.1.fan_out);
            for neuron in &layer.1.neurons {
                println!("|  |-{:?}", neuron);
            }
            println!("`---------------------------");
        }
        println!("=================================");
        println!(" * Total cost: {}", self.cost);
    }

    fn forward_propagation(&mut self) {
        // loop layers
        for i in 0..(self.layer_size) {

            //-------If the layer is last, calculate loss without feed forward-----
            if let LAYER::OUT(_) = self.layers[i].1._type {

                //--clear cost--
                self.cost = 0.;
                //--------------

                //--------Sum all local loss----------------------
                for n in 0..self.layers[i].1.neurons.len() {
                    let loss = self.cost_function.feed_forward(self.layers[i].1.neurons[n].input.reveal(), self.layers[i].1.neurons[n].o);
                    self.cost = self.cost + self.layers[i].1.neurons[n].set_loss(loss);
                }
                //------------------------------------------------

            //---------------------------------------------------------------------

            //--------Do feed forward----------------------------------------------
            } else {

                //--------Validate current fan_in, and next fan_out----------------
                if self.layers[i].1.fan_out != self.layers[i+1].1.fan_in {
                    panic!("layer dimension mismatch!");
                }
                //-----------------------------------------------------------------

                //-------feed forward, calculate each neruons from next layer------
                for k in 0..self.layers[i+1].1.neurons.len() as usize {

                    //------Get current layer's sum of each neuron's weights * input-//
                    let mut weight_sum: Weight = 0.;
                    for j in 0..self.layers[i].1.neurons.len() as usize {
                        // get_wx(k) will get weight * input, on current layer, that the neuron at
                        //  index i that heading to next neuron which is indexed at k
                        weight_sum = self.layers[i].1.neurons[j].get_wx(k) + weight_sum;
                    }
                    //-------------------------------------------------------------

                    //-------update output, activation(weight_sum + bias)----------
                    // get bias of each neurons from next layer, and sum
                    let z = self.layers[i+1].1.neurons[k].bias + weight_sum;
                    // use current layer's activation function
                    let o = self.layers[i].1.activation.feed_forward(z);
                    // update next layers output which is same as this neuron's output
                    self.layers[i+1].1.neurons[k].update_o(z, o);
                    //-------------------------------------------------------------
                }
            }
            //---------------------------------------------------------------------
        }
    }

    fn back_propagation(&mut self) {
        for l in (0..(self.layer_size-1)).rev() {
            //--------Back propagation---------------------------------------------
            //--------When next layer is output------------------------------------
            if let LAYER::OUT(_) = self.layers[l+1].1._type {

                //--------Looping length of next neurons---------------------------
                for n in 0..self.layers[l+1].1.neurons.len() {


                    // calculate (dE_total / dh)
                    let E_total = self.cost_function.back_propagation(self.layers[l+1].1.neurons[n].input.reveal(), self.layers[l+1].1.neurons[n].o);

                    // calculate (do / dz)
                    let o = self.layers[l].1.activation.back_propagation(self.layers[l+1].1.neurons[n].o);

                    let E_total__o = E_total * o;

                    // update neurons weights of current layer using next layer's Errors
                    for w in 0..self.layers[l].1.neurons.len() {
                        // calculate (dz / dW)
                        self.layers[l].1.neurons[w].local_loss[n] = E_total__o * self.layers[l].1.neurons[w].o;

                        // update weight
                        self.layers[l].1.neurons[w].weights[n] = self.layers[l].1.neurons[w].weights[n] - (self.lr * self.layers[l].1.neurons[w].local_loss[n]);
                    }

                }
            } else {
                for n in 0..self.layers[l].1.neurons.len() {
                    for w in 0..self.layers[l+1].1.neurons.len() {

                        // sum sum of errors from next layer
                        let E_total:f64 = self.layers[l+1].1.neurons[w].local_loss.iter().sum();

                        // do back propagation
                        let h = self.layers[l].1.activation.back_propagation(self.layers[l+1].1.neurons[w].o);

                        // get local error, dE_total / dWn
                        self.layers[l].1.neurons[n].local_loss[w] = E_total * self.layers[l].1.neurons[n].o * h;

                        // and update weights, ( Wn - (lr * E_Wn) )
                        self.layers[l].1.neurons[n].weights[w] = self.layers[l].1.neurons[n].weights[w] - (self.lr * self.layers[l].1.neurons[n].local_loss[w]);
                    }
                }
            }
        }
    }

}
