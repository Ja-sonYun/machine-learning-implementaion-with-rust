#[macro_use]
pub mod macros;

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
    pub layers: Vec<LayerObj>,
    pub layers_actions: Vec<(Box<dyn Layer + 'lng>, Box<dyn Activation + 'lng>)>,
    pub layer_size: usize,
    pub lr: f64,                // learning rate
    pub cost: Error,            // total loss
    pub cost_function: Box<dyn CostFunction + 'lng>,
    pub w_initializer: Box<dyn WeightInitializer + 'lng>,
}

impl<'lng> Model<'lng> {
    pub fn new(name: &'static str, cost_function: impl CostFunction + 'lng, w_initializer: impl WeightInitializer + 'lng) -> Self {
        Model { name: name, layers: Vec::new(), layers_actions: Vec::new(), layer_size: 0, lr: 0., cost: 0., cost_function: Box::new(cost_function), w_initializer: Box::new(w_initializer) }
    }

    fn push_layer(&mut self, layer_obj: LayerObj, layer_action: impl Layer + 'lng, activation: impl Activation + 'lng) {
        self.layers_actions.push((Box::new(layer_action), Box::new(activation)));
        self.layers.push(layer_obj);
        self.layer_size = self.layer_size + 1;
    }

    pub fn add_layer(&mut self, layer_t: impl Layer + 'lng, fan_in: Dimension, fan_out: Dimension, layer: LAYER, activation: impl Activation + 'lng, name: Option<&'static str>) {
        self.push_layer(new_layer_obj(fan_in, fan_out, layer, &self.w_initializer, name), layer_t, activation);
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
            println!(",-----{}-----, fan_in:{}, fan_out:{}, type:{:?}", layer.name, layer.fan_in, layer.fan_out, layer._type);
            println!("| -({} Neurons) => {} out", layer.neurons.len(), layer.fan_out);
            for neuron in &layer.neurons {
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
            self.layers_actions[i].0.feed_forward(&mut self.layers, i, &mut self.cost, &self.cost_function, &self.layers_actions[i].1);
        }
    }

    fn back_propagation(&mut self) {
        for i in (0..(self.layer_size-1)).rev() {
            self.layers_actions[i].0.back_propagation(&mut self.layers, i, self.lr, &self.cost_function, &self.layers_actions[i].1);
        }
    }

}
