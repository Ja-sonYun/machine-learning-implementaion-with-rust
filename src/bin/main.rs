extern crate ml;
use ml::*;
use tools::*;

pub mod layer {
    use ml::*;

    use utils::types::*;
    use maths::*;

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
}

use utils::types::*;
use activations::*;
use initializer::*;
use cost::*;
use maths::*;
use maths::matrix::*;
use layer::LAYER::*;
use layer::*;

#[derive(Clone, Debug)]
struct Neuron {
    input: NumOrZeroF64,
    z: Weight,              // summed of weights
    o: Weight,              // value that wrapped by activation func, output
    weights: Vec<Weight>,   // weights where routing from origins
    bias: Bias,             // bias, TODO: toggle this
    loss: Error,            // local loss
    local_loss: Vec<Weight>,
}

impl Neuron {
    fn new(x: NumOrZeroF64, bias: Bias, fan_in: Dimension, fan_out: Dimension) -> Self {
        Neuron {
            input: x,
            z: 0.,
            o: x.reveal(),
            weights: (0..fan_out).map(|_| he_initializer(fan_in, fan_out)).collect(),
            bias: bias,
            loss: 0.,
            local_loss: (0..fan_out).map(|_| 0.).collect(),
        }
    }

    // W * x
    fn get_wx(&self, index: usize) -> Weight {
        self.weights[index] * self.o
    }

    fn get_loss(&mut self) -> Error {
        self.loss = MSE(self.input.reveal(), self.o);
        self.loss
    }

    fn update_o(&mut self, z: Weight, o: Weight) {
        self.z = z;
        self.o = o;
    }
}

struct Model<'lng>
{
    name: &'static str,
    layers: Vec<Layer<'lng>>,
    layer_size: usize,
    lr: f64,                // learning rate
    cost: Error,            // total loss
}

impl<'lng> Model<'lng> {
    fn new(name: &'static str, lr: f64) -> Self {
        Model { name: name, layers: Vec::new(), layer_size: 0, lr: lr, cost: 0. }
    }
    fn push_layer(&mut self, layer: Layer<'lng>) {
        self.layers.push(layer);
        self.layer_size = self.layer_size + 1;
    }
    fn add_layer(&mut self, fan_in: Dimension, fan_out: Dimension, layer: LAYER, activation: impl Activation + 'lng, name: Option<&'static str>) {
        self.push_layer(Layer::new(fan_in, fan_out, layer, activation, name));
    }
    fn debug(&self) {
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
            // loop for get total loss and update local losses if its output layer
            if let LAYER::OUT(_) = self.layers[i]._type {
                self.cost = 0.;
                for neuron in &mut self.layers[i].neurons {
                    self.cost = self.cost + neuron.get_loss();
                }
            } else {
                if self.layers[i].fan_out != self.layers[i+1].fan_in {
                    panic!("layer dimension mismatch!");
                }
                // loop each neurons
                for k in 0..self.layers[i+1].neurons.len() as usize {
                    let mut weight_sum: Weight = 0.;
                    // loop for get sum of neuron's weights
                    for j in 0..self.layers[i].neurons.len() as usize {
                        weight_sum = self.layers[i].neurons[j].get_wx(k) + weight_sum;
                    }
                    let z = self.layers[i+1].neurons[k].bias + weight_sum;
                    let o = self.layers[i].activation.feed_forward(z);
                    self.layers[i+1].neurons[k].update_o(z, o);
                }
            }
        }
    }
    fn back_propagation(&mut self) {
        for l in (0..(self.layer_size-1)).rev() {
            if let LAYER::OUT(_) = self.layers[l+1]._type {
                for n in 0..self.layers[l+1].neurons.len() {
                    // calculate (dE_total / dh)
                    // let mut dv_E_total: Vec<f64> = vec![self.layers[l+1].neurons[n].input.reveal() - self.layers[l+1].neurons[n].o];
                    // let E_total = numerical_derivative(d_MSE, &mut dv_E_total)[0];
                    let E_total = d_MSE(self.layers[l+1].neurons[n].input.reveal(), self.layers[l+1].neurons[n].o);

                    // calculate (do / dz)
                    // let mut dv_o: Vec<f64> = vec![self.layers[l+1].neurons[n].o];
                    // let o = numerical_derivative(d_sigmoid, &mut dv_o)[0];
                    let o = self.layers[l].activation.back_propagation(self.layers[l+1].neurons[n].o);
                    let E_total__o = E_total * o;

                    for w in 0..self.layers[l].neurons.len() {
                        // calculate (dz / dW)
                        // let mut dv_z: Vec<f64> = vec![];
                        // (dE / dh)
                        self.layers[l].neurons[w].local_loss[n] = E_total__o * self.layers[l].neurons[w].o;

                        // update weight
                        self.layers[l].neurons[w].weights[n] = self.layers[l].neurons[w].weights[n] - (self.lr * self.layers[l].neurons[w].local_loss[n]);
                    }
                }
            } else {
                for n in 0..self.layers[l].neurons.len() {
                    for w in 0..self.layers[l+1].neurons.len() {
                        // self.layers[l].neruons[n].weights[w];
                        let dE_total_W:f64 = self.layers[l+1].neurons[w].local_loss.iter().sum();

                        // let mut dh_o: Vec<f64> = vec![self.layers[l+1].neurons[w].o];
                        // let h = numerical_derivative(d_sigmoid, &mut dh_o)[0];
                        let h = self.layers[l].activation.back_propagation(self.layers[l+1].neurons[w].o);
                        self.layers[l].neurons[n].local_loss[w] = dE_total_W * self.layers[l].neurons[n].o * h;

                        self.layers[l].neurons[n].weights[w] = self.layers[l].neurons[n].weights[w] - (self.lr * self.layers[l].neurons[n].local_loss[w]);
                    }
                }
            }
        }
    }

}

// trait LayerList {
// }
// impl <ACT: Activation> LayerList for Layer<ACT> {
// }

struct Layer<'lng>
{
    name: &'static str,
    neurons: Vec<Neuron>,
    fan_in: Dimension,
    fan_out: Dimension,
    activation: Box<dyn Activation + 'lng>,
    _type: LAYER
}

impl<'lng> Layer<'lng> {
    fn new(fan_in: Dimension, fan_out: Dimension, layer: LAYER, activation: impl Activation + 'lng, name: Option<&'static str>) -> Self {
        if !layer.check_size(fan_in) {
            panic!("\n| vector and fan_in length mismatch!\n| at layer '{}', got {} size vector, expect {}\n", name.unwrap(), layer.get_len(), fan_in);
        }
        Layer {
            name: name.unwrap(),
            neurons: Layer::init_neurons(fan_in, fan_out, &layer),
            fan_in: fan_in,
            fan_out: fan_out,
            activation: Box::new(activation),
            _type: layer
        }
    }
    fn init_neurons(fan_in: Dimension, fan_out: Dimension, layer: &LAYER) -> Vec<Neuron> {
        let mut neurons: Vec<Neuron>;
        // if this layer is input or hidden,
        neurons = Vec::with_capacity(fan_in as usize);
        for i in 0..fan_in as usize {
            neurons.push(Neuron::new(layer.unwrap_with_index(i), 0., fan_in, fan_out));
        }
        // if this layer is output, which is doens't have nuerons
        neurons
    }
}

fn main()
{
    let input = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    let output = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    let mut new_model = Model::new("test", 0.1);
    new_model.add_layer(8, 9, IN(input), Sigmoid, Some("input"));
    new_model.add_layer(9, 4, HIDDEN, Sigmoid, Some("input"));
    new_model.add_layer(4, 3, HIDDEN, Sigmoid, Some("input"));
    new_model.add_layer(3, 8, HIDDEN, Sigmoid, Some("input"));
    new_model.add_layer(8, 0, OUT(output), END, Some("output"));

    println!("{}", new_model.layers[0].activation.feed_forward(3.));
    for i in (0..500) {
        // if i % 10 == 0 && 0 != i {
            println!("{:0} epoch, loss: {}", i, new_model.cost);
        // }
        let bp = new_model.cost;
        new_model.forward_propagation();
        // if bp < new_model.cost && bp != 0. {
        //     println!("over");
        //     break
        // };
        new_model.back_propagation();
    }
    let mut w1 = Matrix::<f64>::new(3, 4);
    let mut w = Matrix::<f64>::new(3, 4);
    let mut a = Matrix::<f64>::new_scalar(4.);
    let mut b = Matrix::<f64>::new_scalar(2.);
    w1.set(1, 3, 1.);
    w.set(1, 3, 4.);
    println!("{}", w);
    println!("{}", w1);
    println!("{}", w * w1);
}
