extern crate ml;
use ml::*;
use tools::*;

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
    layers: Vec<LayerObj<'lng>>,
    layer_size: usize,
    lr: f64,                // learning rate
    cost: Error,            // total loss
}

impl<'lng> Model<'lng> {
    fn new(name: &'static str) -> Self {
        Model { name: name, layers: Vec::new(), layer_size: 0, lr: 0., cost: 0. }
    }

    fn push_layer(&mut self, layer: LayerObj<'lng>) {
        self.layers.push(layer);
        self.layer_size = self.layer_size + 1;
    }

    fn add_layer(&mut self, fan_in: Dimension, fan_out: Dimension, layer: LAYER, activation: impl Activation + 'lng, name: Option<&'static str>) {
        self.push_layer(LayerObj::new(fan_in, fan_out, layer, activation, name));
    }

    // param: epoch, learning rate, log, log interval
    fn train(&mut self, epoch: usize, lr: f64, log: bool, log_interval: usize) {
        self.lr = lr;
        for e in 0..epoch {
            self.forward_propagation();
            self.back_propagation();
            if log && e % log_interval == 0 {
                println!("{:0} epoch, loss: {}", e, self.cost);
            }
        }
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

            //-------If the layer is last, calculate loss without feed forward-----
            if let LAYER::OUT(_) = self.layers[i]._type {

                //--clear cost--
                self.cost = 0.;
                //--------------

                //--------Sum all local loss----------------------
                for neuron in &mut self.layers[i].neurons {
                    self.cost = self.cost + neuron.get_loss();
                }
                //------------------------------------------------

            //---------------------------------------------------------------------

            //--------Do feed forward----------------------------------------------
            } else {

                //--------Validate current fan_in, and next fan_out----------------
                if self.layers[i].fan_out != self.layers[i+1].fan_in {
                    panic!("layer dimension mismatch!");
                }
                //-----------------------------------------------------------------

                //-------feed forward, calculate each neruons from next layer------
                for k in 0..self.layers[i+1].neurons.len() as usize {

                    //------Get current layer's sum of each neuron's weights * input-//
                    let mut weight_sum: Weight = 0.;
                    for j in 0..self.layers[i].neurons.len() as usize {
                        // get_wx(k) will get weight * input, on current layer, that the neuron at
                        //  index i that heading to next neuron which is indexed at k
                        weight_sum = self.layers[i].neurons[j].get_wx(k) + weight_sum;
                    }
                    //-------------------------------------------------------------

                    //-------update output, activation(weight_sum + bias)----------
                    // get bias of each neurons from next layer, and sum
                    let z = self.layers[i+1].neurons[k].bias + weight_sum;
                    // use current layer's activation function
                    let o = self.layers[i].activation.feed_forward(z);
                    // update next layers output which is same as this neuron's output
                    self.layers[i+1].neurons[k].update_o(z, o);
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
            if let LAYER::OUT(_) = self.layers[l+1]._type {

                //--------Looping length of next neurons---------------------------
                for n in 0..self.layers[l+1].neurons.len() {


                    // calculate (dE_total / dh)
                    let E_total = d_MSE(self.layers[l+1].neurons[n].input.reveal(), self.layers[l+1].neurons[n].o);

                    // calculate (do / dz)
                    let o = self.layers[l].activation.back_propagation(self.layers[l+1].neurons[n].o);

                    let E_total__o = E_total * o;

                    // update neurons weights of current layer using next layer's Errors
                    for w in 0..self.layers[l].neurons.len() {
                        // calculate (dz / dW)
                        self.layers[l].neurons[w].local_loss[n] = E_total__o * self.layers[l].neurons[w].o;

                        // update weight
                        self.layers[l].neurons[w].weights[n] = self.layers[l].neurons[w].weights[n] - (self.lr * self.layers[l].neurons[w].local_loss[n]);
                    }

                }
            } else {
                for n in 0..self.layers[l].neurons.len() {
                    for w in 0..self.layers[l+1].neurons.len() {

                        // sum sum of errors from next layer
                        let E_total:f64 = self.layers[l+1].neurons[w].local_loss.iter().sum();

                        // do back propagation
                        let h = self.layers[l].activation.back_propagation(self.layers[l+1].neurons[w].o);

                        // get local error, dE_total / dWn
                        self.layers[l].neurons[n].local_loss[w] = E_total * self.layers[l].neurons[n].o * h;

                        // and update weights, ( Wn - (lr * E_Wn) )
                        self.layers[l].neurons[n].weights[w] = self.layers[l].neurons[n].weights[w] - (self.lr * self.layers[l].neurons[n].local_loss[w]);
                    }
                }
            }
        }
    }

}

// trait LayerObj_c {
//     fn feed_forward();
//     fn back_propagation();
// }


struct LayerObj<'lng>
{
    name: &'static str,
    neurons: Vec<Neuron>,
    fan_in: Dimension,
    fan_out: Dimension,
    activation: Box<dyn Activation + 'lng>,
    _type: LAYER
}

impl<'lng> LayerObj<'lng> {
    fn new(fan_in: Dimension, fan_out: Dimension, layer: LAYER, activation: impl Activation + 'lng, name: Option<&'static str>) -> Self {
        if !layer.check_size(fan_in) {
            panic!("\n| vector and fan_in length mismatch!\n| at layer '{}', got {} size vector, expect {}\n", name.unwrap(), layer.get_len(), fan_in);
        }
        LayerObj {
            name: name.unwrap(),
            neurons: LayerObj::init_neurons(fan_in, fan_out, &layer),
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
    let mut new_model = Model::new("test");

    new_model.add_layer(8, 4, IN(input), Sigmoid, Some("input"));
    new_model.add_layer(4, 8, HIDDEN, Sigmoid, Some("input"));
    new_model.add_layer(8, 0, OUT(output), END, Some("output"));
    new_model.train(50, 0.2, true, 10);

    let mut w1 = Matrix::<f64>::new(3, 4);
    let mut w = Matrix::<f64>::new(3, 4);
    let mut a = Matrix::<f64>::new_scalar(4.);
    let mut b = Matrix::<f64>::new_scalar(2.);
    let mat = || Matrix::<f64>::new(3, 4);
    let matmat = Matrix::<Matrix<f64>>::from_fn(mat, 3, 3);
    w1.set(1, 3, 1.);
    w.set(1, 3, 4.);
    println!("{}", matmat);
}
