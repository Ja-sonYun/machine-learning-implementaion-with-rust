pub mod utils {
    pub mod maths {
        // use std::collections::HashMap;
        // pub struct Expression<EXP: Fn(Vec<f64>) -> f64> {
        //     exp: EXP,
        //     map: HashMap<String, i64>
        // }
        // impl<EXP: Fn(Vec<f64>) -> f64> Expression<EXP> {
        //     fn new()
        // }
        pub mod algebra {
            // use crate::utils::maths::derivative;
            #[derive(Debug, Clone)]
            enum Element<T> {
                Zero,
                Num(T)
            }

            #[derive(Debug, Clone)]
            pub struct Matrix<T> {
                n: Vec<Vec<Element<T>>>,
                _nx: i64,
                _ny: i64,
            }

            impl<T> Matrix<T> where T: Clone {
                fn new(ny: i64, nx:i64) -> Matrix<T> {
                    Matrix { n: (0..ny).map(|_| vec![Element::Zero; nx as usize]).collect(), _nx: nx, _ny: ny }
                }
                pub fn from_fn<F: Fn()->T>(init_fn: F, ny: i64, nx:i64) -> Matrix<T> {
                    Matrix { n: (0..ny).map(|_| (0..nx).map(|_| Element::Num(init_fn())).collect()).collect(), _nx: nx, _ny: ny }
                }
                pub fn get(&self, y: i64, x: i64) -> &T {
                    match &self.n[y as usize][x as usize] {
                        Element::Num(n) => n,
                        Element::Zero => panic!("out of index!"),
                    }
                }
                pub fn set(&mut self, y: i64, x: i64, val: T) {
                    self.n[y as usize][x as usize] = Element::Num(val);
                }
                pub fn elem_cal<F: Fn(&T, &T)->T>(&mut self, with: &T, cal_fn: F) -> Matrix<T> {
                    let mut temp = Matrix::<T>::new(self._ny, self._nx);
                    for y in 0..self._ny as usize {
                        for x in 0..self._nx as usize {
                            // println!("{}, {}", y, x);
                            temp.n[y][x] = Element::Num(cal_fn(self.get(y as i64, x as i64), with));
                        }
                    }
                    temp
                }
                // pub fn derivative(&mut self) -> Matrix<f64> {
                    
                // }
            }
        }
        // pub fn derivative(f: fn(f64)->f64) -> impl Fn(f64) -> f64 {
        //     // let df = derivative(|x| x.powi(2));
        //     // println!("{}", df(3.));
        //     let h = 1e-6;
        //     move |x: f64| (f(x+h) - f(x)) / h
        // }
        pub fn numerical_derivative(f: fn(&Vec<f64>)->f64, x: &mut Vec<f64>) -> Vec<f64> {
            // let pf = |v: &Vec<f64>|->f64 { ( v[3]*v[0] ) + ( v[0]*v[1]*v[3] ) + ( 3.*v[2] ) + ( v[3]*v[1].powi(2) ) };
            // let mut vec: Vec<f64> = vec![2., 3., 1., 4.];
            // println!("{:?}", numerical_derivative(pf, &mut vec));
            let delta_x = 1e-4;
            let mut grad = vec![0.; x.len()];

            for i in 0..x.len() as usize {
                let tmp_val = x[i];
                x[i] = tmp_val + delta_x;
                let fx1 = f(x);

                x[i] = tmp_val - delta_x;
                let fx2 = f(x);

                grad[i] = (fx1 - fx2) / (2. * delta_x);
                x[i] = tmp_val;
            }
            grad
        }
        pub mod rand {
            use crate::utils::maths::algebra::*;
            use crate::utils::types::Dimension;
            use rand::Rng;
            pub fn randn(n_input: Dimension, n_output: Dimension) -> Matrix<f64> {
                Matrix::<f64>::from_fn(rand, n_input, n_output)
            }
            pub fn rand() -> f64 {
                rand::thread_rng().gen::<f64>()
            }
        }
    }
    pub mod initializer {
        use crate::utils::types::{Dimension, Weight};
        use crate::utils::maths::rand::*;
        // Sigmoid / Tanh
        pub fn xavier_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight
        {
            // rand()
                1.
        }

        // ReLU
        pub fn he_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight
        {
            // rand()
                1.
        }
    }
    pub mod activations {
        pub fn sigmoid(x: f64) -> f64
        {
            // 1. / (1. + (-x).exp())
            x
        }
    }

    pub mod cost {
        use crate::utils::types::Error;

        pub fn MSE(target: f64, output: f64) -> Error {
            (target - output).powi(2) / 2.
        }
        pub fn d_MSE(vec: &Vec<f64>) -> Error {
            -((vec[0]).powi(2) / 2.)
        }
    }
    pub mod types {
        pub type Dimension = i64;
        pub type Weight = f64;
        pub type Bias = f64;
        pub type Error = f64;
        pub type Input = f64;
    }
}
use utils::types::*;
use utils::activations::*;
use utils::initializer::*;
use utils::cost::*;
use utils::maths::*;

#[derive(Clone, Debug)]
struct Neuron {
    input: NumOrZeroF64,
    z: Weight,              // summed of weights
    h: Weight,              // value that wrapped by activation func
    weights: Vec<Weight>,   // weights where routing from origins
    bias: Bias,             // bias, TODO: toggle this
    loss: Error,            // local loss
}

#[derive(Debug, Clone, Copy)]
enum NumOrZeroF64 {
    Num(f64),
    Zero,
}
impl NumOrZeroF64 {
    fn reveal(self) -> f64 {
        match self {
            NumOrZeroF64::Num(x) => x,
            _ => 0.,
        }
    }
}

impl Neuron {
    fn new(x: NumOrZeroF64, bias: Bias, fan_in: Dimension, fan_out: Dimension) -> Neuron {
        Neuron {
            input: x,
            z: 0.,
            h: x.reveal(),
            weights: (0..fan_out).map(|_| he_initializer(fan_in, fan_out)).collect(),
            bias: bias,
            loss: 0.
        }
    }

    // W * x
    fn get_wx(&self, index: usize) -> Weight {
        self.weights[index] * self.h
    }

    fn get_loss(&mut self) -> Error {
        self.loss = MSE(self.input.reveal(), self.z);
        self.loss
    }

    fn update_z(&mut self, weight_sum: Weight) {
        self.z = self.bias + weight_sum;
        self.h = sigmoid(self.z);
    }
}

#[derive(Debug)]
struct Model
{
    name: &'static str,
    layers: Vec<Layer>,
    layer_size: usize,
    cost: Error,
}

impl Model {
    fn new(name: &'static str) -> Model {
        Model { name: name, layers: Vec::new(), layer_size: 0, cost: 0. }
    }
    fn push_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
        self.layer_size = self.layer_size + 1;
    }
    fn add_layer(&mut self, fan_in: Dimension, fan_out: Dimension, layer: LAYER, name: Option<&'static str>) {
        self.push_layer(Layer::new(fan_in, fan_out, layer, name));
    }
    fn forward_propagation(&mut self) {
        // loop layers
        for i in 0..(self.layer_size-1) {
            // loop each neurons
            for k in 0..self.layers[i+1].neurons.len() as usize {
                if self.layers[i].fan_out != self.layers[i+1].fan_in {
                    panic!("layer dimension mismatch!");
                }
                let mut weight_sum: Weight = 0.;
                // loop for get sum of neuron's weights
                for j in 0..self.layers[i].neurons.len() as usize {
                    weight_sum = self.layers[i].neurons[j].get_wx(k) + weight_sum;
                }
                self.layers[i+1].neurons[k].update_z(weight_sum);
            }
        }
        // loop for get total loss and update local losses if its output layer
        if let LAYER::OUT(_) = self.layers[self.layer_size - 1]._type {
            for neuron in &mut self.layers[self.layer_size - 1].neurons {
                self.cost = self.cost + neuron.get_loss();
            }
        }
    }
    fn back_propagation(&mut self) {
        println!("{:?}", self.layers[self.layer_size-1].neurons);
        for neuron in &self.layers[self.layer_size-1].neurons {
            // calculate E_total
            let mut target_output: Vec<f64> = vec![neuron.input.reveal() - neuron.z];
            let E_total = numerical_derivative(d_MSE, &mut target_output);
            println!("E_total {:?}", E_total);
        }
        // for i in (self.layer_size)..1 {
        //     for k in 0..self.layers[i].fan_in as usize {
        //         self.layers[i-1]
        //     }
        // }
    }
}

trait LayerBlock {
    type _Layer;
    fn new(fan_in: Dimension, fan_out: Dimension, layer: LAYER, name: Option<&'static str>) -> Self::_Layer;
    fn init_neurons(fan_in: Dimension, fan_out: Dimension, layer: &LAYER) -> Vec<Neuron> {
        let mut neurons: Vec<Neuron>;
        // if this layer is input or hidden,
        neurons = Vec::with_capacity(fan_in as usize);
        for i in 0..layer.get_dim(fan_in, fan_out) as usize {
            neurons.push(Neuron::new(layer.unwrap_with_index(i), 0., fan_in, fan_out));
        }
        // if this layer is output, which is doens't have nuerons
        neurons
    }
    // fn forward(&mut self, prev_layer: &Layer);
}

#[derive(Debug)]
struct Layer
{
    name: &'static str,
    neurons: Vec<Neuron>,
    fan_in: Dimension,
    fan_out: Dimension,
    _type: LAYER
}

impl LayerBlock for Layer {
    type _Layer = Layer;
    fn new(fan_in: Dimension, fan_out: Dimension, layer: LAYER, name: Option<&'static str>) -> Layer {
        Layer {
            name: name.unwrap(),
            neurons: Layer::init_neurons(fan_in, fan_out, &layer),
            fan_in: fan_in,
            fan_out: fan_out,
            _type: layer
        }
    }
}

#[derive(Debug)]
enum LAYER {
    IN(Vec<Weight>),
    OUT(Vec<Weight>),
    HIDDEN,
}
impl LAYER {
    fn unwrap(self) -> Option<Vec<Weight>> {
        match self {
            LAYER::IN(vw) | LAYER::OUT(vw) => Some(vw),
            _ => None
        }
    }
    fn unwrap_with_index(&self, i: usize) -> NumOrZeroF64 {
        match &self {
            LAYER::IN(vw) | LAYER::OUT(vw) => NumOrZeroF64::Num(vw[i]),
            _ => NumOrZeroF64::Zero
        }
    }
    fn get_dim(&self, fan_in: Dimension, fan_out: Dimension) -> Dimension {
        match &self {
            LAYER::IN(_) | LAYER::HIDDEN => fan_in,
            LAYER::OUT(_) => fan_out
        }
    }
}

fn main()
{
    let input = vec![0.1, 0.2, 0.5];
    let output = vec![0.4, 0.3];
    let mut new_model = Model::new("test");
    // 0, 4
    // 4, 3
    // 3, 0
    new_model.add_layer(3, 4, LAYER::IN(input), Some("first"));
    new_model.add_layer(4, 4, LAYER::HIDDEN, Some("first"));
    new_model.add_layer(4, 2, LAYER::HIDDEN, Some("first"));
    new_model.add_layer(2, 0, LAYER::OUT(output), Some("first"));
    // new_model.forward_propagation();
    // new_model.back_propagation();
    // new_model.back_propagation();
    println!("{:?}", new_model);
    // let mut newModel = Model::<2>::new("new");
    // const A: [usize;2] = [1, 2];

    // newModel.add_layer::<{A[0]}, 3>(input, Some("new"));
    // layer.forward(&mut layer1);
    // println!("Hello, world!{}", L[0]);
    // let mut vec: Vec<f64> = vec![, 4.];
    // println!("{:?}", numerical_derivative(pf, &mut vec));
}
