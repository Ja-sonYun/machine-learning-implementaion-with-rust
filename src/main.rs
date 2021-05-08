pub mod utils {
    pub mod maths {
        pub mod algebra {
            use crate::utils::maths::derivative;
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
        pub fn derivative(f: fn(f64)->f64) -> impl Fn(f64) -> f64 {
            // let df = derivative(|x| x.powi(2));
            // println!("{}", df(3.));
            let h = 1e-6;
            move |x: f64| (f(x+h) - f(x)) / h
        }
        pub fn numerical_derivative(f: fn(&Vec<f64>)->f64, x: &mut Vec<f64>) -> Vec<f64> {
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
            rand()
        }

        // ReLU
        pub fn he_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight
        {
            rand()
        }
    }
    pub mod activations {
        pub fn sigmoid(x: f64) -> f64
        {
            1. / (1. + (-x).exp())
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

#[derive(Clone, Debug)]
struct Neuron
{
    input: Weight,
    past_z: Weight,
    weights: Vec<Weight>, // weights where routing from origins
    bias: Bias,
}

impl Neuron
{
    fn new(X: Weight, bias: Bias, dim: Dimension) -> Neuron
    {
        Neuron { input: X, past_z: 0., weights: vec![0.; dim as usize] , bias: bias }
    }

    fn get_Y(&self, index: usize) -> Weight
    {
        self.weights[index] * self.input
    }

    fn local_err(&self) -> Error
    {
        (self.past_z - self.input).powi(2) // int power
    }

    fn update_z(&mut self, weight_sum: Weight)
    {
        self.past_z = self.input;
        self.input = sigmoid(self.bias + weight_sum);
    }

    fn init_weights(&mut self, init_f: fn(Dimension, Dimension)->Weight, dimension: Dimension, fan_out: Dimension)
    {
        for i in 0..fan_out as usize
        {
            self.weights[i] = init_f(dimension, fan_out);
        }
    }

}

#[derive(Debug)]
struct Model
{
    name: &'static str,
    layers: Vec<Layer>,
    layer_size: usize,
}

impl Model {
    fn new(name: &'static str) -> Model {
        Model { name: name, layers: Vec::new(), layer_size: 0 }
    }
    fn push_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
        self.layer_size = self.layer_size + 1;
    }
    fn add_layer(&mut self, fan_in: Dimension, fan_out: Dimension, input: Option<Vec<Input>>, name: Option<&'static str>) {
        self.push_layer(Layer::new(fan_in, fan_out, input, name));
    }
    fn forward_propagation(&mut self) {
        for i in 0..(self.layer_size-1) {
            for k in 0..self.layers[i].fan_out as usize {
                if self.layers[i].fan_out != self.layers[i+1].fan_in {
                    panic!("layer dimension mismatch!");
                }
                let mut weight_sum: Weight = 0.;
                for j in 0..self.layers[i].fan_in as usize {
                    weight_sum = self.layers[i].neurons[j].get_Y(k) + weight_sum;
                }
                self.layers[i+1].neurons[k].update_z(weight_sum);
            }
        }
    }
    fn back_propagation(&mut self) {

    }
}

trait LayerBlock {
    type _Layer;
    type _Neurons;
    fn new(fan_in: Dimension, fan_out: Dimension, input: Option<Vec<Input>>, name: Option<&'static str>) -> Self::_Layer;
    // fn forward(&mut self, prev_layer: &Layer);
}

#[derive(Debug)]
struct Layer
{
    name: &'static str,
    neurons: Vec<Neuron>,
    fan_in: Dimension,
    fan_out: Dimension,
}

impl LayerBlock for Layer {
    type _Layer = Layer;
    type _Neurons = Vec<Neuron>;
    fn new(fan_in: Dimension, fan_out: Dimension, input: Option<Vec<Input>>, name: Option<&'static str>) -> Layer {
        // use input.len() as dimension
        let mut neurons = Vec::with_capacity(fan_in as usize);
        for i in 0..fan_in as usize {
            let mut neuron = Neuron::new(match &input {
                None => 0.,
                Some(val) => val[i],
            }, 0., fan_out);
            neuron.init_weights(xavier_initializer, fan_in, fan_out);
            neurons.push(neuron);
        }

        Layer { name: name.unwrap(), neurons: neurons, fan_in: fan_in, fan_out: fan_out }
    }

    // fn forward(&mut self, prev_layer: &Layer) {
    //     for i in 0..self.fan_out as usize
    //     {
    //         let mut weight_sum: Weight = 0.;

    //         // sum of weight(index j) that heading to i
    //         for j in 0..self.fan_in as usize
    //         {
    //             weight_sum = self.neurons[j].get_Y(i) + weight_sum;
    //         }

    //         // self.next.unwrap().neurons[i].update_z(weight_sum);
    //         // z = b + sum_{N}{i=1} a_i * w_i
    //     }
    // }
    // fn next_new(&mut self) -> Layer
    // {
    //     self.weight_initialize(self.dimension as i64);
    // }

    // MSE
    // fn loss(&self) -> Error
    // {
    //     let mut err: Error = 0.;
    //     for neuron in self.neurons.clone()
    //     {
    //         err += neuron.local_err();
    //     }

    //     err / Error::from(self.dimension as f64)
    // }


    // fn forward(&mut self, next_layer: &mut Layer)
    // {
    //     for i in 0..next_layer.dimension as usize
    //     {
    //         let mut weight_sum: Weight = 0.;

    //         // sum of weight(index j) that heading to i
    //         for j in 0..self.dimension as usize
    //         {
    //             weight_sum = self.neurons[j].get_Y(i) + weight_sum;
    //         }

    //         next_layer.neurons[i].update_z(weight_sum);
    //         // z = b + sum_{N}{i=1} a_i * w_i
    //     }
    // }
}
use utils::maths::rand;
use utils::maths::*;

fn main()
{
    let input = Some(vec![0.1, 0.2, 0.5]);
    let mut new_model = Model::new("test");
    new_model.add_layer(3, 4, input, Some("first"));
    new_model.add_layer(4, 2, None, Some("first"));
    new_model.add_layer(2, 2, None, Some("first"));
    new_model.forward_propagation();
    println!("{:?}", new_model);
    // let mut newModel = Model::<2>::new("new");
    // const A: [usize;2] = [1, 2];

    // newModel.add_layer::<{A[0]}, 3>(input, Some("new"));
    // layer.forward(&mut layer1);
    // println!("Hello, world!{}", L[0]);
    let df = derivative(|x| 3.*x*(x).exp());
    let pf = |v: &Vec<f64>|->f64 { ( v[3]*v[0] ) + ( v[0]*v[1]*v[3] ) + ( 3.*v[2] ) + ( v[3]*v[1].powi(2) ) };
    let mut vec: Vec<f64> = vec![2., 3., 1., 4.];

    println!("{:?}", numerical_derivative(pf, &mut vec));
}
