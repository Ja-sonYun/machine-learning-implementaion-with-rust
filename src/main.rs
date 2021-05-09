#[macro_use]
pub mod macros {
    #[macro_export]
    macro_rules! forfor {
        ($y: expr, $yn: ident, $x: expr, $xn: ident, $bk: block) => {
            for $yn in 0..$y {
                for $xn in 0..$x {
                    $bk
                }
            }
        }
    }
}
pub mod layer {

    use crate::utils::types::*;
    use crate::utils::maths::*;

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

pub mod utils {
    pub mod maths {
        #[derive(Debug, Clone, Copy)]
        pub enum NumOrZeroF64 {
            Num(f64),
            Zero,
        }
        impl NumOrZeroF64 {
            pub fn reveal(self) -> f64 {
                match self {
                    NumOrZeroF64::Num(x) => x,
                    _ => 0.,
                }
            }
        }

        // use std::collections::HashMap;
        // pub struct Expression<EXP: Fn(Vec<f64>) -> f64> {
        //     exp: EXP,
        //     map: HashMap<String, i64>
        // }
        // impl<EXP: Fn(Vec<f64>) -> f64> Expression<EXP> {
        //     fn new()
        // }
        pub mod algebra {
            use std::ops::{Rem, Add, Mul, Sub, Div};
            use std::fmt::{Display, Debug, Formatter, Result};
            use num_traits::{Zero};

            // type DebugElement = Zero + Display + Copy;

            // use crate::utils::maths::derivative;
            #[derive(Clone, Copy)]
            enum Element<T> {
                Zero,
                Num(T)
            }
            impl<T> Element<T> {
                fn num(self) -> T where T: Zero + Display + Copy {
                    match self {
                        Element::Num(v) => v,
                        _ => Zero::zero(),
                    }
                }
            }
            impl<T> Debug for Element<T> where T: Zero + Display + Copy {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(f, "{}", self.num())
                }
            }

            pub struct Matrix<T> {
                n: Vec<Vec<Element<T>>>,
                _nx: i64,
                _ny: i64,
            }

            impl<T> Matrix<T> where T: Clone + Copy + Zero {
                pub fn new(ny: i64, nx:i64) -> Matrix<T> {
                    Matrix { n: (0..ny).map(|_| vec![Element::Zero; nx as usize]).collect(), _nx: nx, _ny: ny }
                }
                pub fn new_scalar(val: T) -> Matrix<T> {
                    Matrix { n: vec![vec![Element::Num(val)]], _nx: 1, _ny: 1 }
                }
                pub fn is_scalar(&self) -> bool {
                    self._nx == 1 && self._ny == 1
                }
                pub fn from_fn<F: Fn()->T>(init_fn: F, ny: i64, nx:i64) -> Matrix<T> {
                    Matrix { n: (0..ny).map(|_| (0..nx).map(|_| Element::Num(init_fn())).collect()).collect(), _nx: nx, _ny: ny }
                }
                pub fn get(&self, y: i64, x: i64) -> T {
                    match &self.n[y as usize][x as usize] {
                        Element::Num(n) => *n,
                        Element::Zero => Zero::zero(),
                    }
                }
                pub fn s_get(&self) -> T {
                    if self.is_scalar() {
                        self.get(0, 0)
                    } else {
                        panic!("this is matrix");
                    }
                }
                pub fn get_dim(&self) -> (i64, i64) {
                    (self._nx, self._ny)
                }
                pub fn comp_dim_with(&self, shr: &Self) -> bool {
                    self.get_dim() == shr.get_dim()
                }
                pub fn set(&mut self, y: i64, x: i64, val: T) {
                    if self.is_scalar() {
                        panic!("this is scalar")
                    }
                    self.n[y as usize][x as usize] = Element::Num(val);
                }
                pub fn s_set(&mut self, val: T) {
                    if self.is_scalar() {
                        self.n[0][0] = Element::Num(val);
                    } else {
                        panic!("This is a matrix not the scalar.");
                    }
                }
                pub fn elem_with_scalar<F: Fn(T, T)->T>(&self, with: T, cal_fn: F) -> Self {
                    // TODO: Refactoring this
                    let mut temp = Matrix::<T>::new(self._ny, self._nx);
                    for y in 0..self._ny as usize {
                        for x in 0..self._nx as usize {
                            // println!("{}, {}", y, x);
                            temp.n[y][x] = Element::Num(cal_fn(self.get(y as i64, x as i64), with));
                        }
                    }
                    temp
                }
                pub fn elemwise_cal<F: Fn(T, T)->T>(&self, with: Self, cal_fn: F) -> Self {
                    let mut temp = Matrix::<T>::new(self._ny, self._nx);
                    forfor!(self._ny, y, self._nx, x, {
                        temp.set(y, x, cal_fn(self.get(y, x), with.get(y, x)));
                    });
                    temp
                }
                pub fn cal_with_scalar<F: Fn(T, T)->T>(this: Self, andthis: Self, cal_fn: F) -> Self {
                    if this.is_scalar() {
                        andthis.elem_with_scalar(this.s_get(), cal_fn)
                    } else if andthis.is_scalar() {
                        this.elem_with_scalar(andthis.s_get(), cal_fn)
                    } else if this.comp_dim_with(&andthis) {
                        this.elemwise_cal(andthis, cal_fn)
                    } else {
                        panic!("can't calculate with this");
                    }
                }
                // pub fn derivative(&mut self) -> Matrix<f64> {
                // }
            }

            impl<T> Display for Matrix<T> where T: Display + Zero + Copy {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    let mut strg = "".to_owned();
                    for y in 0..self._ny as usize {
                        for x in 0..self._nx as usize {
                            strg.push_str(&format!("{:>5}", format!("{:?}", self.n[y][x])));
                        }
                        strg.push_str("\n");
                    }
                    // write!(f, "{:?}", self.n[y][x]);
                    write!(f, "{}", strg)
                }
            }
            macro_rules! opt_impl {
                ($funcn:ident, $func:ident, $c:expr) => {
                    impl<T> $funcn<Matrix<T>> for Matrix<T> where T: Clone + Copy + Zero + Display + $funcn<Output = T> {
                        type Output = Self;
                        fn $func(self, rhs: Self) -> Self {
                            Self::cal_with_scalar(self, rhs, $c)
                        }
                    }
                }
            }
            opt_impl!(Add, add, |a, b| a + b);
            opt_impl!(Mul, mul, |a, b| a + b);
            opt_impl!(Sub, sub, |a, b| a - b);
            opt_impl!(Div, div, |a, b| a / b);

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
            pub fn randin(from: f64, to: f64) -> f64 {
                rand::thread_rng().gen_range(from..to)
            }
        }
    }
    pub mod initializer {
        use crate::utils::types::{Dimension, Weight};
        use crate::utils::maths::rand::*;
        // Sigmoid / Tanh
        pub fn xavier_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight {
            // rand()
                1.
        }

        // ReLU
        pub fn he_initializer(fan_in: Dimension, fan_out: Dimension) -> Weight {
            rand()
        }
    }
    pub mod activations {
        pub fn sigmoid(x: f64) -> f64 {
            1. / (1. + (-x).exp())
        }
        pub fn d_sigmoid_v(vec: &Vec<f64>) -> f64 {
            vec[0] * ( 1. - vec[0] )
        }
        pub fn d_sigmoid(x: f64) -> f64 {
            x * (1. - x)
        }
        pub fn ReLU(x: f64) -> f64 {
            if 0. < x {
                x
            } else {
                0.
            }
        }
        pub fn d_ReLU(x: f64) -> f64 {
            if 0. < x {
                1.
            } else {
                0.
            }
        }
        pub fn Leaky_ReLU(x: f64) -> f64 {
            if 0. < x {
                x
            } else {
                x * 0.001
            }
        }
        pub fn d_Leaky_ReLU(x: f64) -> f64 {
            if 0. < x {
                1.
            } else {
                0.001
            }
        }
        // pub fn ELU(x: f64) -> f64 {
        //     if 0. < x {
        //         x
        //     } else {
        //         x.exp() - 1.
        //     }
        // }
        // pub fn d_ELU(x: f64) -> f64 {
        //     if 0. < x {
        //         1.
        //     } else {
        //         x.exp()
        //     }
        // }
    }

    pub mod cost {
        use crate::utils::types::Error;

        pub fn MSE(target: f64, output: f64) -> Error {
            (target - output).powi(2) / 2.
        }
        pub fn d_MSE_v(vec: &Vec<f64>) -> Error {
            (vec[0]).powi(2) / 2.
        }
        pub fn d_MSE(target: f64, output: f64) -> Error {
            -(target - output)
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
use utils::maths::algebra::*;
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
    fn new(x: NumOrZeroF64, bias: Bias, fan_in: Dimension, fan_out: Dimension) -> Neuron {
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

    fn update_z(&mut self, weight_sum: Weight, act: fn(f64)->f64) {
        self.z = self.bias + weight_sum;
        self.o = act(self.z);
    }
}

#[derive(Debug)]
struct Model
{
    name: &'static str,
    layers: Vec<Layer>,
    layer_size: usize,
    lr: f64,                // learning rate
    cost: Error,            // total loss
}

impl Model {
    fn new(name: &'static str, lr: f64) -> Model {
        Model { name: name, layers: Vec::new(), layer_size: 0, lr: lr, cost: 0. }
    }
    fn push_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
        self.layer_size = self.layer_size + 1;
    }
    fn add_layer(&mut self, fan_in: Dimension, fan_out: Dimension, layer: LAYER, name: Option<&'static str>) {
        self.push_layer(Layer::new(fan_in, fan_out, layer, name));
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
                    self.layers[i+1].neurons[k].update_z(weight_sum, sigmoid);
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
                    let o = d_sigmoid(self.layers[l+1].neurons[n].o);
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
                        let h = d_sigmoid(self.layers[l+1].neurons[w].o);
                        self.layers[l].neurons[n].local_loss[w] = dE_total_W * self.layers[l].neurons[n].o * h;

                        self.layers[l].neurons[n].weights[w] = self.layers[l].neurons[n].weights[w] - (self.lr * self.layers[l].neurons[n].local_loss[w]);
                    }
                }
            }
        }
    }

}

trait LayerBlock {
    type _Layer;
    fn new(fan_in: Dimension, fan_out: Dimension, layer: LAYER, name: Option<&'static str>) -> Self::_Layer;
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
        if !layer.check_size(fan_in) {
            panic!("\n| vector and fan_in length mismatch!\n| at layer '{}', got {} size vector, expect {}\n", name.unwrap(), layer.get_len(), fan_in);
        }
        Layer {
            name: name.unwrap(),
            neurons: Layer::init_neurons(fan_in, fan_out, &layer),
            fan_in: fan_in,
            fan_out: fan_out,
            _type: layer
        }
    }
}

fn main()
{
    let input = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    let output = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    let mut new_model = Model::new("test", 0.1);
    new_model.add_layer(8, 9, IN(input), Some("input"));
    new_model.add_layer(9, 4, HIDDEN, Some("input"));
    new_model.add_layer(4, 3, HIDDEN, Some("input"));
    new_model.add_layer(3, 8, HIDDEN, Some("input"));
    new_model.add_layer(8, 0, OUT(output), Some("output"));

    // for i in (0..500) {
    //     // if i % 10 == 0 && 0 != i {
    //         println!("{:0} epoch, loss: {}", i, new_model.cost);
    //     // }
    //     let bp = new_model.cost;
    //     new_model.forward_propagation();
    //     // if bp < new_model.cost && bp != 0. {
    //     //     println!("over");
    //     //     break
    //     // };
    //     new_model.back_propagation();
    // }
    let mut w1 = Matrix::<f64>::new(3, 4);
    let mut w = Matrix::<f64>::new(3, 4);
    let mut a = Matrix::<f64>::new_scalar(4.);
    let mut b = Matrix::<f64>::new_scalar(2.);
    w.set(1, 3, 1.);
    w.set(1, 3, 4.);
    println!("{}", w * w1);
}
