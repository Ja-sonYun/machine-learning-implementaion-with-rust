pub mod utils {
    pub mod activations {
        use crate::utils::types::*;
        use rand::Rng;
        // Sigmoid / Tanh
        pub fn xavier_initialization(fan_in: Dimension, fan_out: Dimension) -> Weight
        {
            let mut rng = rand::thread_rng();
            rng.gen_range((fan_in as f64)..(fan_out as f64)) as f64 / (fan_in as f64).sqrt()
        }

        // ReLU
        pub fn he_initialization(fan_in: Dimension, fan_out: Dimension) -> Weight
        {
            let mut rng = rand::thread_rng();
            rng.gen_range((fan_in as f64)..(fan_out as f64)) as f64 / ((fan_in / 2) as f64).sqrt()
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
        self.input = Neuron::sigmoid(self.bias + weight_sum);
        println!("{} ", self.input);
    }

    fn init_weights(&mut self, init_f: fn(Dimension, Dimension)->Weight, dimension: Dimension, fan_out: Dimension)
    {
        for i in 0..fan_out as usize
        {
            self.weights[i] = init_f(dimension, fan_out);
        }
    }

    fn sigmoid(x: f64) -> f64
    {
        1. / (1. + (-x).exp())
    }
}

#[derive(Debug)]
struct Model<const DEP: usize>
{
    name: &'static str,
    layers: [Option<Layer>; DEP],
}

impl<const DEP: usize> Model<DEP> {
    fn new(name: &'static str) -> Model<DEP> {
        Model { name: name, layers: [None; DEP] }
    }

}

trait LayerBlock {
    type _Layer;
    type _Neurons;
    fn new(fan_in: Dimension, fan_out: Dimension, input: Option<Vec<Input>>, name: Option<&'static str>) -> Self::_Layer;
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
            neuron.init_weights(xavier_initialization, fan_in, fan_out);
            neurons.push(neuron);
        }

        Layer { name: name.unwrap(), neurons: neurons, fan_in: fan_in, fan_out: fan_out }
    }
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

fn main()
{
    let input = Some(vec![0.1, 0.2, 0.5]);

    // let mut newModel = Model::<2>::new("new");
    // const A: [usize;2] = [1, 2];

    // newModel.add_layer::<{A[0]}, 3>(input, Some("new"));
    // layer.forward(&mut layer1);
    // println!("Hello, world!{}", L[0]);

}
