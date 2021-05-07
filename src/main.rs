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
use std::mem::MaybeUninit;
use std::any::{Any};
use utils::types::*;
use utils::activations::*;

#[derive(Clone, Debug)]
struct Neuron<const OUT: usize>
{
    input: Weight,
    past_z: Weight,
    weights: [Weight; OUT], // weights where routing from origins
    bias: Bias,
}

impl<const OUT: usize> Neuron<OUT>
{
    fn new(X: Weight, bias: Bias) -> Neuron<OUT>
    {
        Neuron { input: X, past_z: 0., weights: [0.;OUT], bias: bias }
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
        self.input = Neuron::<OUT>::sigmoid(self.bias + weight_sum);
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
    layers: Vec<Box<dyn Any>>,
    structure: Option<[usize;DEP]>,
}

impl<const DEP: usize> Model<DEP> {
    fn new(name: &'static str) -> Model<DEP> {
        Model { name: name, layers: Vec::new(), structure: None }
    }

    fn add_layer<const IN: usize, const OUT: usize>(&mut self, input: Option<Vec<Input>>, name: Option<&'static str>) {
        let mut new = &Layer::<IN, OUT>::new(input, name) as &dyn Any;
        const A: [usize;2] = [IN, OUT];

        // self.layers.push(Box::new(Layer::<IN, OUT>::new(input, name)));
        // let ll = self.layers[0].downcast();
        println!("{:?}", new.downcast_ref::<Layer<IN, OUT>>());
    }
}

trait LayerBlock {
    type _Layer;
    type _Neurons;
    const IN: usize;
    const OUT: usize;
    fn new(input: Option<Vec<Input>>, name: Option<&'static str>) -> Self::_Layer;
}

#[derive(Debug)]
struct Layer<const IN: usize, const OUT: usize>
{
    name: &'static str,
    neurons: [Neuron<OUT>; IN],
    dimension: Dimension,
}

impl<const IN: usize, const OUT: usize> LayerBlock for Layer<IN, OUT> {
    type _Layer = Layer<IN, OUT>;
    type _Neurons = [Neuron<OUT>; IN];
    const IN: usize = IN;
    const OUT: usize = OUT;
    fn new(input: Option<Vec<Input>>, name: Option<&'static str>) -> Layer<IN, OUT> {
        let neurons = unsafe {
            let mut neurons: [MaybeUninit<Neuron<OUT>>; IN] = MaybeUninit::uninit().assume_init();
            for i in 0..IN as usize {
                let mut neuron = Neuron::<OUT>::new(match &input {
                    None => 0.,
                    Some(val) => val[i],
                }, 0.);
                neuron.init_weights(xavier_initialization, IN as Dimension, OUT as Dimension);
                std::ptr::write(neurons[i].as_mut_ptr(), neuron);
            }
            std::mem::transmute_copy::<_, [Neuron<OUT>; IN]>(&neurons)
        };

        Layer { name: name.unwrap(), neurons: neurons, dimension: IN as Dimension }
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

#[derive(Debug)]
struct A<const D: usize>
{}

trait AA {
    type Item;
    fn new() -> Self::Item;
    fn get_d<const A: usize>(&self) ->usize;
}

impl<const D: usize> AA for A<D>{
    type Item = A<D>;
    fn new() -> A<D> {
        A {}
    }
    fn get_d<const A: usize>(&self) ->usize {
        A * D
    }
}

fn main()
{
    let mut aaa: Vec<&dyn Any> = Vec::new();
    let mut a = A::<2>::new();
    // aaa.push(&a);
    println!("{:?}", a.get_d::<2>());
    // println!("{:?}", aaa[0].downcast_ref::<AA>());

    let input = Some(vec![0.1, 0.2, 0.5]);

    let mut newModel = Model::<2>::new("new");
    const A: [usize;2] = [1, 2];

    newModel.add_layer::<{A[0]}, 3>(input, Some("new"));
    // layer.forward(&mut layer1);
    // println!("Hello, world!{}", L[0]);

}
