use rand::Rng;
use std::mem::{self, MaybeUninit};

type Dimension = i64;
type Weight = f64;
type Bias = f64;
type Error = f64;
type Input = f64;

#[derive(Clone, Copy)]
struct Neuron<const D: usize>
{
    input: Weight,
    past_z: Weight,
    weights: [Weight; D], // weights where routing from origins
    bias: Bias,
}

impl<const D: usize> Neuron <D>
{
    fn new(X: Weight, bias: Bias) -> Neuron<D>
    {
        Neuron { input: X, past_z: 0., weights: [0.; D], bias: bias }
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
        self.input = self.sigmoid(self.bias + weight_sum);
        println!("{} ", self.input);
    }

    fn init_weights(&mut self, init_f: fn(Dimension, Dimension)->Weight, dimension: Dimension, fan_out: Dimension)
    {
        for i in 0..fan_out as usize
        {
            self.weights[i] = init_f(dimension, fan_out);
        }
    }

    fn sigmoid(self, x: f64) -> f64
    {
        1. / (1. + (-x).exp())
    }
}

struct Layer<const D: usize>
{
    neurons: [Neuron<D>; D],
    dimension: Dimension,
}

impl<const D: usize> Layer <D>
{
    fn new(input: Option<Vec<Input>>) -> Layer<D>
    {
        match &input {
            Some(val) => {
                if val.len() != D {
                    panic!("Input Dimension Mismatch!")
                }
            },
            None => {},
        };
        let neurons: [Neuron<D>; D] = {
            let mut neurons: [MaybeUninit<Neuron<D>>; D] = unsafe {
                MaybeUninit::uninit().assume_init()
            };
            for i in 0..D as usize
            {
                let x = match &input {
                    None => 0.,
                    Some(val) => val[i],
                };
                neurons[i] = MaybeUninit::new(Neuron::<D>::new(x, 0.));
            }
            unsafe { mem::transmute_copy::<_, [Neuron<D>; D]>(&neurons) }
        };

        Layer { neurons: neurons, dimension: D as Dimension }
    }

    fn next_new(&mut self) -> Layer<D>
    {
        self.weight_initialize(D as i64);
        Layer::new(None)
    }

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

    // Sigmoid / Tanh
    fn xavier_initialization(fan_in: Dimension, fan_out: Dimension) -> Weight
    {
        let mut rng = rand::thread_rng();
        rng.gen_range((fan_in as f64)..(fan_out as f64)) as f64 / (fan_in as f64).sqrt()
    }

    // ReLU
    fn he_initialization(fan_in: Dimension, fan_out: Dimension) -> Weight
    {
        let mut rng = rand::thread_rng();
        rng.gen_range((fan_in as f64)..(fan_out as f64)) as f64 / ((fan_in / 2) as f64).sqrt()
    }

    fn weight_initialize(&mut self, fan_out: Dimension)
    {
        for i in 0..self.dimension as usize
        {
            self.neurons[i].init_weights(Layer::<D>::xavier_initialization, self.dimension, fan_out);
        }
    }

    fn forward(&mut self, next_layer: &mut Layer<D>)
    {
        for i in 0..next_layer.dimension as usize
        {
            let mut weight_sum: Weight = 0.;

            // sum of weight(index j) that heading to i
            for j in 0..self.dimension as usize
            {
                weight_sum = self.neurons[j].get_Y(i) + weight_sum;
            }

            next_layer.neurons[i].update_z(weight_sum);
            // z = b + sum_{N}{i=1} a_i * w_i
        }
    }
}

struct A<const L: usize>
{
    layer: [f64; L]
}
impl<const L: usize> A<L>
{
    fn new() -> A<L>
    {
        A { layer: [0.; L] }
    }
}

fn main()
{
    let input = vec![0.1, 0.2, 0.5];

    let mut layer = Layer::<4>::new(Some(input));
    // layer.forward(&mut layer1);
    println!("Hello, world!");

}
