use rand::Rng;

type Dimension = i64;
type Weight = f64;
type Bias = f64;
type Error = f64;
type Input = f64;

#[derive(Copy, Clone)]
struct Routes
{
    weight: Weight,
    bias: Bias
}

#[derive(Clone)]
struct Neuron
{
    input: Weight,
    past_z: Weight,
    weights: Vec<Weight>, // weights where routing from origins
    bias: Bias,
}

impl Neuron
{
    fn new(X: Weight, bias: Bias) -> Neuron
    {
        Neuron { input: X, past_z: 0., weights: Vec::new(), bias: bias }
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

    fn add_origin(&mut self, weight: Weight)
    {
        self.weights.push(weight);
    }

    fn sigmoid(x: f64) -> f64
    {
        1. / (1. + (-x).exp())
    }
}

struct Layer
{
    neurons: Vec<Neuron>,
    dimension: Dimension,
}

impl Layer
{
    fn new(dimension: Dimension, input: Option<Vec<Input>>) -> Layer
    {
        let mut neurons: Vec<Neuron> = Vec::new();

        for i in 0..dimension as usize
        {
            let x = match &input {
                None => 0.,
                Some(val) => val[i],
            };
            let new_local_neuron = Neuron::new(x, 0.);
            neurons.push(new_local_neuron);
        }

        Layer { neurons: neurons, dimension: dimension as Dimension }
    }

    fn next_new(&mut self, dimension: Dimension) -> Layer
    {
        self.weight_initialize(dimension);
        Layer::new(dimension, None)
    }

    // MSE
    fn loss(&self) -> Error
    {
        let mut err: Error = 0.;
        for neuron in self.neurons.clone()
        {
            err += neuron.local_err();
        }

        err / Error::from(self.dimension as f64)
    }

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
            for _ in 0..fan_out as usize
            {
                self.neurons[i].add_origin(Layer::xavier_initialization(self.dimension, fan_out));
            }
        }
    }

    fn forward(&mut self, next_layer: &mut Layer)
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

fn main()
{
    let input = vec![0.1, 0.2, 0.5];
    let mut layer = Layer::new(3, Some(input));
    let mut layer1 = layer.next_new(11);
    let mut layer2 = layer1.next_new(11);
    layer.forward(&mut layer1);
    println!("Hello, world!");

}
