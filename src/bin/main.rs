extern crate ml;
use ml::*;
use tools::*;

use activations::*;
use initializer::*;
use cost::*;
// use maths::matrix::*;
use maths::n_matrix::*;
use layer::LAYER::*;
use layer::*;
use model::*;

fn main()
{
    // let input = vec![0.1, 0.2, 0.5, 0.8];
    // let output = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    // let mut new_model = Model::new("test", MSE, he_initializer);

    // new_model.add_layer(SGD, 4, 4, IN(input), Sigmoid, Some("input"));
    // new_model.add_layer(SGD, 4, 4, HIDDEN, Sigmoid, Some("input"));
    // new_model.add_layer(SGD, 4, 8, HIDDEN, Sigmoid, Some("input"));
    // new_model.add_layer(SGD, 8, 0, OUT(output), Sigmoid, Some("output"));
    // new_model.train(500, 0.2, true, 10);

    let mut a = Matrix::<f64>::fill_with(5., vec![2, 3, 4, 5]);
    a.set(vec![1, 0, 2, 0], 2.);
    a.set(vec![1, 0, 2, 1], 2.);
    a.set(vec![1, 0, 2, 2], 2.);
    a.set(vec![1, 1, 2, 2], 2.);
    let result = a.clone() + Matrix::<f64>::new_scalar(3.);
    println!("a: {}", a.get(vec![1]).get(vec![2]).get(vec![1]));
    println!("add: {}", result);

    // vector index => 0 is x, 1 is y, 2 is z...
    // let mut y = Matrix::<f64>::fill_with(1., vec![2, 3]);
    // println!("a: {}", y.get(vec![0]));


}
