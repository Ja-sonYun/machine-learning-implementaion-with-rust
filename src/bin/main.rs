extern crate ml;
use ml::*;
use tools::*;

use activations::*;
use initializer::*;
use cost::*;
use maths::matrix::*;
use layer::LAYER::*;
use layer::*;
use model::*;

fn main()
{
    let input = vec![0.1, 0.2, 0.5, 0.8];
    let output = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    let mut new_model = Model::new("test", MSE, he_initializer);

    new_model.add_layer(SGD, 4, 4, IN(input), Sigmoid, Some("input"));
    new_model.add_layer(SGD, 4, 4, HIDDEN, Sigmoid, Some("input"));
    new_model.add_layer(SGD, 4, 8, HIDDEN, Sigmoid, Some("input"));
    new_model.add_layer(SGD, 8, 0, OUT(output), Sigmoid, Some("output"));
    new_model.train(500, 0.2, true, 10);

}
