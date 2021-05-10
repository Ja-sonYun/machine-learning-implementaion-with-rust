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
use layer::neuron::*;
use model::*;

fn main()
{
    let input = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    let output = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    let mut new_model = Model::new("test", MSE, he_initializer);

    new_model.add_layer(DummyLayer, 8, 4, IN(input), Sigmoid, Some("input"));
    new_model.add_layer(DummyLayer, 4, 8, HIDDEN, Sigmoid, Some("input"));
    new_model.add_layer(DummyLayer, 8, 0, OUT(output), A_END, Some("output"));
    new_model.train(50, 0.2, true, 10);

    let w1 = Matrix::<f64>::new(3, 4);
    let mut w = Matrix::<f64>::new(3, 4);
    let mut a = Matrix::<f64>::new_scalar(4.);
    let mut b = Matrix::<f64>::new_scalar(2.);
    let mat = || Matrix::<f64>::new(3, 4);
    let matmat = || Matrix::<Matrix<f64>>::from_fn(mat, 3, 3);
    let matmatmat = || Matrix::<Matrix<Matrix<f64>>>::from_fn(matmat, 3, 3);
    let matmatmatmat = Matrix::<Matrix<Matrix<Matrix<f64>>>>::from_fn(matmatmat, 3, 3);
    w.set(1, 3, 4.);
    if matmatmatmat.is_zero() {
        println!("{}", w1);
        println!("zero");
    }
}
