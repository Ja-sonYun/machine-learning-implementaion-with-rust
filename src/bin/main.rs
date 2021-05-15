extern crate ml;
use ml::*;
use tools::*;
use tools::test_tools::test_excution_time;
use std::fmt::{Display, Debug, Formatter, Result};

use ndarray::{IxDyn, ArrayD};

use activations::*;
use initializer::*;
use cost::*;
use maths::matrix::*;
use maths::n_matrix::*;
use maths::matrix::Matrix_d;
use layer::LAYER::*;
use layer::*;
use model::*;

fn main()
{
    // let input = vec![0.1, 0.2, 0.5, 0.8];
    // let output = vec![0.1, 0.2, 0.5, 0.8, 0.7, 0.7, 0.2, 0.4];
    // let mut new_model = Model::new("test", &MSE, &he_initializer);

    // new_model.add_layer(&SGD, 4, 4, IN(input), &Sigmoid, Some("input"));
    // new_model.add_layer(&SGD, 4, 4, HIDDEN, &Sigmoid, Some("input"));
    // new_model.add_layer(&SGD, 4, 8, HIDDEN, &Sigmoid, Some("input"));
    // new_model.add_layer(&SGD, 8, 0, OUT(output), &Sigmoid, Some("output"));
    // new_model.train(500, 0.2, true, 10);

    test_excution_time(|| {
        let _ = Matrix::<f64>::ones(vec![1, 5, 4, 5]);
    });

    test_excution_time(|| {
        let _ = ArrayD::<f64>::ones(IxDyn(&[1, 5, 4, 5]));
    });

}

use crate::maths::c_num_traits::{Zero, One};
fn test<T: Debug + Copy + Zero>(index: &[T]) {
    let mut a = vec![T::zero(); index.len()];
    a.copy_from_slice(index);
    println!("i {:?}", a);
}

// 2, 3, 4
