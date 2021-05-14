extern crate ml;
use ml::*;
use tools::*;

use time::OffsetDateTime;
use ndarray::{IxDyn, ArrayD};

use activations::*;
use initializer::*;
use cost::*;
// use maths::matrix::*;
use maths::n_matrix::*;
use maths::matrix::Matrix_d;
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

    // a.set(vec![1, 0, 2, 0], 2.);

    // a.set(vec![1, 0, 2, 0], 2.);
    test_excution_time(|| {
        let _ = Matrix::<f64>::ones(vec![1, 300]).get(vec![0, 2]);
    });

    test_excution_time(|| {
        let _ = ArrayD::<f64>::ones(IxDyn(&[1, 300]))[[0, 2]];
    });
    // let mut a = Matrix::<f64>::zeros(vec![100, 300, 100, 4, 5, 8, 8]);
    // println!("a: {}", a.get(vec![1, 0, 2, 0]));



    // a[1][0][2][0] = 2.;
    // let now2 = OffsetDateTime::now_utc();
    // let mut a = vec![vec![vec![vec![vec![0; 8]; 8]; 4]; 300]; 100];
    // println!("a: {}", a[1][0][2][0]);
    // println!("time: {:?}", OffsetDateTime::now_utc() - now2);
    // let mut a = 100;

    // let now2 = OffsetDateTime::now_utc();
    // let mut b = ArrayD::<f64>::zeros(IxDyn(&[100, 300, 1000, 4, 5, 8, a]));
    // a[1][0][2][0] = 2.;
    // println!("a: {}", a[1][0][2][0]);
    // println!("time: {:?}", OffsetDateTime::now_utc() - now2);


    // vector index => 0 is x, 1 is y, 2 is z...
    // let mut y = Matrix::<f64>::fill_with(1., vec![2, 3]);
    // println!("a: {}", y.get(vec![0]));


}

fn test_excution_time<F: Fn()>(f: F) {
    let now2 = OffsetDateTime::now_utc();
    f();
    // a[1][0][2][0] = 2.;
    // println!("a: {}", a[1][0][2][0]);
    println!("time: {:?}", OffsetDateTime::now_utc() - now2);
}

// 2, 3, 4
