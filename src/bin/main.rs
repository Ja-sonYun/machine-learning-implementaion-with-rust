extern crate ml;
use ml::*;
use tools::*;
use tools::test_tools::test_excution_time;
use std::fmt::{Display, Debug, Formatter, Result};

use std::fs::File;
use reader::png::create_matrix_from_image;

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
    //
    //
    let a = Matrix::<i64>::from(vec![3, 4], vec![1,  2,  3,  4,
                                                 5,  6,  7,  8,
                                                 9, 10, 11, 12]);
    println!("  a = {}", a);
    println!("   a.get() -> {}", a.get(vec![2, 1]));

    test_excution_time(|| {
        let _ = Matrix::<f64>::ones(vec![1, 5, 4, 5, 5, 5]).get(vec![0, 3, 2, 2, 2, 2]);
    });

    test_excution_time(|| {
        let _ = ArrayD::<f64>::ones(IxDyn(&[1, 5, 4, 5, 5, 5]))[[0, 3, 2, 2, 2, 2]];
    });

    let num = create_matrix_from_image("src/32011.png");
    for i in 0..23 {
        println!("{}", num.get(vec![i]));
    }

    // let decoder = png::Decoder::new(File::open("src/32011.png").unwrap());
    // let (info, mut reader) = decoder.read_info().unwrap();
    // // Allocate the output buffer.
    // let mut buf = vec![0; info.buffer_size()];
    // reader.next_frame(&mut buf).unwrap();
    // println!("decode{:?}", buf)
}

// fn caching_dim(dim: Vec<usize>) -> Vec<usize> {
//     let mut tmp = 1;
//     let mut tmpv = vec![0;dim.len()];
//     tmpv[dim.len()-1] = 1;
//     for i in (1..dim.len()).rev() {
//         tmp *= dim[i];
//         tmpv[i+1] = tmp;
//     }
//     tmpv
// }

use crate::maths::c_num_traits::{Zero, One};
fn test<T: Debug + Copy + Zero>(index: &[T]) {
    let mut a = vec![T::zero(); index.len()];
    a.copy_from_slice(index);
    println!("i {:?}", a);
}

// 2, 3, 4
