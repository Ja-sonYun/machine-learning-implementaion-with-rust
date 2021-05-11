#[cfg(test)]
mod matrix {
    use ml::maths::matrix::{Matrix};
    #[test]
    fn validate_zero() {
        let w1 = Matrix::<f64>::new(3, 4);
        let mut w = Matrix::<f64>::new(3, 4);
        let mut a = Matrix::<f64>::new_scalar(4.);
        let mut b = Matrix::<f64>::new_scalar(2.);
        let mat = || Matrix::<f64>::new(3, 4);
        let matmat = || Matrix::<Matrix<f64>>::from_fn(mat, 3, 3);
        let matmatmat = || Matrix::<Matrix<Matrix<f64>>>::from_fn(matmat, 3, 3);
        let matmatmatmat = || Matrix::<Matrix<Matrix<Matrix<f64>>>>::from_fn(matmatmat, 3, 3);
        let matmatmatmatmat = Matrix::<Matrix<Matrix<Matrix<Matrix<f64>>>>>::from_fn(matmatmatmat, 3, 3);
        w.set(1, 3, 4.);
        assert_eq!(matmatmatmatmat.is_zero(), true);
    }

    // fn create_matrix() {
    //     let matrix = Matrix::<f64>::new(3, 4);
    // }
}
