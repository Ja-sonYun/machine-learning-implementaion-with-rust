#[cfg(test)]
mod matrix {
    use ml::maths::n_matrix::{Matrix};

    #[test]
    fn matrix_set_and_get() {
        let mut a = Matrix::<f64>::fill_with(5., vec![2, 3, 4, 5]);
        a.set(vec![1, 0, 2, 0], 2.);
        assert_eq!(a.get(vec![1, 0, 2, 0]).unwrap(), 2.);
        assert_eq!(a.get(vec![1, 0]).get(vec![2, 0]).unwrap(), 2.);
    }

    // fn matrix_index() {
    //     let matrix = Matrix::<f64>::new(3, 4);
    // }
}
