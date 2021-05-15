#[cfg(test)]
mod matrix {
    use ml::maths::n_matrix::{Matrix};
    use tools::test_tools::test_excution_time;

    // #[test]
    // fn matrix_set_and_get() {
    //     let mut a = Matrix::<f64>::fill_with(5., vec![2, 3, 4, 5]);
    //     a.set(vec![1, 0, 2, 0], 2.);
    //     assert_eq!(a.get(vec![1, 0, 2, 0]).unwrap(), 2.);
    //     assert_eq!(a.get(vec![1, 0]).get(vec![2, 0]).unwrap(), 2.);
    // }

    #[test]
    fn matrix_elementwise_calculate() {
        let a = Matrix::<i64>::from(vec![1, 3],
                                    vec![1,
                                        5,
                                        9]);

        let b = Matrix::<i64>::from(vec![1, 3],
                                    vec![1,
                                        5,
                                        9]);

        let expect = Matrix::<i64>::from(vec![1, 3], vec![1,25,81]);
        assert_eq!(a * b, expect);
    }

    #[test]
    fn matrix_scalar_calculate() {
        let a = Matrix::<i64>::from(vec![1, 3],
                                    vec![1,
                                        5,
                                        9]);

        let b = Matrix::<i64>::new_scalar(4);

        let expect = Matrix::<i64>::from(vec![1, 3], vec![4,20,36]);
        assert_eq!(a * b, expect);
    }


    // fn matrix_index() {
    //     let matrix = Matrix::<f64>::new(3, 4);
    // }
}
