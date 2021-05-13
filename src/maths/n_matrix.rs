use std::ops::{Add, Mul, Sub, Div, IndexMut, Index};
use std::fmt::{Display, Debug, Formatter, Result};
use crate::forfor;
use crate::maths::c_num_traits::{Zero, One};

#[derive(Clone)]
pub struct Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug {
    __ndarray: Vec<T>,
    __ndarray_size: usize,
    __dimension: Vec<usize>,
    __depth: usize,
    __inited: bool,
    ___is_query_stacked: bool,
    ___last_query: Option<Vec<usize>>,
}

impl<T> Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug {
    #[inline]
    fn _new(__ndarray_size: usize, __ndarray: Vec<T>, __depth: usize, __dimension: Vec<usize>, __inited: bool) -> Matrix<T> {
        Matrix {
            __ndarray: __ndarray,
            __ndarray_size: __ndarray_size,
            __depth: __depth,
            __dimension: __dimension,
            __inited: __inited,
            ___is_query_stacked: false,
            ___last_query: None,
        }
    }

    pub fn new(_dimension: Vec<usize>) -> Matrix<T> {
        if _dimension.len() == 0 { panic!("cannot create 0-dimension matrix") };
        let mut array_size = 1;
        for a in _dimension.iter() { array_size *= a }
        Matrix::_new(array_size, Vec::<T>::with_capacity(array_size), _dimension.len(), _dimension, false)
    }

    fn _new_from_raw_vec(_dimension: Vec<usize>, ndarray: Vec<T>) -> Matrix<T> {
        let mut temp = Matrix::<T>::new(_dimension);
        temp._push_ndarray(ndarray);
        temp
    }

    pub fn new_scalar(value: T) -> Matrix<T> {
        Matrix::_new(1, vec![value; 1], 0, vec![1; 1], true)
    }

    pub fn is_scalar(&self) -> bool {
        self.__ndarray_size == 1 && self.__depth == 0
    }

    fn _cal_elemwise<F: Fn(T, T) -> T>(&self, with: Self, cal_fn: F) -> Self {
        if self.cmp_dim_with(&with) {
            let mut temp = Matrix::<T>::new(self._raw_dim());
            for i in 0..self._ndarray_size() {
                temp.__ndarray.push(cal_fn(self._ndarray(i), with._ndarray(i)));
            }
            temp
        } else {
            panic!("Dimensions are different, so that cannot calculate")
        }
    }

    fn _cal_with_scalar<F: Fn(T, T) -> T>(&self, with: Self, cal_fn: F) -> Self {
        if with.is_scalar() {
            let mut temp = Matrix::<T>::new(self._raw_dim());
            for i in 0..self._ndarray_size() {
                temp.__ndarray.push(cal_fn(self._ndarray(i), with._ndarray(0)));
            }
            temp
        } else {
            panic!("not scalar!")
        }
    }

    pub fn cmp_dim_with(&self, with: &Self) -> bool {
        self._raw_dim() == with._raw_dim()
    }

    // Matrix to T if matrix is scalar
    pub fn unwrap(&self) -> T {
        if self.__depth == 0 {
            self._ndarray(0)
        } else {
            panic!("cannot unwrap because matrix is not scalar.")
        }
    }

    pub fn zeros(_dimension: Vec<usize>) -> Matrix<T> {
        let mut temp = Matrix::<T>::new(_dimension.clone());
        temp._fill_with(T::zero());
        temp
    }

    pub fn ones(_dimension: Vec<usize>) -> Matrix<T> {
        let mut temp = Matrix::<T>::new(_dimension.clone());
        temp._fill_with(T::one());
        temp
    }

    pub fn fill_with(_element: T, _dimension: Vec<usize>) -> Matrix<T> {
        let mut temp = Matrix::<T>::new(_dimension);
        temp._fill_with(_element);
        temp
    }

    fn _fill_with(&mut self, _element: T) {
        for _ in 0..self._ndarray_size() {
            self.__ndarray.push(_element);
        }
        self.__inited = true;
    }

    fn _get_cell(&self, query: &mut Vec<usize>) -> Vec<T> {
        if query.len() + 1 != self._depth() {
            panic!("query isn't deep enough to get whole cell")
        }
        query.push(0);
        // get
        let offset = self._get_index(&query);
        let cell_size = self._cell_size();
        // let mut cell = Vec::<T>::with_capacity(22);
        let mut cell = vec![T::zero(); cell_size];
        cell.copy_from_slice(&self._ndarray_r(offset, offset+cell_size));
        cell
    }

    pub fn get(&self, query: Vec<usize>) -> Matrix<T> {
        let query_len = query.len();
        if query_len == self._depth() {
            // return single element
            Matrix::<T>::new_scalar(self._ndarray(self._get_index(&query)))
        } else {
            let dim_diff = self._depth() - query_len;
            let mut new_dim = vec![0; dim_diff];
            new_dim.copy_from_slice(&self._dimension_r(query_len, self._depth()));

            let part_area = self._get_offset(&query);
            let mut new_array = vec![T::zero(); part_area.1];
            new_array.copy_from_slice(&self._ndarray_r(part_area.0, part_area.0+part_area.1));

            Matrix::<T>::_new(part_area.1,new_array, dim_diff, new_dim, true)
        }
    }

    fn _borrow_partial(&self, query: Vec<usize>) -> Matrix<T> {
        let query_len = query.len();
        if query_len == self._depth() {
            // return single element
            Matrix::<T>::new_scalar(self._ndarray(self._get_index(&query)))
        } else {
            let dim_diff = self._depth() - query_len;
            let mut new_dim = vec![0; dim_diff];
            new_dim.copy_from_slice(&self._dimension_r(query_len, self._depth()));

            let part_area = self._get_offset(&query);
            let mut new_array = vec![T::zero(); part_area.1];
            new_array.copy_from_slice(&self._ndarray_r(part_area.0, part_area.0+part_area.1));

            Matrix::<T>::_new(part_area.1, new_array, dim_diff, new_dim, true)
        }
    }


    // return  ( offset, length )
    fn _get_offset(&self, query: &Vec<usize>) -> (usize, usize) {
        let mut length = 1;
        let mut owned_query = query.clone();
        for i in (query.len()..self._depth()).rev() {
            owned_query.push(0);
            length *= self._dimension(i);
        }
        let offset = self._get_index(&owned_query);
        (offset, length)
    }

    pub fn set(&mut self, query: Vec<usize>, val: T) {
        let index = self._get_index(&query);
        self._set_ndarray(index, val);
    }

    #[inline]
    pub fn _get_index(&self, query: &Vec<usize>) -> usize {
        let mut index = query[query.len() - 1];
        for i in 0..query.len()-1 {
            let mut tmp = 1;
            for j in i+1..query.len() {
                tmp *= self._dimension(j);
            }
            index += query[i] * tmp;
        }
        index
    }

/////////////////////////////////////////
//
// GETTERS
//

    #[inline]
    fn _ndarray(&self, index: usize) -> T {
        self.__ndarray[index]
    }

    #[inline]
    fn _ndarray_r(&self, from: usize, to: usize) -> &[T] {
        &self.__ndarray[from..to]
    }

    #[inline]
    fn _ndarray_size(&self) -> usize {
        self.__ndarray_size
    }

    #[inline]
    fn _depth(&self) -> usize {
        self.__depth
    }

    #[inline]
    fn _raw_dim(&self) -> Vec<usize> {
        self.__dimension.clone()
    }

    #[inline]
    fn _dimension(&self, index: usize) -> usize {
        self.__dimension[index]
    }

    #[inline]
    fn _dimension_r(&self, from: usize, to: usize) -> &[usize] {
        &self.__dimension[from..to]
    }

    #[inline]
    fn _cell_size(&self) -> usize {
        self.__dimension[self.__depth-1]
    }

/////////////////////////////////////////
//
// SETTERS
//

    #[inline]
    fn _push_ndarray(&mut self, new_ndarray: Vec<T>) {
        for e in new_ndarray.iter() {
            self.__ndarray.push(*e);
        }
    }

    #[inline]
    fn _set_dimension(&mut self, index: usize, value: usize) {
        self.__dimension[index] = value;
    }

    #[inline]
    fn _set_ndarray(&mut self, index: usize, value: T) {
        self.__ndarray[index] = value;
    }
}

impl<T> Display for Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug {
    // todo: formatting
    fn fmt(&self, f: &mut Formatter) -> Result {
        if self._ndarray_size() == 1 {
            return write!(f, "{}", self._ndarray(0))
        }
        let mut strg = "".to_owned();
        strg.push_str(&format!("\ndimension: {:?}, depth: {}\n", self.__dimension, self.__depth));
        for i in 0..self._ndarray_size() {
            strg.push_str(&format!(" {} ", &self._ndarray(i)));
        }

        // let mut strg = "\n[".to_owned();
        // for i in 0..self.__ndarray_size {
        //     for dim in &self.__dimension {
        //         if i % *dim == 1 {
        //             strg.push_str("]\n[");
        //         }
        //     }
        //     strg.push_str(&format!(" {} ", &self.__ndarray[i]));
        // }
        write!(f, "{}", strg)
    }
}


// implement matrix calculate
macro_rules! opt_impl {
    ($funcn:ident, $func:ident, $c:expr) => {
        impl<T> $funcn<Matrix<T>> for Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug + $funcn<Output = T> {
            type Output = Self;
            #[inline]
            fn $func(self, rhs: Self) -> Self {
                self._cal_with_scalar(rhs, $c)
            }
        }
    }
}
opt_impl!(Add, add, |a, b| a + b);
opt_impl!(Mul, mul, |a, b| a * b);
opt_impl!(Sub, sub, |a, b| a - b);
opt_impl!(Div, div, |a, b| a / b);

// impl<T> Index<usize> for Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug {
//     type Output = Matrix<T>;
//     fn index(&self, _: usize) -> &Self::Output {
//         panic!("Matrix should be mutable!")
//     }
// }

// impl<T> IndexMut<usize> for Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug {
//     fn index_mut(&mut self, i: usize) -> &mut Self::Output {
//         match &mut self.___last_query {
//             Some(val) => val.push(i),
//             None => self.___last_query = Some(vec![i])
//         };
//         println!("i;{:?}", self.___last_query);
//         self
//     }
// }
