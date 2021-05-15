use std::ops::{Add, Mul, Sub, Div};
use std::fmt::{Display, Debug, Formatter, Result};
use crate::maths::c_num_traits::{Zero, One};
use std::cmp::{PartialEq};

//
//  1,  2,  3
//  4,  5,  6
//  7,  8,  9
//  10, 11, 12
//
//
//

#[derive(Clone)]
pub enum Dimension {
    Not(usize),
    Transpose(usize),
}
impl Dimension {
    fn vec_to_dims(this: Vec<usize>) -> Vec<Dimension> {
        let mut iter = this.iter();
        (0..this.len()).map(|_| Dimension::Not(match iter.next() {
            Some(val) => *val,
            None => panic!("something went wrong")
        })).collect()
    }
}

#[derive(Clone)]
pub struct Matrix<T> where T: Zero + One + Copy {
    __ndarray: Vec<T>,
    __ndarray_size: usize,
    __dimension: Vec<Dimension>,
    __depth: usize,
    __inited: bool,
    ___is_query_stacked: bool,
    ___last_query: Option<Vec<usize>>,
    ___is_mutated: Option<Vec<usize>>,
}

impl<T> Matrix<T> where T: Zero + One + Copy {
    #[inline]
    fn _new(__ndarray_size: usize, __ndarray: Vec<T>, __depth: usize, __dimension: Vec<usize>, __inited: bool) -> Matrix<T> {
        // Matrix::<T>::_mapping(&__dimension, __depth);
        Matrix {
            __ndarray,
            __ndarray_size,
            __depth,
            __dimension: Dimension::vec_to_dims(__dimension),
            __inited,
            ___is_query_stacked: false,
            ___last_query: None,
            ___is_mutated: None,
        }
    }

    pub fn new(_dimension: Vec<usize>) -> Matrix<T> {
        if _dimension.is_empty() { panic!("cannot create 0-dimension matrix") };
        let mut array_size = 1;
        for a in _dimension.iter() { array_size *= a }
        Matrix::_new(array_size, Vec::<T>::with_capacity(array_size), _dimension.len(), _dimension, false)
    }

    pub fn fill_with(_dimension: Vec<usize>, _element: T) -> Matrix<T> {
        if _dimension.is_empty() { panic!("cannot create 0-dimension matrix") };
        let mut array_size = 1;
        for a in _dimension.iter() { array_size *= a }
        Matrix::_new(array_size, vec![_element; array_size], _dimension.len(), _dimension, true)
    }

    pub fn from(_dimension: Vec<usize>, ndarray: Vec<T>) -> Matrix<T> {
        Matrix::_new(ndarray.len(), ndarray, _dimension.len(), _dimension, true)
    }

    // pub fn from_vec(rows: Box::) {

    // }

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
        let mut iter = self.__ndarray.iter();
        let temp = Matrix::<T>::from(self._raw_dim(), (0..self.__ndarray_size).map(|_| cal_fn(match iter.next() {
            Some(val) => *val,
            None => panic!("something goes wrong!")
        }, with._ndarray(0))).collect());
        temp
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
        Matrix::<T>::fill_with(_dimension, T::zero())
    }

    pub fn ones(_dimension: Vec<usize>) -> Matrix<T> {
        Matrix::<T>::fill_with(_dimension, T::one())
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

    fn _get_dim_by_query(&self, _query_depth: usize) -> Vec<usize> {
        let dim_diff = self._depth() - _query_depth;
        let mut new_dim = vec![0; dim_diff];
        new_dim.copy_from_slice(&self._dimension_r(_query_depth, self._depth()));
        new_dim
    }

    pub fn mut_by(&mut self, query: Vec<usize>) {
        self.___is_mutated = Some(query);
    }

    fn _borrow_partial(&self, query: Vec<usize>) {//-> Matrix<T> {
        //TODO:
    }


    // return  ( offset, length )
    fn _get_offset(&self, query: &[usize]) -> (usize, usize) {
        let mut length = 1;
        let mut owned_query = query.to_owned();
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

    // pub fn _mapping(_dimension: &Vec<usize>, _depth: usize) -> Vec<usize> {
    //     let temp_vec = Vec::<usize>::with_capacity(_depth);
    //     let mut index;
    //     let mut tmp;
    //     for i in 0.._depth - 1 {
    //         tmp = 1;
    //         for j in i+1.._depth {
    //             tmp *= _dimension[j];
    //         }
    //         index += 
    //         temp_vec.push()
    //     }
    //     _dimension.iter().map(|dim| {
    //         for i in dim.._depth {
    //             tmp *= 
    //         }
    //     }).collect()
    // }

    #[inline]
    pub fn _get_index(&self, query: &[usize]) -> usize {
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

/////////////////////////////////////////
//
// CAL
//
// impl<T> Matrix<T> where T: Zero + One + Copy + Mul <Output = T>{
//     pub fn e_mul(&self, with: Self) -> Self {
//         self._cal_elemwise(with, |a, b| a * b)
//     }

//     // pub fn e_mul(&self, with: Self) -> Self {
//     //     self._cal_elemwise(with, |a, b| a * b)
//     // }
// }

impl<T> Display for Matrix<T> where T: Zero + One + Copy + Display {
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

impl<T> Debug for Matrix<T> where T: Zero + One + Copy + Display {
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
        write!(f, "{}", strg)
    }
}

impl<T> PartialEq<Matrix<T>> for Matrix<T> where T: Zero + One + PartialEq + Copy {
    fn eq(&self, other: &Self) -> bool {
        self.__ndarray == other.__ndarray && self.__dimension == other.__dimension
    }
    fn ne(&self, other: &Self) -> bool {
        self.__ndarray == other.__ndarray && self.__dimension == other.__dimension
    }
}

// implement matrix calculate
macro_rules! opt_impl {
    ($funcn:ident, $func:ident, $c:expr) => {
        impl<T> $funcn<Matrix<T>> for Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug + $funcn<Output = T> {
            type Output = Self;
            #[inline]
            fn $func(self, rhs: Self) -> Self {
                if self.cmp_dim_with(&rhs) {
                    self._cal_elemwise(rhs, $c)
                } else if rhs.is_scalar() || self.is_scalar() {
                    self._cal_with_scalar(rhs, $c)
                } else {
                    panic!("mismatch dimensions, cannot calcuate");
                }
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
