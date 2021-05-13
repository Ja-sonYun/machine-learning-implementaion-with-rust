use std::ops::{Add, Mul, Sub, Div};
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
}

// dimension would be like -> [ 2, 2, 3 ] which is same as
// [0] -> x axis, [1] -> y axis, [2] -> z axis
//
//  [ [ [0, 0, 0] , [0, 0, 0] ],
//    [ [0, 0, 0] , [0, 0, 0] ],
//    [ [0, 0, 0] , [0, 0, 0] ] ]
//
// dimension would be like -> [ 3, 4 ] which is same as
// [0] -> x axis, [1] -> y axis, [2] -> z axis
//
//  [ [ 0, 0, 0 ],
//    [ 0, 0, 0 ],
//    [ 0, 0, 0 ],
//    [ 0, 0, 0 ] ]
//
// query: [0][0]    -> 0, dim [3, 4]
// query: [0][1]    -> 1, dim [3, 4]
// query: [0][2]    -> 2, dim [3, 4]
// query: [1][0]    -> 3, dim [3, 4]
// query: [1][1]    -> 4, dim [3, 4]
// query: [1][2]    -> 5, dim [3, 4]
// query: [2][2]    -> 9, dim [3, 4]
//
// query: [0][0] -> 2, dim [2, 2, 3]
// query: [0][1] -> 3, dim [2, 2, 3]
// query: [0][1] -> 5, dim [2, 2, 3]
// query: [1][0] -> 9, dim [2, 2, 3]
// query: [1][1] -> 11,dim [2, 2, 3]
impl<T> Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug {
    pub fn new(_dimension: Vec<usize>) -> Matrix<T> {
        if _dimension.len() == 0 { panic!("cannot create 0-dimension matrix") };
        let mut array_size = 1;
        for a in _dimension.iter() { array_size *= a }
        Matrix { __ndarray: Vec::<T>::with_capacity(array_size),
            __ndarray_size: array_size,
            __depth: _dimension.len(),
            __dimension: _dimension,
            __inited: false
        }
    }

    fn _new_from_raw_vec(_dimension: Vec<usize>, ndarray: Vec<T>) -> Matrix<T> {
        let mut temp = Matrix::<T>::new(_dimension);
        temp._set_ndarray(ndarray);
        temp
    }

    fn _new_scalar(value: T) -> Matrix<T> {
        Matrix {
            __ndarray: vec![value; 1],
            __ndarray_size: 1,
            __depth: 0,
            __dimension: vec![1; 1],
            __inited: true
        }
    }

    pub fn zeros(_dimension: Vec<usize>) -> Matrix<T> {
        let mut temp = Matrix::<T>::new(_dimension.clone());
        temp.fill_with(T::zero());
        temp
    }

    pub fn ones(_dimension: Vec<usize>) -> Matrix<T> {
        let mut temp = Matrix::<T>::new(_dimension.clone());
        temp.fill_with(T::one());
        temp
    }

    pub fn fill_with(&mut self, _element: T) {
        for _ in 0..self.__ndarray_size {
            self.__ndarray.push(_element);
        }
        self.__inited = true;
    }

    pub fn _get_cell(&self, query: &mut Vec<usize>) -> Vec<T> {
        if query.len() + 1 != self._depth() {
            panic!("query isn't deep enough to get whole cell")
        }
        query.push(0);
        // get
        let offset = self._get_index(&query);
        let cell_size = self._cell_size();
        // let mut cell = Vec::<T>::with_capacity(22);
        let mut cell = vec![T::zero(); cell_size];
        cell.copy_from_slice(&self.__ndarray[offset..offset+cell_size]);
        cell
    }


    pub fn get(&self, query: Vec<usize>) -> Matrix<T> {
        let query_len = query.len();
        if query_len == self._depth() {
            // return single element
            Matrix::<T>::_new_scalar(self.__ndarray[self._get_index(&query)])
        } else {
            let dim_diff = self._depth() - query_len;
            let mut new_dim = vec![0; dim_diff];
            new_dim.copy_from_slice(&self.__dimension[query_len..self._depth()]);

            let part_area = self._get_offset(&query);
            let mut new_array = vec![T::zero(); part_area.1];
            new_array.copy_from_slice(&self.__ndarray[part_area.0..part_area.0+part_area.1]);

            Matrix { __ndarray: new_array,
                     __ndarray_size: part_area.1,
                     __depth: dim_diff,
                     __dimension: new_dim,
                     __inited: true
            }
        }
    }

    // return  ( offset, length )
    pub fn _get_offset(&self, query: &Vec<usize>) -> (usize, usize) {
        let mut length = 1;
        let mut owned_query = query.clone();
        for i in (query.len()..self._depth()).rev() {
            owned_query.push(0);
            length *= self._dimension(i);
        }
        let offset = self._get_index(&owned_query);
        (offset, length)
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

    pub fn set(&mut self, query: Vec<usize>, val: T) {
        let mut index = query[query.len() - 1];
        for i in 0..query.len()-1 {
            let mut tmp = 1;
            for j in i+1..query.len() {
                tmp *= self._dimension(j);
            }
            index += query[i] * tmp;
        }
        self.__ndarray[index] = val;
    }

//
// GETTERS
//

    #[inline]
    fn _ndarray_size(&self) -> usize {
        self.__ndarray_size
    }

    #[inline]
    fn _depth(&self) -> usize {
        self.__depth
    }

    #[inline]
    fn _dimension(&self, index: usize) -> usize {
        self.__dimension[index]
    }

    #[inline]
    fn _cell_size(&self) -> usize {
        self.__dimension[self.__depth-1]
    }

//
// SETTERS
//

    #[inline]
    fn _set_ndarray(&mut self, new_ndarray: Vec<T>) {
        for e in new_ndarray.iter() {
            self.__ndarray.push(*e);
        }
    }
}

impl<T> Display for Matrix<T> where T: Zero + One + PartialEq + Copy + Display + Debug {
    fn fmt(&self, f: &mut Formatter) -> Result {
        if self._ndarray_size() == 1 {
            return write!(f, "{}", self.__ndarray[0])
        }
        let mut strg = "".to_owned();
        strg.push_str(&format!("\ndimension: {:?}, depth: {}\n", self.__dimension, self.__depth));
        for i in 0..self._ndarray_size() {
            strg.push_str(&format!(" {} ", &self.__ndarray[i]));
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


