use std::ops::{Add, Mul, Sub, Div, IndexMut, Index};
use std::fmt::{Display, Debug, Formatter, Result};
use crate::forfor;
use crate::maths::c_num_traits::{Zero, One};
use std::cmp::{PartialEq};
use std::ptr;

#[derive(Clone)]
#[allow(non_snake_case)]
pub struct Vector<T> where T: Zero + One + Copy {
    __from: Option<Vec<T>>,
    __to: Option<Vec<T>>,
    __i: Vec<T>,
    __k: T,
    __R: usize,
}

impl<T> Vector<T> where T: Zero + One + Copy + Sub<Output = T> {
    pub fn new_dir(__from: Vec<T>, __to: Vec<T>) -> Self {
        let length = __from.len();
        if length != __to.len() { panic!("dimension mismatch!") }
        let mut __i = Vec::<T>::with_capacity(length);
        for i in 0..length {
            __i.push(__to[i] - __from[i]);
        }
        Vector { __R: length, __from: Some(__from), __to: Some(__to), __i, __k: T::one() }
    }

    pub fn new(__i: Vec<T>) -> Self {
        Vector { __R: __i.len(), __k: T::one(), __i, __from: None, __to: None }
    }

    // pub fn norm(&self) -> T {
        
    // }
}

impl<T> PartialEq<Vector<T>> for Vector<T> where T: Zero + One + PartialEq + Copy {
    fn eq(&self, other: &Self) -> bool {
        self.__i == other.__i
    }
    fn ne(&self, other: &Self) -> bool {
        self.__i == other.__i
    }
}

// implement matrix calculate
macro_rules! opt_impl {
    ($funcn:ident, $func:ident, $c:expr) => {
        impl<T> $funcn<Vector<T>> for Vector<T> where T: Zero + One + Copy + $funcn<Output = T> {
            type Output = Self;
            #[inline]
            fn $func(self, rhs: Self) -> Self {
                if self.__R != rhs.__R {
                    panic!("mismatch dimension!")
                }
                let mut __i = Vec::<T>::with_capacity(self.__R);
                for i in 0..self.__R {
                    __i.push($c(self.__i[i], rhs.__i[i]));
                }
                Vector { __R: self.__R, __k: self.__k, __i, __from: None, __to: None }
            }
        }
    }
}
opt_impl!(Add, add, |a, b| a + b);
opt_impl!(Mul, mul, |a, b| a * b);
opt_impl!(Sub, sub, |a, b| a - b);
opt_impl!(Div, div, |a, b| a / b);
