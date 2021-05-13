// use std::collections::HashMap;
// pub struct Expression<EXP: Fn(Vec<f64>) -> f64> {
//     exp: EXP,
//     map: HashMap<String, i64>
// }
// impl<EXP: Fn(Vec<f64>) -> f64> Expression<EXP> {
//     fn new()
// }
use std::ops::{Add, Mul, Sub, Div};
use std::fmt::{Display, Debug, Formatter, Result};
use crate::forfor;
use crate::maths::c_num_traits::{Zero, One};

// type DebugElement = Zero + Display + Copy;

// use crate::utils::maths::derivative;
// #[derive(Clone, Copy)]
// enum Element<T> {
//     Zero,
//     Num(T)
// }
// impl<T> Element<T> {
//     fn num(self) -> T where T: Zero + Display + Copy {
//         match self {
//             Element::Num(v) => v,
//             _ => Zero::zero(),
//         }
//     }
// }
// impl<T> Debug for Element<T> where T: Zero + Display + Copy {
//     fn fmt(&self, f: &mut Formatter) -> Result {
//         write!(f, "{}", self.num())
//     }
// }

#[derive(Clone)]
pub struct Matrix<T> where T: Zero + One {
    n: Vec<Vec<T>>,
    _shape: (i64, i64),
    _d_shape: Option<Vec<i64>>
}

impl<T> Matrix<T> where T: Clone + Zero + One {

    #[inline]
    pub fn new(ny: i64, nx:i64) -> Matrix<T> {
        Matrix { n: (0..ny).map(|_| vec![T::zero(); nx as usize]).collect(), _shape: (ny, nx), _d_shape: None }
    }

    // shape[0] -> y, 1 -> x, 2 -> z, 3 -> ...
    pub fn zeros(d_shape: Vec<i64>) -> Matrix<T> {
        // y vector
        let mut shape_size = d_shape.len();
        Matrix { n: vec![vec![T::zero(); shape_size]; d_shape[0] as usize], _shape:(0, d_shape[0]), _d_shape: None }
        // if shape_size > 0 {
        //     if shape_size == 1 {
        //         Matrix { n: vec![vec![T::zero(); shape_size]; d_shape[0] as usize], _shape:(0, d_shape[0]), _d_shape: None }
        //     } else {
        //         Matrix { n: vec![vec![T::zero(); shape_size]; d_shape[0] as usize], _shape:(0, d_shape[0]), _d_shape: None }
        //     }
        // } else {
        //     Matrix { n: vec![vec![T::zero(); shape_size]; d_shape[0] as usize], _shape:(0, d_shape[0]), _d_shape: None }
        // }
        // Matrix { n: (0..) }
    }

    // fn _create_deeply<F: Fn()->T>(mat: Matrix<T>, l_d_shape: Vec<i64>, shape_size: usize, init_f: F) -> Matrix<T> {
    //     if shape_size == 0 {
            
    //     }
    // }

    // pub fn zeros(dims: Vec<T>) -> Matrix<T> {

    // }

    // #[inline]
    // pub fn new_d()

    #[inline]
    pub fn new_scalar(val: T) -> Matrix<T> {
        Matrix { n: vec![vec![val]], _shape: (1, 1), _d_shape: None }
    }

    #[inline]
    fn is_scalar(&self) -> bool {
        self._shape == (1, 1)
    }

    #[inline]
    pub fn y(&self) -> i64 {
        self._shape.0
    }

    #[inline]
    pub fn x(&self) -> i64 {
        self._shape.1
    }

    #[inline]
    pub fn from_fn<F: Fn()->T>(init_fn: F, ny: i64, nx:i64) -> Matrix<T> {
        Matrix { n: (0..ny).map(|_| (0..nx).map(|_| init_fn()).collect()).collect(), _shape: (ny, nx), _d_shape: None }
    }

    #[inline]
    pub fn get(&self, y: i64, x: i64) -> T {
        self.n[y as usize][x as usize].clone()
    }

    #[inline]
    fn zero() -> Matrix<T> {
        Matrix::<T>::new(1, 1)
    }
    pub fn s_get(&self) -> T {
        if self.is_scalar() {
            self.get(0, 0)
        } else {
            panic!("this is matrix");
        }
    }

    #[inline]
    pub fn get_shape(&self) -> (i64, i64) {
        self._shape
    }

    #[inline]
    pub fn comp_dim_with(&self, shr: &Self) -> bool {
        self.get_shape() == shr.get_shape()
    }
    pub fn set(&mut self, y: i64, x: i64, val: T) {
        if self.is_scalar() {
            panic!("this is scalar")
        }
        self.n[y as usize][x as usize] = val;
    }
    pub fn s_set(&mut self, val: T) {
        if self.is_scalar() {
            self.n[0][0] = val;
        } else {
            panic!("This is a matrix not the scalar.");
        }
    }
    pub fn elem_with_scalar<F: Fn(T, T)->T>(&self, with: T, cal_fn: F) -> Self {
        // TODO: Refactoring this, use map?
        let mut temp = Matrix::<T>::new(self.y(), self.x());
        forfor!(self.y(), y, self.x(), x, {
            temp.set(y, x, cal_fn(self.get(y, x), with.clone()));
        });
        temp
    }
    pub fn elemwise_cal<F: Fn(T, T)->T>(&self, with: Self, cal_fn: F) -> Self {
        let mut temp = Matrix::<T>::new(self.y(), self.x());
        forfor!(self.y(), y, self.x(), x, {
            temp.set(y, x, cal_fn(self.get(y, x), with.get(y, x)));
        });
        temp
    }
    pub fn cal_with_scalar<F: Fn(T, T)->T>(this: Self, andthis: Self, cal_fn: F) -> Self {
        if this.is_scalar() {
            andthis.elem_with_scalar(this.s_get(), cal_fn)
        } else if andthis.is_scalar() {
            this.elem_with_scalar(andthis.s_get(), cal_fn)
        } else if this.comp_dim_with(&andthis) {
            this.elemwise_cal(andthis, cal_fn)
        } else {
            panic!("can't calculate with this");
        }
    }
    pub fn comp_elem<F: Fn(T)->bool>(&self, cal_fn: F) -> bool {
        forfor!(self.y(), y, self.x(), x, {
            if !cal_fn(self.get(y, x)) {
                return false
            }
        });
        true
    }
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.comp_elem(|x| x.is_zero())
    }
    // pub fn derivative(&mut self) -> Matrix<f64> {
    // }
}

impl<T> Zero for Matrix<T> where T: Zero + Clone + Display + One {
    #[inline]
    fn zero() -> Matrix<T> { Matrix::<T>::zero() }
    fn is_zero(&self) -> bool {
        self.comp_elem(|x| x.is_zero())
    }
}

// TODO: replace zero to one
impl<T> One for Matrix<T> where T: Zero + Clone + Display + One {
    #[inline]
    fn one() -> Matrix<T> { Matrix::<T>::zero() }
    fn is_one(&self) -> bool {
        self.comp_elem(|x| x.is_zero())
    }
}

impl<T> Display for Matrix<T> where T: Display + Zero + Clone + One {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut strg = "".to_owned();
        for y in 0..self.y() as usize {
            strg.push_str("[");
            for x in 0..self.x() as usize {
                strg.push_str(&format!(" {} ", &self.get(y as i64, x as i64)));
            }
            strg.push_str("]\n");
        }
        // write!(f, "{:?}", self.n[y][x]);
        write!(f, "{}", strg)
    }
}

// implement matrix calculate
macro_rules! opt_impl {
    ($funcn:ident, $func:ident, $c:expr) => {
        impl<T> $funcn<Matrix<T>> for Matrix<T> where T: Clone + Copy + Zero + Display + $funcn<Output = T> + One {
            type Output = Self;
            #[inline]
            fn $func(self, rhs: Self) -> Self {
                Self::cal_with_scalar(self, rhs, $c)
            }
        }
    }
}
opt_impl!(Add, add, |a, b| a + b);
opt_impl!(Mul, mul, |a, b| a * b);
opt_impl!(Sub, sub, |a, b| a - b);
opt_impl!(Div, div, |a, b| a / b);
