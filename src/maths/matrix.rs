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
use crate::maths::custom_zero::Zero;

// type DebugElement = Zero + Display + Copy;

// use crate::utils::maths::derivative;
#[derive(Clone, Copy)]
enum Element<T> {
    Zero,
    Num(T)
}
impl<T> Element<T> {
    fn num(self) -> T where T: Zero + Display + Copy {
        match self {
            Element::Num(v) => v,
            _ => Zero::zero(),
        }
    }
}
impl<T> Debug for Element<T> where T: Zero + Display + Copy {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{}", self.num())
    }
}

#[derive(Clone)]
pub struct Matrix<T> where T: Zero {
    n: Vec<Vec<Element<T>>>,
    _nx: i64,
    _ny: i64,
}

impl<T> Matrix<T> where T: Clone + Zero {

    #[inline]
    pub fn new(ny: i64, nx:i64) -> Matrix<T> {
        Matrix { n: (0..ny).map(|_| vec![Element::Zero; nx as usize]).collect(), _nx: nx, _ny: ny }
    }

    #[inline]
    pub fn new_scalar(val: T) -> Matrix<T> {
        Matrix { n: vec![vec![Element::Num(val)]], _nx: 1, _ny: 1 }
    }

    #[inline]
    pub fn is_scalar(&self) -> bool {
        self._nx == 1 && self._ny == 1
    }

    #[inline]
    pub fn y(&self) -> i64 {
        self._ny
    }

    #[inline]
    pub fn x(&self) -> i64 {
        self._nx
    }

    #[inline]
    pub fn from_fn<F: Fn()->T>(init_fn: F, ny: i64, nx:i64) -> Matrix<T> {
        Matrix { n: (0..ny).map(|_| (0..nx).map(|_| Element::Num(init_fn())).collect()).collect(), _nx: nx, _ny: ny }
    }

    #[inline]
    pub fn get(&self, y: i64, x: i64) -> T {
        match &self.n[y as usize][x as usize] {
            Element::Num(n) => n.clone(),
            Element::Zero => Zero::zero(),
        }
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
    pub fn get_dim(&self) -> (i64, i64) {
        (self._nx, self._ny)
    }

    #[inline]
    pub fn comp_dim_with(&self, shr: &Self) -> bool {
        self.get_dim() == shr.get_dim()
    }
    pub fn set(&mut self, y: i64, x: i64, val: T) {
        if self.is_scalar() {
            panic!("this is scalar")
        }
        self.n[y as usize][x as usize] = Element::Num(val);
    }
    pub fn s_set(&mut self, val: T) {
        if self.is_scalar() {
            self.n[0][0] = Element::Num(val);
        } else {
            panic!("This is a matrix not the scalar.");
        }
    }
    pub fn elem_with_scalar<F: Fn(T, T)->T>(&self, with: T, cal_fn: F) -> Self {
        // TODO: Refactoring this, use map?
        let mut temp = Matrix::<T>::new(self._ny, self._nx);
        forfor!(self._ny, y, self._nx, x, {
            temp.set(y, x, cal_fn(self.get(y, x), with.clone()));
        });
        temp
    }
    pub fn elemwise_cal<F: Fn(T, T)->T>(&self, with: Self, cal_fn: F) -> Self {
        let mut temp = Matrix::<T>::new(self._ny, self._nx);
        forfor!(self._ny, y, self._nx, x, {
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
        forfor!(self._ny, y, self._nx, x, {
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

impl<T> Zero for Matrix<T> where T: Zero + Clone + Display {
    #[inline]
    fn zero() -> Matrix<T> { Matrix::<T>::zero() }
    fn is_zero(&self) -> bool {
        self.comp_elem(|x| x.is_zero())
    }
}

impl<T> Display for Matrix<T> where T: Display + Zero + Clone {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut strg = "".to_owned();
        for y in 0..self._ny as usize {
            strg.push_str("[");
            for x in 0..self._nx as usize {
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
        impl<T> $funcn<Matrix<T>> for Matrix<T> where T: Clone + Copy + Zero + Display + $funcn<Output = T> {
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