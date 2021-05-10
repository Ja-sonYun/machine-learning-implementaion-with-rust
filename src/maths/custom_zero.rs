// original source code is from below,
//
// https://docs.rs/num/0.1.28/src/num/.cargo/registry/src/github.com-1ecc6299db9ec823/num-0.1.28/src/traits.rs.html#215-236
//
//
// Remove Copy trait for implement Zero for Matrix
pub trait Zero: Sized {
    fn zero() -> Self;
    fn is_zero(&self) -> bool;
}

macro_rules! zero_impl {
    ($t:ty, $v:expr) => {
        impl Zero for $t {
            #[inline]
            fn zero() -> $t { $v }
            #[inline]
            fn is_zero(&self) -> bool { *self == $v }
        }
    }
}

zero_impl!(usize, 0usize);
zero_impl!(u8,   0u8);
zero_impl!(u16,  0u16);
zero_impl!(u32,  0u32);
zero_impl!(u64,  0u64);

zero_impl!(isize, 0isize);
zero_impl!(i8,  0i8);
zero_impl!(i16, 0i16);
zero_impl!(i32, 0i32);
zero_impl!(i64, 0i64);

zero_impl!(f32, 0.0f32);
zero_impl!(f64, 0.0f64);
