pub mod matrix;
pub mod calculus;
pub mod rand;
pub mod custom_zero;

#[derive(Debug, Clone, Copy)]
pub enum NumOrZeroF64 {
    Num(f64),
    Zero,
}
impl NumOrZeroF64 {
    pub fn reveal(self) -> f64 {
        match self {
            NumOrZeroF64::Num(x) => x,
            _ => 0.,
        }
    }
}

