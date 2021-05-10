use crate::utils::types::*;
use crate::maths::*;

#[derive(Debug)]
pub enum LAYER {
    IN(Vec<Weight>),
    OUT(Vec<Weight>),
    HIDDEN,
}
impl LAYER {
    pub fn unwrap(self) -> Option<Vec<Weight>> {
        match self {
            LAYER::IN(vw) | LAYER::OUT(vw) => Some(vw),
            _ => None
        }
    }
    pub fn unwrap_with_index(&self, i: usize) -> NumOrZeroF64 {
        match &self {
            LAYER::IN(vw) | LAYER::OUT(vw) => {
                NumOrZeroF64::Num(vw[i])
            },
            _ => NumOrZeroF64::Zero
        }
    }
    pub fn get_dim(&self, fan_in: Dimension, fan_out: Dimension) -> Dimension {
        match &self {
            LAYER::IN(_) | LAYER::HIDDEN => fan_in,
            LAYER::OUT(_) => fan_out
        }
    }
    pub fn get_len(&self) -> usize {
        match &self {
            LAYER::IN(val) | LAYER::OUT(val) => val.len(),
            _ => 0
        }
    }
    pub fn check_size(&self, dim: Dimension) -> bool {
        match &self {
            LAYER::IN(val) | LAYER::OUT(val) => (val.len() == dim as usize),
            _ => true
        }
    }
}
