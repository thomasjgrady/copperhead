use crate::tensor::*;
use std::ops::{
    Add,
    AddAssign,
    Sub,
    SubAssign,
    Mul,
    MulAssign,
    Div,
    DivAssign
};

impl<T, const N: usize> Add for StaticTensor<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(self.shape == rhs.shape); 
        
        // SAFETY: This tensor is immediately filled, so it is ok to create
        // it as empty
        let mut result = unsafe { Self::empty(self.shape.clone()) };
        self.data.into_iter()
            .zip(rhs.data.into_iter())
            .enumerate()
            .for_each(|(i, (e0, e1))| result.data[i] = e0 + e1);

        result
    }
}

impl<T, const N: usize> AddAssign for StaticTensor<T, N>
where
    T: AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.data.iter_mut()
            .zip(rhs.data.into_iter())
            .for_each(|(e0, e1)| *e0 += e1);
    }
}
