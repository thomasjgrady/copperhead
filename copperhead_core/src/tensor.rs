use crate::indexing::*;

/// Multidimensional array with known dimensionality at compile-time.
pub struct StaticTensor<T, const N: usize> {
    pub data: Vec<T>,
    pub shape: Array<N>,
    pub strides: Array<N>,
    pub flat_size: usize
}

impl<T, const N: usize> StaticTensor<T, N>
where
    T: Copy
{

    #[inline(always)]
    pub fn at(&self, index: Array<N>) -> T {
        self.data[flat_index_from_nd_index(index, self.strides)]
    }

    #[inline(always)]
    pub fn at_ref_mut(&mut self, index: Array<N>) -> &mut T {
        &mut self.data[flat_index_from_nd_index(index, self.strides)]
    }

    /// Creates a tensor with the given shape, without setting and values. This is unsafe
    /// because the resulting tensor contains unitialized memory. Normally used as
    /// part of some other constructor.
    pub unsafe fn empty(shape: Array<N>) -> Self {
        let flat_size = flat_size_from_shape(&shape);
        let mut data = Vec::<T>::with_capacity(flat_size);

        // SAFETY: We wish to just allocate memory on the heap with no
        // default value, so it is ok to set the vector's length
        data.set_len(flat_size);
        
        let strides = strides_from_shape(&shape);

        Self {
            data,
            shape,
            strides,
            flat_size
        }
    }
    
    /// Fills this tensor with the value returned by the given closure
    pub fn fill_with<F>(&mut self, mut f: F)
    where
        F: FnMut() -> T
    {
        self.data.iter_mut().for_each(|t| *t = f());
    }
    
    /// Fills this tensor with the value returned by the given closure
    /// at each index
    pub fn fill_with_index<F>(&mut self, mut f: F)
    where
        F: FnMut(Array<N>) -> T
    {
        let range = StaticRange::new([0; N], self.shape);
        range.into_iter()
            .for_each(|index| *self.at_ref_mut(index.clone()) = f(index));
    }
    
    /// Creates a tensor with the given shape containing the designated fill
    /// value in every position.
    pub fn full(shape: Array<N>, fill_value: T) -> Self {
        let flat_size = flat_size_from_shape(&shape);
        let data = vec![fill_value; flat_size];
        let strides = strides_from_shape(&shape);

        Self {
            data,
            shape,
            strides,
            flat_size
        }
    }
}

impl<T, const N: usize> IntoNdIterator<N> for StaticTensor<T, N> {
    fn into_nd_iter(self) -> StaticRangeIterator<N> {
        let starts = [0; N];
        let stops = self.shape;
        let range = StaticRange::new(starts, stops);
        StaticRangeIterator::new(range)
    }
}
