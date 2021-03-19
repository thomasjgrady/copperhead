/// Type used to index datatypes within this crate. This type must support
/// negative values to allow for indexing relative to the end of a given
/// tensor.
pub type Ordinal = i32;

/// Used for N-dimensional indices, shapes, etc
pub type Array<const N: usize> = [usize; N];

/// Computes the total flat size of an N-dimensional volume with the given
/// shape
pub fn flat_size_from_shape<const N: usize>(shape: &Array<N>) -> usize {
    shape.iter().fold(1, |a, e| a*e)
}

/// Computes the strides needed to perform N-dimensional indexing on an
/// N-dimensional volume with the given shape
pub fn strides_from_shape<const N: usize>(shape: &Array<N>) -> Array<N> {
    let mut strides: Array<N> = [1; N];
    strides.iter_mut()
        .enumerate()
        .for_each(|(dim, stride)| {
            if dim < N-1 {
                *stride = ((dim+1)..N).fold(1 as usize, |a, j| a*shape[j]);
            }
        });
    strides
}

#[inline(always)]
pub fn flat_index_from_nd_index<const N: usize>(index: Array<N>, strides: Array<N>) -> usize {
    index.iter()
        .zip(strides.iter())
        .fold(0, |a, (i, s)| a + i*s)
}

/// N-dimensional range with dimension known at compile-time.
#[derive(Clone, Debug)]
pub struct StaticRange<const N: usize> {
    pub starts: Array<N>,
    pub stops: Array<N>
}

impl<const N: usize> StaticRange<N> {
    pub fn new(starts: Array<N>, stops: Array<N>) -> Self {
        Self { starts, stops }
    }
}

/// Struct containing state information needed for N-dimensional iteration
pub struct StaticRangeIterator<const N: usize> {
    pub range: StaticRange<N>,
    pub shape: Array<N>,
    pub strides: Array<N>,

    pub flat_size: usize,
    pub flat_index: usize
}

impl<const N: usize> StaticRangeIterator<N> {
    pub fn new(range: StaticRange<N>) -> Self {
        let mut shape: Array<N> = [1; N];
        range.starts.iter()
            .zip(range.stops.iter())
            .enumerate()
            .for_each(|(i, (start, stop))| shape[i] = stop-start);

        let strides = strides_from_shape(&shape);
        let flat_size = flat_size_from_shape(&shape);
        let flat_index = 0;

        Self {
            range,
            shape,
            strides,
            flat_size,
            flat_index
        }
    }
}

impl<const N: usize> Iterator for StaticRangeIterator<N> {
    type Item = Array<N>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.flat_index >= self.flat_size {
            return None;
        }

        let mut index: Array<N> = [0; N];
        for i in 0..N {
            let dim = self.shape[i];
            let stride = self.strides[i];
            let start = self.range.starts[i];

            index[i] = (self.flat_index / stride) % dim + start;
        }
        self.flat_index += 1;
        Some(index)
    }
}

impl<const N: usize> IntoIterator for StaticRange<N> {
    type Item = Array<N>;
    type IntoIter = StaticRangeIterator<N>;
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

pub trait IntoNdIterator<const N: usize> {
    fn into_nd_iter(self) -> StaticRangeIterator<N>;
}
