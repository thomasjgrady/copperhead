use copperhead_core::indexing::Array;
use copperhead_core::tensor::*;

#[test]
pub fn test_static_tensor_full() {
    let t: StaticTensor<i32, 3> = StaticTensor::full([2, 3, 4], 11);
    assert_eq!(t.data, vec![11; 24]);
    assert_eq!(t.shape, [2, 3, 4]);
    assert_eq!(t.flat_size, 24);
    assert_eq!(t.strides, [12, 4, 1]);
}

#[test]
pub fn test_static_tensor_fill_with() {
    let mut t: StaticTensor<i32, 3> = StaticTensor::full([2, 3, 4], 0);
    t.fill_with(|| 5+6);
    assert_eq!(t.data, vec![11; 24]);
}

#[test]
pub fn test_static_tensor_fill_with_index() {
    let mut t: StaticTensor<i32, 2> = StaticTensor::full([2, 3], 0);
    t.fill_with_index(|index: Array<2>| (index[0]+index[1]) as i32);

    assert_eq!(t.data[0], 0);
    assert_eq!(t.data[1], 1);
    assert_eq!(t.data[2], 2);
    assert_eq!(t.data[3], 1);
    assert_eq!(t.data[4], 2);
    assert_eq!(t.data[5], 3);
}

#[test]
pub fn test_static_tensor_init_with() {
    let t: StaticTensor<i32, 3> = StaticTensor::init_with([2, 3, 4], || 5+6);
    assert_eq!(t.data, vec![11; 24]);
}

#[test]
pub fn test_static_tensor_init_with_index() {
    let t: StaticTensor<i32, 2> = StaticTensor::init_with_index(
        [2, 3], 
        |index: Array<2>| (index[0]+index[1]) as i32
    );
    assert_eq!(t.data[0], 0);
    assert_eq!(t.data[1], 1);
    assert_eq!(t.data[2], 2);
    assert_eq!(t.data[3], 1);
    assert_eq!(t.data[4], 2);
    assert_eq!(t.data[5], 3);
}
