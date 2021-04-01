use copperhead_core::tensor::*;
use copperhead_core::ops::*;

#[test]
fn test_tensor_add() {
    let t0: StaticTensor<i32, 2> = StaticTensor::full([2, 3], 1);
    let t1: StaticTensor<i32, 2> = StaticTensor::full([2, 3], 2);
    let t2 = t0 + t1;

    assert_eq!(t2.data, vec![3; 6]);
}

#[test]
fn test_tensor_add_assign() {
    let mut t0: StaticTensor<i32, 2> = StaticTensor::full([2, 3], 1);
    let t1: StaticTensor<i32, 2> = StaticTensor::full([2, 3], 2);

    t0 += t1;

    assert_eq!(t0.data, vec![3; 6]);
}
