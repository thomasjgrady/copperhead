use copperhead_core::indexing::*;

#[test]
fn test_static_range_iterator() {
    let starts = [1, 1, 1];
    let stops = [3, 3, 4];
    let range = StaticRange::new(starts, stops);

    let mut iterator = range.into_iter();

    assert_eq!(iterator.next(), Some([1, 1, 1]));
    assert_eq!(iterator.next(), Some([1, 1, 2]));
    assert_eq!(iterator.next(), Some([1, 1, 3]));
    assert_eq!(iterator.next(), Some([1, 2, 1]));
    assert_eq!(iterator.next(), Some([1, 2, 2]));
    assert_eq!(iterator.next(), Some([1, 2, 3]));
    assert_eq!(iterator.next(), Some([2, 1, 1]));
    assert_eq!(iterator.next(), Some([2, 1, 2]));
    assert_eq!(iterator.next(), Some([2, 1, 3]));
    assert_eq!(iterator.next(), Some([2, 2, 1]));
    assert_eq!(iterator.next(), Some([2, 2, 2]));
    assert_eq!(iterator.next(), Some([2, 2, 3]));
    assert_eq!(iterator.next(), None);
}
