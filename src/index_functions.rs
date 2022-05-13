use crate::computation::Computation;
use crate::array::DArray;

/// A computation that takes indices from an array.
/// Can be used to take ranges of an array, to perform permutations, etc.
#[derive(Clone)]
pub struct IndexComp {
    /// The parent array.
    array: DArray,
    /// A list of indices in the parent array taken to the child array, in the format `[(par_idx, child_idx), ...]`.
    /// If an index in the child array appears several times, the appropriate elements of the parent
    /// array are summed.
    indices: Vec<(usize, usize)>,
    /// The length of the child array generated.
    length: usize,
}

impl IndexComp {
    /// Takes indices from the parent array into a child array.
    /// The iterator is an iterator of `(parent_idx, child_idx)`, specifying elements of the parent
    /// added to elements of the child array.
    fn new(
        array: &DArray,
        iter: impl Iterator<Item = (usize, usize)>,
        length: usize,
    ) -> IndexComp {
        // Making sure all indices are legal.
        let indices: Vec<(usize, usize)> = iter.collect();
        assert!(indices.iter().all(|idx| idx.0 < array.len()));
        assert!(indices.iter().all(|idx| idx.1 < length));

        IndexComp {array: array.clone(), indices, length}
    }

    pub fn map_indices(array: &DArray,
                       iter: impl Iterator<Item = (usize, usize)>,
                       length: usize) -> DArray {
        DArray::from(IndexComp::new(array, iter, length))
    }
}

impl Computation for IndexComp {
    fn sources(&self) -> Vec<DArray> {
        vec![self.array.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        let src_len = self.array.len();
        let inverted_indices = self.indices.iter().map(|(i, j)| (*j, *i));

        vec![DArray::from(IndexComp::new(&res_grads, inverted_indices, src_len))]
    }

    fn apply(&self, res_array: &mut [f64]) {
        let data = self.array.data();
        for (src, tar) in self.indices.iter() {
            res_array[*tar] += data[*src];
        }
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl DArray {
    pub fn index(&self, idx: usize) -> DArray {
        IndexComp::map_indices(self, [(idx, 0)].iter().cloned(), 1)
    }

    /// Returns the maximal element of the array.
    pub fn reduce_max(&self) -> DArray {
        let mx_idx = self.data().iter().enumerate().reduce(|p1, p2| if p1.1 > p2.1 {p1} else {p2}).unwrap().0;
        self.index(mx_idx)
    }
    /// Returns the minimal element of the array.
    pub fn reduce_min(&self) -> DArray {
        let mn_idx = self.data().iter().enumerate().reduce(|p1, p2| if p1.1 < p2.1 {p1} else {p2}).unwrap().0;
        self.index(mn_idx)
    }
}

/// A computation that handles summing all elements in an array.
#[derive(Clone)]
struct SumComp {
    src: DArray,
}

impl Computation for SumComp {
    fn sources(&self) -> Vec<DArray> {
        vec![self.src.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        assert_eq!(res_grads.len(), self.len());
        vec![DArray::from(ExpandComp::new(res_grads, self.src.len()))]
    }

    fn apply(&self, res_array: &mut [f64]) {
        assert_eq!(res_array.len(), 1);
        for v in self.src.data() {
            res_array[0] += v;
        }
    }

    fn len(&self) -> usize {
        1
    }
}

impl DArray {
    pub fn sum(&self) -> DArray {
        DArray::from(SumComp {src: self.clone()})
    }
}


/// A computation that handles expanding a scalar to an array.
/// This operation can be done with IndexComp, but this should be both lighter, since it doesn't require
/// the index mapping array, and easier to use.
#[derive(Clone)]
pub struct ExpandComp {
    src: DArray,
    length: usize,
}

impl ExpandComp {
    pub fn new(src: DArray, length: usize) -> ExpandComp {
        assert_eq!(src.len(), 1);
        ExpandComp {src, length}
    }
}

impl Computation for ExpandComp {
    fn sources(&self) -> Vec<DArray> {
        vec![self.src.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        vec![DArray::from(SumComp {src: res_grads})]
    }

    fn apply(&self, res_array: &mut [f64]) {
        assert_eq!(res_array.len(), self.len());
        let src = self.src.data()[0];
        for i in res_array.iter_mut() {
            *i += src;
        }
    }

    fn len(&self) -> usize {
        self.length
    }
}

/// If the source array is a scalar, expand it to an array of the same length as the target length.
pub fn expand_array(src: DArray, tar_len: &DArray) -> DArray {
    if src.is_scalar() && !tar_len.is_scalar() {
        DArray::from(ExpandComp::new(src, tar_len.len()))
    } else {
        src
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use crate::DArray;

    const SEED: [u8; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
    const ALLOWED_ERROR: f64 = 1e-3;

    /// Asserts that two floating point numbers are close to each other.
    /// Tests that the ratio of the difference and the average is smaller than the allowed value.
    fn assert_close(a: f64, b: f64) {
        if a != b {
            let error = (a - b).abs() * 2. / (a.abs() + b.abs());
            assert!(error < ALLOWED_ERROR, "Values are not close: a={} b={} error={}", a, b, error);
        }
    }

    #[test]
    fn test_sum() {
        let mut rng = StdRng::from_seed(SEED);

        for _ in 0..100 {
            let arr: Vec<f64> = (0..10).map(|_|rng.gen::<f64>()).collect();
            let arr_sum = arr.iter().sum();
            let array_arr = DArray::from(arr);
            let array_sum = array_arr.sum();

            assert!(array_sum.is_scalar());
            assert_close(arr_sum, array_sum.data()[0]);
            array_sum.derive().get(&array_arr).unwrap().data().iter().for_each(|f|assert_close(*f, 1.))
        }
    }

    #[test]
    fn test_max() {
        let mut rng = StdRng::from_seed(SEED);

        for _ in 0..100 {
            let arr: Vec<f64> = (0..10).map(|_|rng.gen::<f64>()).collect();
            let arr_max = arr.iter().cloned().reduce(|f1, f2|f1.max(f2)).unwrap();
            let array_arr = DArray::from(arr);
            let array_max = array_arr.reduce_max();

            assert!(array_max.is_scalar());
            assert_close(arr_max, array_max.data()[0]);
        }
    }

    #[test]
    fn test_min() {
        let mut rng = StdRng::from_seed(SEED);

        for _ in 0..100 {
            let arr: Vec<f64> = (0..10).map(|_|rng.gen::<f64>()).collect();
            let arr_min = arr.iter().cloned().reduce(|f1, f2|f1.min(f2)).unwrap();
            let array_arr = DArray::from(arr);
            let array_min = array_arr.reduce_min();

            assert!(array_min.is_scalar());
            assert_close(arr_min, array_min.data()[0]);
        }
    }

    /// Tests that the binary functions on an array and a scalar work properly.
    #[test]
    fn test_expand() {
        let mut rng = StdRng::from_seed(SEED);

        for _ in 0..100 {
            let arr: Vec<f64> = (0..10).map(|_|rng.gen::<f64>()).collect();
            let arr_array = DArray::from(arr.clone());

            let scalar = rng.gen::<f64>();
            let add_array_right = &arr_array + &DArray::from(scalar);
            let add_array_left = &DArray::from(scalar) + &arr_array;
            let mul_array_right = &arr_array * &DArray::from(scalar);
            let mul_array_left = &DArray::from(scalar) * &arr_array;

            for i in 0..arr.len() {
                assert_close(add_array_right.data()[i], arr[i] + scalar);
                assert_close(add_array_left.data()[i], arr[i] + scalar);
                assert_close(mul_array_right.data()[i], arr[i] * scalar);
                assert_close(mul_array_left.data()[i], arr[i] * scalar);
            }
        }
    }

    /// Testing that pointwise addition of two non-scalar arrays of different lengths fails.
    #[test]
    #[should_panic]
    fn test_expand_fail() {
        let array_1 = DArray::from(vec![1., 2., 3.]);
        let array_2 = DArray::from(vec![1., 2.]);
        let _array_3 = &array_1 + &array_2;
    }
}