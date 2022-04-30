use crate::computation::{Computation, FromDataComp};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;
use fxhash::FxHashMap;
use itertools::izip;
use rand::Rng;
use sloth::Lazy;

type Map<K, V> = FxHashMap<K, V>;


/// A wrapper over a float array, which dynamically creates a computation graph.
/// The computation graph can then be used to automatically calculate derivatives of complex functions
/// using backward propagation.
struct DArrayInternal {
    /// The data stored by the array.
    data: Lazy<Vec<f64>, Box<dyn FnOnce() -> Vec<f64>>>,
    /// The computation used to calculate the array. Tracks the computation graph.
    comp: Box<dyn Computation>,
    /// An ID, used to easily sort the arrays by order of creation.
    id: usize,
}

/// A struct proving a comfortable handle for the actual arrays.
#[derive(Clone)]
pub struct DArray {
    internal: Arc<DArrayInternal>,
}

impl DArray {
    /// A constructor for array references from a raw array.
    /// Used only in the array's constructors.
    fn new(array: DArrayInternal) -> Self {
        DArray {
            internal: Arc::new(array),
        }
    }

    /// Initializes an array from a slice of floats.
    fn from_data(data: &[f64]) -> DArray {
        DArray::from_comp(FromDataComp {data: data.to_vec()})
    }

    /// Initializes an array from a slice of floats and the computation used to calculate it.
    fn from_comp(
        comp: impl Computation + 'static + Clone,
    ) -> DArray {
        let ln = comp.len();
        let cloned_comp = comp.clone();

        DArray::new(DArrayInternal {
            data: Lazy::new(Box::new(move || {
                let mut data = vec![0.; ln];
                cloned_comp.apply(&mut data);
                data
            })),
            comp: Box::new(comp),
            id: rand::thread_rng().gen::<usize>(),
        })

    }
    /// Returns the length of the array held by the array.
    pub fn len(&self) -> usize {
        self.internal.data.len()
    }

    /// Returns a reference to the array's data.
    pub fn data(&self) -> &Vec<f64> {
        &self.internal.data
    }

    /// Returns a reference to the array's computation.
    pub fn comp(&self) -> &Box<dyn Computation> {
        &self.internal.comp
    }

    /// Performs topological sorting of the array.
    /// To ensure correct derivations, the backpropagation has to be called on all arrays using a
    /// given array before being called on it. The topological sorting ensures that the calls to the backpropagation
    /// satisfies this requirement.
    fn topological_sort(&self) -> Vec<DArray> {
        // Topologically sorting the required arrays of the computation graph.
        let mut parent_count = Map::default();
        let mut queue = vec![self.clone()];
        let mut idx = 0;
        parent_count.insert(self.clone(), 0);
        while idx < queue.len() {
            for array in queue[idx].comp().sources() {
                if !parent_count.contains_key(&array) {
                    parent_count.insert(array.clone(), 0);
                    queue.push(array.clone());
                }
                *parent_count.get_mut(&array).unwrap() += 1;
            }
            idx += 1;
        }

        let mut res = vec![self.clone()];
        let mut idx = 0;
        while idx < res.len() {
            for array in res[idx].comp().sources() {
                *parent_count.get_mut(&array).unwrap() -= 1;
                if *parent_count.get_mut(&array).unwrap() == 0 {
                    res.push(array);
                }
            }
            idx += 1;
        }

        res
    }

    /// Calculates the derivative of the target value with respect to all intermediates in the computation graph.
    pub fn derive(&self) -> Map<DArray, DArray> {
        assert_eq!(
            self.len(),
            1,
            "Derivatives are supported only for scalars! Array length is {}",
            self.len()
        );

        // Initializing the derivative map.
        let mut grads = Map::default();
        grads.insert(self.clone(), DArray::from_data(&[1.]));

        for array in self.topological_sort() {
            let array_grads = grads.get(&array).unwrap();
            let sources = array.comp().sources();
            let source_grads = array.comp().derivatives(array_grads.clone());

            for (source, grad) in izip!(sources.iter(), source_grads.iter()) {
                let old_grad = grads.get(source);
                if old_grad.is_none() {
                    grads.insert(source.clone(), grad.clone());
                } else {
                    let old_grad = old_grad.unwrap();
                    let new_grad = old_grad + grad;
                    grads.insert(source.clone(), new_grad);
                }
            }
        }

        grads
    }

    /// Returns if the array represents a single item.
    pub fn is_scalar(&self) -> bool {
        self.len() == 1
    }
}

impl Eq for DArrayInternal {}

impl PartialEq<Self> for DArrayInternal {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for DArrayInternal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl Eq for DArray {}

impl PartialEq<Self> for DArray {
    fn eq(&self, other: &Self) -> bool {
        self.internal.deref().eq(&*other.internal.deref())
    }
}

impl Hash for DArray {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal.deref().hash(state)
    }
}

impl From<f64> for DArray {
    fn from(src: f64) -> Self {
        DArray::from_data(&[src])
    }
}

impl From<Vec<f64>> for DArray {
    fn from(src: Vec<f64>) -> Self {
        DArray::from_data(src.as_slice())
    }
}

impl <Comp: Computation + Clone + 'static> From<Comp> for DArray {
    fn from(src: Comp) -> Self {
        DArray::from_comp(src)
    }
}


#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use crate::array::DArray;

    const SEED: [u8; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
    const DIFF: f64 = 1e-7;
    const REL_ERROR: f64 = 1e-3;
    const ABS_ERROR: f64 = 1e-10;

    /// Asserts that two floating point numbers are close to each other.
    /// Tests that the ratio of the difference and the average is smaller than the allowed value.
    fn assert_close(a: f64, b: f64) {
        if a != b {
            let error = (a - b).abs() * 2. / (a.abs() + b.abs());
            assert!((error < REL_ERROR) || (a - b).abs() < ABS_ERROR, "Values are not close: a={} b={} error={}", a, b, error);
        }
    }

    /// Tests that the derivative of complex random rational functions are evaluated correctly.
    #[test]
    fn test_derivation() {
        let mut rng = StdRng::from_seed(SEED);
        for _ in 0..100 {
            let v1: f64 = rng.gen::<f64>() * 100. - 50.;
            let v2: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v1 * (1. - DIFF);

            let root = DArray::from_data(&[v1, v2]);

            let mut arr = vec![];
            for _ in 0..3 {
                arr.push(root.clone());
            }

            for _ in 0..20 {
                let p1: usize = rng.gen_range(0..arr.len());
                let p2: usize = rng.gen_range(0..arr.len());
                let op: usize = rng.gen_range(0..3);

                match op {
                    0 => {arr[p1] = &arr[p1] + &arr[p2]},
                    1 => {arr[p1] = &arr[p1] * &arr[p2]},
                    2 => {arr[p1] = &arr[p1] / &arr[p2]},
                    _ => panic!()
                }
            }
            let res = arr[0].index(0);
            let grad = res.derive().get(&root).unwrap().data()[0];
            assert_close(arr[0].data()[1] - arr[0].data()[0], grad * (root.data()[1] - root.data()[0]));
        }
    }
}