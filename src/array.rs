use std::cell::UnsafeCell;
use crate::computation::{Computation, ComputationType, FromDataComp};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::{Arc, RwLock};
use fxhash::{FxHashMap, FxHashSet};
use itertools::izip;
use rand::Rng;
use crate::unary_functions::{DerivableOp, UnaryComp};

type Map<K, V> = FxHashMap<K, V>;
type IdType = usize;

/// A wrapper over a float array, which dynamically creates a computation graph.
/// The computation graph can then be used to automatically calculate derivatives of complex functions
/// using backward propagation.
struct DArrayInternal {
    /// The data stored by the array.
    data: RwLock<UnsafeCell<Option<Vec<f64>>>>,
    /// The computation used to calculate the array. Tracks the computation graph.
    comp: Box<dyn Computation>,
    /// The length of the array held by the DArray.
    length: usize,
    /// An ID, used to easily sort the arrays by order of creation.
    id: IdType,
}

impl DArrayInternal {
    /// Gets the data of the internal array.
    fn data(&self) -> &Vec<f64> {
        // Initializing.
        unsafe {
            if self.data.read()
                .unwrap()
                .get()
                .as_ref()
                .unwrap()
                .is_none() {
                // Before a thread is allowed to modify the data, it must first obtain the lock.
                // Since the DArrays form a DAG, there is a partial ordering on the mutexes.
                // One of the mutexes will always be minimal, and will be acquired successfully.
                let mut guard = self.data.write().unwrap();
                let data = guard.get_mut();
                if data.is_none() {
                    *data = Some(vec![0.; self.comp.len()]);
                    self.comp.apply_on_zero(data.as_mut().unwrap());
                }
            }

            self.data.read()
                .unwrap()
                .get()
                .as_ref()
                .unwrap()
                .as_ref()
                .unwrap()
        }
    }
    /// Checks if the data in the DArrayInternal is initialized.
    fn is_init(&self) -> bool {
        unsafe {
            self.data.read().unwrap().get().as_ref().unwrap().is_some()
        }
    }
}

unsafe impl Sync for DArray {}
unsafe impl Send for DArray {}


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
        comp: impl Computation + Clone,
    ) -> DArray {
        DArray::new(DArrayInternal {
            data: RwLock::new(UnsafeCell::new(None)),
            length: comp.len(),
            comp: Box::new(comp),
            id: rand::thread_rng().gen::<IdType>(),
        })
    }

    /// Returns the length of the array held by the array.
    pub fn len(&self) -> usize {
        self.internal.length
    }

    /// Returns a reference to the array's data.
    pub fn data(&self) -> &Vec<f64> {
        // If the node is already initialized, we return the data and require no further computations.
        if self.is_initialized() {
            return self.internal.data();
        }

        // The function selects a subset of the parent nodes of the given node, and calls `.data()` on them.
        // This reduces the number of recursive calls to the function in the internal .data() .
        // However, calling the function interferes with the allocation-reducing mechanism, so it should be minimized.
        // The functions that call .data() are:
        // * Unary, when called not on zero.
        // * Binary. When called normally allocate twice, when called on zero allocate once.
        // * Other. Perform no optimization, and always allocate all their child nodes.
        // Additions never allocate, but may propagate the call on zero.
        //
        // In addition, nodes with two or more parents should always be evaluated, to prevent evaluating them twice.

        // Counting the number of parents of each node.
        let mut parent_count = FxHashMap::default();
        parent_count.insert(self.clone(), 0);

        let mut queue = vec![self.clone()];
        let mut idx = 0;
        while idx < queue.len() {
            let node = &queue[idx];
            idx += 1;

            for child_node in node.comp().sources() {
                if child_node.is_initialized() {
                    continue;
                }
                if !parent_count.contains_key(&child_node) {
                    queue.push(child_node.clone());
                    parent_count.insert(child_node.clone(), 0);
                }
                *parent_count.get_mut(&child_node).unwrap() += 1;
            }
        }

        // Initializing is_allocated with all nodes with more than one parent.
        let mut is_allocated: FxHashSet<DArray> = parent_count.iter().filter_map(|(node, &par_count)|if par_count > 1 {Some(node)} else {None}).cloned().collect();
        let mut is_applied_on_zero = FxHashSet::default();

        // Topological sorting.
        let mut topo = vec![self.clone()];
        let mut idx = 0;
        while idx < topo.len() {
            let node = &topo[idx];
            idx += 1;

            for child_node in node.comp().sources() {
                if child_node.is_initialized() {
                    continue;
                }

                let par_count = parent_count.get_mut(&child_node).unwrap();
                *par_count -= 1;
                if *par_count == 0 {
                    topo.push(child_node);
                }
            }
        }

        for node in topo.iter() {
            let sources = node.comp().sources();
            let on_zero = is_allocated.contains(node) || is_applied_on_zero.contains(node);

            match node.comp().get_type() {
                // Additions never allocate, and propagate applies on zero.
                ComputationType::Add => {
                    assert_eq!(sources.len(), 2);
                    if on_zero {
                        if is_allocated.contains(&sources[0]) {
                            is_applied_on_zero.insert(sources[1].clone());
                        } else {
                            is_applied_on_zero.insert(sources[0].clone());
                        }
                    }
                }
                // Binaries applied on zero allocate one of their children, and otherwise both.
                ComputationType::Binary => {
                    assert_eq!(sources.len(), 2);
                    if on_zero {
                        if is_allocated.contains(&sources[0]) {
                            is_applied_on_zero.insert(sources[1].clone());
                        } else {
                            is_applied_on_zero.insert(sources[0].clone());
                            is_allocated.insert(sources[1].clone());
                        }
                    } else {
                        for node in sources.iter() {
                            is_allocated.insert(node.clone());
                        }
                    }
                }
                // Unaries always apply their child node on zero. They allocate if they are not applied on zero.
                ComputationType::Unary => {
                    is_applied_on_zero.insert(sources[0].clone());
                    assert_eq!(sources.len(), 1);
                    // Unary are allocated if they are not applied on zero.
                    if !on_zero {
                        is_allocated.insert(node.clone());
                    }
                }
                // Other type allocate all their child nodes.
                ComputationType::Other => {
                    for node in sources {
                        is_allocated.insert(node);
                    }
                }
            }
        }

        for node in topo.iter().rev() {
            if is_allocated.contains(node) {
                node.internal.data();
            }
        }

        self.internal.data()
    }

    /// Returns a reference to the array's computation.
    pub fn comp(&self) -> &Box<dyn Computation> {
        &self.internal.comp
    }

    /// Returns the list of all intermediates of the array.
    /// The ordering of the list guarantees that every array in the list is placed before all its source arrays.
    /// To ensure correct derivations, the backpropagation has to be called on all arrays using a
    /// given array before being called on it. The topological sorting ensures that the calls to the backpropagation
    /// satisfies this requirement.
    pub fn topological_sort(&self) -> Vec<DArray> {
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

    /// Maps the array using a derivable function.
    pub fn map(&self, op: impl DerivableOp) -> DArray {
        DArray::from(UnaryComp::new(self.clone(), op.clone()))
    }

    pub fn is_initialized(&self) -> bool {
        self.internal.is_init()
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

impl From<&DArray> for DArray {
    fn from(src: &DArray) -> Self {
        src.clone()
    }
}

impl <Comp: Computation + Clone> From<Comp> for DArray {
    fn from(src: Comp) -> Self {
        DArray::from_comp(src)
    }
}

pub trait DArrayRef {
    fn into(self) -> DArray;
}

impl DArrayRef for DArray {
    fn into(self) -> DArray {
        self
    }
}
impl DArrayRef for &DArray {
    fn into(self) -> DArray {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use crate::array::{DArray, DArrayInternal};

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