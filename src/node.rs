use crate::comps::{Computation, FromDataComp, NullComp};
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use fxhash::FxHashMap;
use itertools::izip;
use rand::Rng;

type Map<K, V> = FxHashMap<K, V>;


/// A wrapper over a float array, which dynamically creates a computation graph.
/// The computation graph can then be used to automatically calculate derivatives of complex functions
/// using backward propagation.
/// It is never used directly, and only used inside a NodeRef.
struct NodeInternal {
    /// The data stored by the node.
    data: Vec<f64>,
    /// The computation used to calculate the node. Tracks the computation graph.
    comp: Box<dyn Computation>,
    /// An ID, used to easily sort the nodes by order of creation.
    id: usize,
}

/// A struct proving a comfortable handle for the actual nodes.
#[derive(Clone)]
pub struct Node {
    node: Rc<NodeInternal>,
}

impl Node {
    /// A constructor for node references from a raw node.
    /// Used only in the Node's constructors.
    fn new(node: NodeInternal) -> Self {
        Node {
            node: Rc::new(node),
        }
    }

    /// Initializes a node from a slice of floats.
    pub fn from_data(data: &[f64]) -> Node {
        Node::from_comp(FromDataComp {data: data.to_vec()})
    }

    /// Initializes a node from a slice of floats and the computation used to calculate it.
    pub fn from_comp(
        comp: impl Computation + 'static,
    ) -> Node {
        let mut data = vec![0.; comp.len()];
        comp.apply(&mut data);

        Node::new(NodeInternal {
            data: data.to_vec(),
            comp: Box::new(comp),
            id: rand::thread_rng().gen::<usize>(),
        })

    }
    /// Returns the length of the array held by the node.
    pub fn len(&self) -> usize {
        self.node.data.len()
    }

    /// Returns a reference to the node's data.
    pub fn data(&self) -> &Vec<f64> {
        &self.node.data
    }

    /// Returns a reference to the node's computation.
    pub fn comp(&self) -> &Box<dyn Computation> {
        &self.node.comp
    }

    /// Performs topological sorting of the node.
    /// To ensure correct derivations, the backpropagation has to be called on all nodes using a
    /// given node before being called on it. The topological sorting ensures that the calls to the backpropagation
    /// satisfies this requirement.
    fn topological_sort(&self) -> Vec<Node> {
        // Topologically sorting the required nodes of the computation graph.
        let mut parent_count = Map::default();
        let mut queue = vec![self.clone()];
        let mut idx = 0;
        parent_count.insert(self.clone(), 0);
        while idx < queue.len() {
            for node in queue[idx].comp().sources() {
                if !parent_count.contains_key(&node) {
                    parent_count.insert(node.clone(), 0);
                    queue.push(node.clone());
                }
                *parent_count.get_mut(&node).unwrap() += 1;
            }
            idx += 1;
        }

        let mut res = vec![self.clone()];
        let mut idx = 0;
        while idx < res.len() {
            for node in res[idx].comp().sources() {
                *parent_count.get_mut(&node).unwrap() -= 1;
                if *parent_count.get_mut(&node).unwrap() == 0 {
                    res.push(node);
                }
            }
            idx += 1;
        }

        res
    }

    /// Calculates the derivative of the target value with respect to all intermediates in the computation graph.
    pub fn derive(&self) -> Map<Node, Node> {
        assert_eq!(
            self.len(),
            1,
            "Derivatives are supported only for Nodes with a single element! Real len is {}",
            self.len()
        );

        // Initializing the derivative map.
        let mut grads = Map::default();
        grads.insert(self.clone(), Node::from_data(&[1.]));

        for node in self.topological_sort() {
            let node_grads = grads.get(&node).unwrap();
            let sources = node.comp().sources();
            let source_grads = node.comp().derivatives(node_grads.clone());

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
}

impl Eq for NodeInternal {}

impl PartialEq<Self> for NodeInternal {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for NodeInternal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl Eq for Node {}

impl PartialEq<Self> for Node {
    fn eq(&self, other: &Self) -> bool {
        self.node.deref().eq(&*other.node.deref())
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.deref().hash(state)
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use crate::index_functions::IndexComp;
    use crate::node::Node;

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

            let root = Node::from_data(&[v1, v2]);

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