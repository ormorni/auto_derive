use crate::comps::Computation;
use crate::node::Node;

/// A computation that takes indices from a node.
/// Can be used to take ranges of an array, to perform permutations, etc.
#[derive(Clone)]
pub struct IndexComp {
    /// The parent node.
    node: Node,
    /// A list of indices in the parent node taken to the child node, in the format `[(par_idx, child_idx), ...]`.
    /// If an index in the child node appears several times, the appropriate elements of the parent
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
        node: &Node,
        iter: impl Iterator<Item = (usize, usize)>,
        length: usize,
    ) -> IndexComp {
        // Making sure all indices are legal.
        let indices: Vec<(usize, usize)> = iter.collect();
        assert!(indices.iter().all(|idx| idx.0 < node.len()));
        assert!(indices.iter().all(|idx| idx.1 < length));

        IndexComp {node: node.clone(), indices, length}
    }

    pub fn map_indices(node: &Node,
                              iter: impl Iterator<Item = (usize, usize)>,
                              length: usize) -> Node {
        Node::from(IndexComp::new(node, iter, length))
    }
}

impl Computation for IndexComp {
    fn sources(&self) -> Vec<Node> {
        vec![self.node.clone()]
    }

    fn derivatives(&self, res_grads: Node) -> Vec<Node> {
        let src_len = self.node.len();
        let inverted_indices = self.indices.iter().map(|(i, j)| (*j, *i));

        vec![Node::from(IndexComp::new(&res_grads, inverted_indices, src_len))]
    }

    fn apply(&self, res_array: &mut [f64]) {
        for (src, tar) in self.indices.iter() {
            res_array[*tar] += self.node.data()[*src];
        }
    }

    fn len(&self) -> usize {
        self.length
    }
}

impl Node {
    pub fn index(&self, idx: usize) -> Node {
        IndexComp::map_indices(self, [(idx, 0)].iter().cloned(), 1)
    }
}

/// A computation that handles summing all elements in an array.
#[derive(Clone)]
struct SumComp {
    src: Node,
}

impl Computation for SumComp {
    fn sources(&self) -> Vec<Node> {
        vec![self.src.clone()]
    }

    fn derivatives(&self, res_grads: Node) -> Vec<Node> {
        assert_eq!(res_grads.len(), self.len());
        vec![Node::from(ExpandComp::new(res_grads, self.src.len()))]
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

impl Node {
    pub fn sum(&self) -> Node {
        Node::from(SumComp {src: self.clone()})
    }
}


/// A computation that handles expanding a scalar to an array.
/// This operation can be done with IndexComp, but this should be both lighter, since it doesn't require
/// the index mapping array, and easier to use.
#[derive(Clone)]
pub struct ExpandComp {
    src: Node,
    length: usize,
}

impl ExpandComp {
    pub fn new(src: Node, length: usize) -> ExpandComp {
        assert_eq!(src.len(), 1);
        ExpandComp {src, length}
    }
}

impl Computation for ExpandComp {
    fn sources(&self) -> Vec<Node> {
        vec![self.src.clone()]
    }

    fn derivatives(&self, res_grads: Node) -> Vec<Node> {
        vec![Node::from(SumComp {src: res_grads})]
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

/// If the source node is a scalar, expand it to an array of the same length as the target length.
pub fn expand_node(src: Node, tar_len: &Node) -> Node {
    if src.is_scalar() && !tar_len.is_scalar() {
        Node::from(ExpandComp::new(src, tar_len.len()))
    } else {
        src
    }
}

#[cfg(test)]
mod tests {
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use crate::Node;

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
            let node_arr = Node::from(arr);
            let node_sum = node_arr.sum();

            assert!(node_sum.is_scalar());
            assert_close(arr_sum, node_sum.data()[0]);
            node_sum.derive().get(&node_arr).unwrap().data().iter().for_each(|f|assert_close(*f, 1.))
        }
    }

    /// Tests that the binary functions on an array and a scalar work properly.
    #[test]
    fn test_expand() {
        let mut rng = StdRng::from_seed(SEED);

        for _ in 0..100 {
            let arr: Vec<f64> = (0..10).map(|_|rng.gen::<f64>()).collect();
            let arr_node = Node::from(arr.clone());

            let scalar = rng.gen::<f64>();
            let add_node_right = &arr_node + &Node::from(scalar);
            let add_node_left = &Node::from(scalar) + &arr_node;
            let mul_node_right = &arr_node * &Node::from(scalar);
            let mul_node_left = &Node::from(scalar) * &arr_node;

            for i in 0..arr.len() {
                assert_close(add_node_right.data()[i], arr[i] + scalar);
                assert_close(add_node_left.data()[i], arr[i] + scalar);
                assert_close(mul_node_right.data()[i], arr[i] * scalar);
                assert_close(mul_node_left.data()[i], arr[i] * scalar);
            }
        }
    }

    /// Testing that pointwise addition of two non-scalar arrays of different lengths fails.
    #[test]
    #[should_panic]
    fn test_expand_fail() {
        let node_1 = Node::from(vec![1., 2., 3.]);
        let node_2 = Node::from(vec![1., 2.]);
        let _node_3 = &node_1 + &node_2;
    }
}


