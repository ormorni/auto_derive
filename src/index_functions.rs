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
        Node::from_comp(Box::new(IndexComp::new(node, iter, length)))
    }
}

impl Computation for IndexComp {
    fn sources(&self) -> Vec<Node> {
        vec![self.node.clone()]
    }

    fn derivatives(&self, res_grads: Node) -> Vec<Node> {
        let src_len = self.node.len();
        let inverted_indices = self.indices.iter().map(|(i, j)| (*j, *i));

        vec![Node::from_comp(Box::new(IndexComp::new(&res_grads, inverted_indices, src_len)))]
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