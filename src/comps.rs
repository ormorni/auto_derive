use crate::node::{Node, NodeRef};
use itertools::izip;
use std::ops::{Add, Div, Mul};

/// A trait representing the computations which were used to generate nodes in the computation graph.
/// Used to perform the backward propagation.
pub trait Computation {
    /// Returns a vector of the parent nodes involved in the computation.
    fn sources(&self) -> Vec<NodeRef>;
    /// Calculates the derivatives of the computation by each of the parent nodes.
    fn derivatives(&self, res_grads: NodeRef) -> Vec<NodeRef>;
}

/// A computation that does nothing.
/// Used for nodes which are initialized from raw data, and are not computed.
#[derive(Copy, Clone)]
pub struct NullComp {}

impl Computation for NullComp {
    fn sources(&self) -> Vec<NodeRef> {
        vec![]
    }

    fn derivatives(&self, _: NodeRef) -> Vec<NodeRef> {
        vec![]
    }
}

/// A computation handling pointwise addition of two nodes.
#[derive(Clone)]
struct AddComp {
    p1: NodeRef,
    p2: NodeRef,
}

impl<'t> AddComp {
    fn apply(p1: &NodeRef, p2: &NodeRef) -> NodeRef {
        assert_eq!(p1.len(), p2.len());
        let data: Vec<f64> = izip!(p1.data.iter(), p2.data.iter())
            .map(|(v1, v2)| v1 + v2)
            .collect();
        let comp = Box::new(AddComp {
            p1: p1.clone(),
            p2: p2.clone(),
        });
        let node = Node::from_comp(&data, comp, p1.alloc.clone());
        node
    }
}

impl Computation for AddComp {
    fn sources(&self) -> Vec<NodeRef> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn derivatives(&self, res_grads: NodeRef) -> Vec<NodeRef> {
        vec![res_grads.clone(), res_grads.clone()]
    }
}

impl Add for &NodeRef {
    type Output = NodeRef;

    fn add(self, rhs: Self) -> Self::Output {
        AddComp::apply(self, rhs)
    }
}

/// A computation handling pointwise multiplication of two nodes.
#[derive(Clone)]
struct MulComp {
    p1: NodeRef,
    p2: NodeRef,
}

impl<'t> MulComp {
    fn apply(p1: &NodeRef, p2: &NodeRef) -> NodeRef {
        assert_eq!(p1.len(), p2.len());
        let data: Vec<f64> = izip!(p1.data.iter(), p2.data.iter())
            .map(|(v1, v2)| v1 * v2)
            .collect();
        let comp = Box::new(MulComp {
            p1: p1.clone(),
            p2: p2.clone(),
        });
        let node = Node::from_comp(&data, comp, p1.alloc.clone());
        node
    }
}

impl Computation for MulComp {
    fn sources(&self) -> Vec<NodeRef> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn derivatives(&self, res_grads: NodeRef) -> Vec<NodeRef> {
        vec![
            MulComp::apply(&self.p2, &res_grads),
            MulComp::apply(&self.p1, &res_grads),
        ]
    }
}

impl Mul for &NodeRef {
    type Output = NodeRef;

    fn mul(self, rhs: Self) -> Self::Output {
        MulComp::apply(self, rhs)
    }
}

impl Div for &NodeRef {
    type Output = NodeRef;

    fn div(self, rhs: Self) -> Self::Output {
        MulComp::apply(self, &rhs.powi(-1))
    }
}

/// A computation that takes indices from a node.
/// Can be used to take ranges of an array, to perform permutations, etc.
#[derive(Clone)]
pub struct IndexComp {
    /// The parent node.
    node: NodeRef,
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
    pub fn apply<Iter: Iterator<Item = (usize, usize)>>(
        node: &NodeRef,
        iter: Iter,
        length: usize,
    ) -> NodeRef {
        // Making sure all indices are legal.
        let indices: Vec<(usize, usize)> = iter.collect();
        assert!(indices.iter().all(|idx| idx.0 < node.len()));
        assert!(indices.iter().all(|idx| idx.1 < length));

        let mut data = vec![0.; length];
        for (src, tar) in indices.iter() {
            data[*tar] += node.data[*src];
        }
        let comp = Box::new(IndexComp {
            node: node.clone(),
            indices,
            length,
        });
        let node = Node::from_comp(&data, comp, node.alloc.clone());
        node
    }
}

impl Computation for IndexComp {
    fn sources(&self) -> Vec<NodeRef> {
        vec![self.node.clone()]
    }

    fn derivatives(&self, res_grads: NodeRef) -> Vec<NodeRef> {
        let src_len = self.node.len();
        let inverted_indices = self.indices.iter().map(|(i, j)| (*j, *i));
        vec![IndexComp::apply(&res_grads, inverted_indices, src_len)]
    }
}
