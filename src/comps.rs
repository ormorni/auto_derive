use crate::node::{Node, NodeRef};
use itertools::izip;
use std::ops::{Add, Div, Mul, Sub};

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

impl Sub for &NodeRef {
    type Output = NodeRef;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
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

#[cfg(test)]
mod tests {
    use crate::node::{Node, NodeRef};
    use rand::prelude::{StdRng, Rng};
    use rand::SeedableRng;
    use crate::context::Context;

    const SEED: [u8; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
    const DIFF: f64 = 1e-7;
    const ALLOWED_ERROR: f64 = 1e-3;

    /// Asserts that two floating point numbers are close to each other.
    /// Tests that the ratio of the difference and the average is smaller than the allowed value.
    fn assert_close(a: f64, b: f64) {
        if a != b {
            let error = (a - b).abs() * 2. / (a.abs() + b.abs());
            assert!(error < ALLOWED_ERROR, "Values are not close: a={} b={} error={}", a, b, error);
        }
    }

    /// Tests a generic binary function.
    fn test_binary(func: impl Fn(NodeRef, NodeRef) -> NodeRef) {
        let mut rng = StdRng::from_seed(SEED);
        for _ in 0..100 {
            let v1: f64 = rng.gen::<f64>() * 100. - 50.;
            let v2: f64 = rng.gen::<f64>() * 100. - 50.;
            let d1: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v1 * (1. - DIFF);
            let d2: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v2 * (1. - DIFF);

            let ctx = Context::new();
            let node1 = Node::from_data(&[v1], ctx.nodes.clone());
            let node2 = Node::from_data(&[v2], ctx.nodes.clone());
            let diff1 = Node::from_data(&[d1], ctx.nodes.clone());
            let diff2 = Node::from_data(&[d2], ctx.nodes.clone());

            let calc = func(node1.clone(), node2.clone());
            let calc_d1 = func(diff1.clone(), node2.clone());
            let calc_d2 = func(node1.clone(), diff2.clone());

            let grad_map = ctx.derive(func(node1.clone(), node2.clone()));

            let grad1 = grad_map.get(&node1).unwrap().data[0];
            let grad2 = grad_map.get(&node2).unwrap().data[0];

            assert_close(grad1 * (d1 - v1), calc_d1.data[0] - calc.data[0]);
            assert_close(grad2 * (d2 - v2), calc_d2.data[0] - calc.data[0]);
        }
    }

    #[test]
    fn test_add() {
        test_binary(|node1, node2| &node1 + &node2);
    }
    #[test]
    fn test_sub() {
        test_binary(|node1, node2| &node1 - &node2);
    }
    #[test]
    fn test_mul() {
        test_binary(|node1, node2| &node1 * &node2);
    }
    #[test]
    fn test_div() {
        test_binary(|node1, node2| &node1 / &node2);
    }
}
