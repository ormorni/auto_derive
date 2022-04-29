use itertools::izip;
use std::ops::{Add, Div, Mul, Sub};

use crate::comps::Computation;
use crate::node::Node;

/// A computation handling pointwise addition of two nodes.
#[derive(Clone)]
struct AddComp {
    p1: Node,
    p2: Node,
}

impl<'t> AddComp {
    fn apply(p1: &Node, p2: &Node) -> Node {
        assert_eq!(p1.len(), p2.len());
        let data: Vec<f64> = izip!(p1.data().iter(), p2.data().iter())
            .map(|(v1, v2)| v1 + v2)
            .collect();
        let comp = Box::new(AddComp {
            p1: p1.clone(),
            p2: p2.clone(),
        });
        let node = Node::from_comp(&data, comp, p1.alloc());
        node
    }
}

impl Computation for AddComp {
    fn sources(&self) -> Vec<Node> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn derivatives(&self, res_grads: Node) -> Vec<Node> {
        vec![res_grads.clone(), res_grads.clone()]
    }
}

impl Add for &Node {
    type Output = Node;

    fn add(self, rhs: Self) -> Self::Output {
        AddComp::apply(self, rhs)
    }
}

impl Sub for &Node {
    type Output = Node;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

/// A computation handling pointwise multiplication of two nodes.
#[derive(Clone)]
struct MulComp {
    p1: Node,
    p2: Node,
}

impl<'t> MulComp {
    fn apply(p1: &Node, p2: &Node) -> Node {
        assert_eq!(p1.len(), p2.len());
        let data: Vec<f64> = izip!(p1.data().iter(), p2.data().iter())
            .map(|(v1, v2)| v1 * v2)
            .collect();
        let comp = Box::new(MulComp {
            p1: p1.clone(),
            p2: p2.clone(),
        });
        let node = Node::from_comp(&data, comp, p1.alloc());
        node
    }
}

impl Computation for MulComp {
    fn sources(&self) -> Vec<Node> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn derivatives(&self, res_grads: Node) -> Vec<Node> {
        vec![
            MulComp::apply(&self.p2, &res_grads),
            MulComp::apply(&self.p1, &res_grads),
        ]
    }
}

impl Mul for &Node {
    type Output = Node;

    fn mul(self, rhs: Self) -> Self::Output {
        MulComp::apply(self, rhs)
    }
}

impl Div for &Node {
    type Output = Node;

    fn div(self, rhs: Self) -> Self::Output {
        MulComp::apply(self, &rhs.powi(-1))
    }
}


#[cfg(test)]
mod tests {
    use crate::node::Node;
    use rand::prelude::{StdRng, Rng};
    use rand::SeedableRng;

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
    fn test_binary(func: impl Fn(Node, Node) -> Node) {
        let mut rng = StdRng::from_seed(SEED);
        for _ in 0..100 {
            let v1: f64 = rng.gen::<f64>() * 100. - 50.;
            let v2: f64 = rng.gen::<f64>() * 100. - 50.;
            let d1: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v1 * (1. - DIFF);
            let d2: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v2 * (1. - DIFF);

            let node1 = Node::from_data(&[v1]);
            let node2 = Node::from_data_and_node(&[v2], &node1);
            let diff1 = Node::from_data_and_node(&[d1], &node1);
            let diff2 = Node::from_data_and_node(&[d2], &node1);

            let calc = func(node1.clone(), node2.clone());
            let calc_d1 = func(diff1.clone(), node2.clone());
            let calc_d2 = func(node1.clone(), diff2.clone());

            let grad_map = func(node1.clone(), node2.clone()).derive();

            let grad1 = grad_map.get(&node1).unwrap().data()[0];
            let grad2 = grad_map.get(&node2).unwrap().data()[0];

            assert_close(grad1 * (d1 - v1), calc_d1.data()[0] - calc.data()[0]);
            assert_close(grad2 * (d2 - v2), calc_d2.data()[0] - calc.data()[0]);
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