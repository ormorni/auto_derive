use std::ops::Neg;
/// Implementation of unary functions for the node.
/// To make implementing unary functions simpler,
/// the trait DerivableOp allows easy definition of derivable functions,
/// which can then be used with UnaryComp.
use crate::comps::Computation;
use crate::node::Node;

/// A trait for derivable functions.
/// Used to more easily implement pointwise functions on arrays.
pub trait DerivableOp : Clone {
    type Derivative: DerivableOp;

    /// Applies the function to a float.
    fn apply(&self, src: &f64) -> f64;
    /// Calculates the derivative of the function.
    fn derivative(&self) -> Self::Derivative;
}

/// A function returning zero.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ZeroFunc {}

impl DerivableOp for ZeroFunc {
    type Derivative = ZeroFunc;

    fn apply(&self, _: &f64) -> f64 {
        0.
    }

    fn derivative(&self) -> Self::Derivative {
        ZeroFunc {}
    }
}

/// A function returning a constant.
#[derive(Copy, Clone, PartialEq)]
struct ConstFunc {
    cons: f64,
}

impl DerivableOp for ConstFunc {
    type Derivative = ZeroFunc;

    fn apply(&self, _: &f64) -> f64 {
        self.cons
    }

    fn derivative(&self) -> Self::Derivative {
        ZeroFunc {}
    }
}

/// The identity function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct IdentFunc {}

impl DerivableOp for IdentFunc {
    type Derivative = ConstFunc;

    fn apply(&self, src: &f64) -> f64 {
        *src
    }

    fn derivative(&self) -> Self::Derivative {
        ConstFunc { cons: 1. }
    }
}

/// The signum function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct SignumFunc {}

impl DerivableOp for SignumFunc {
    type Derivative = ZeroFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.signum()
    }

    fn derivative(&self) -> Self::Derivative {
        ZeroFunc {}
    }
}

/// The absolute value function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct AbsFunc {}

impl DerivableOp for AbsFunc {
    type Derivative = SignumFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.abs()
    }

    fn derivative(&self) -> Self::Derivative {
        SignumFunc {}
    }
}

/// The exponent function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ExpFunc {}

impl DerivableOp for ExpFunc {
    type Derivative = ExpFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.exp()
    }

    fn derivative(&self) -> Self::Derivative {
        *self
    }
}

/// The power function, given an integer power.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct PowiFunc {
    power: i32,
    coef: i32,
}

impl DerivableOp for PowiFunc {
    type Derivative = PowiFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.powi(self.power) * (self.coef as f64)
    }

    fn derivative(&self) -> Self::Derivative {
        if self.power != 0 {
            PowiFunc {
                power: self.power - 1,
                coef: self.coef * self.power,
            }
        } else {
            PowiFunc {
                power: 0,
                coef: self.coef * self.power,
            }
        }
    }
}

/// The natural logarithm function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct LnFunc {}

impl DerivableOp for LnFunc {
    type Derivative = PowiFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.ln()
    }

    fn derivative(&self) -> Self::Derivative {
        PowiFunc { power: -1, coef: 1 }
    }
}

/// The natural logarithm function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct NegFunc {}

impl DerivableOp for NegFunc {
    type Derivative = ConstFunc;

    fn apply(&self, src: &f64) -> f64 {
        -*src
    }

    fn derivative(&self) -> Self::Derivative {
        ConstFunc { cons: -1. }
    }
}

impl Neg for &Node {
    type Output = Node;

    fn neg(self) -> Self::Output {
        Node::from_comp(UnaryComp::new(self.clone(), NegFunc {}))
    }
}

/// The sine function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct SinFunc {
    sign_flip: bool,
}

/// The cosine function.
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct CosFunc {
    sign_flip: bool,
}

impl DerivableOp for SinFunc {
    type Derivative = CosFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.sin() * if self.sign_flip { -1. } else { 1. }
    }

    fn derivative(&self) -> Self::Derivative {
        CosFunc {
            sign_flip: self.sign_flip,
        }
    }
}

impl DerivableOp for CosFunc {
    type Derivative = SinFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.cos() * if self.sign_flip { -1. } else { 1. }
    }

    fn derivative(&self) -> Self::Derivative {
        SinFunc {
            sign_flip: !self.sign_flip,
        }
    }
}

/// A computation handling generic differentiable functions applied pointwise to arrays.
#[derive(Clone)]
pub struct UnaryComp<Op: DerivableOp> {
    /// The parent node.
    src: Node,
    /// The function applied to the array.
    op: Op,
}

impl<Op: 'static + DerivableOp> UnaryComp<Op> {
    fn new(src: Node, op: Op) -> UnaryComp<Op> {
        UnaryComp {src, op}
    }


}

impl<Op: DerivableOp + 'static> Computation for UnaryComp<Op> {
    fn sources(&self) -> Vec<Node> {
        vec![self.src.clone()]
    }

    fn derivatives(&self, res_grads: Node) -> Vec<Node> {
        vec![&Node::from_comp(UnaryComp::new(self.src.clone(), self.op.derivative())) * &res_grads]
    }

    fn apply(&self, res_array: &mut [f64]) {
        for i in 0..self.len() {
            res_array[i] += self.op.apply(&self.src.data()[i]);
        }
    }

    fn len(&self) -> usize {
        self.src.len()
    }
}

/// An implementation of the standard f64 functions to floats.
impl Node {
    pub fn sin(&self) -> Node {
        Node::from_comp(UnaryComp::new(self.clone(), SinFunc { sign_flip: false }))
    }
    pub fn cos(&self) -> Node {
        Node::from_comp(UnaryComp::new(self.clone(), CosFunc { sign_flip: false }))
    }
    pub fn exp(&self) -> Node {
        Node::from_comp(UnaryComp::new(self.clone(), ExpFunc {}))
    }
    pub fn powi(&self, power: i32) -> Node {
        Node::from_comp(UnaryComp::new(self.clone(), PowiFunc { power, coef: 1 }))
    }
    pub fn signum(&self) -> Node {
        Node::from_comp(UnaryComp::new(self.clone(), SignumFunc {}))
    }
    pub fn abs(&self) -> Node {
        Node::from_comp(UnaryComp::new(self.clone(), AbsFunc {}))
    }
    pub fn ln(&self) -> Node {
        Node::from_comp(UnaryComp::new(self.clone(), LnFunc {}))
    }
}

/// Tests for the unary functions.
#[cfg(test)]
mod tests {
    use std::ops::Neg;
    use crate::node::Node;
    use rand::prelude::{StdRng, Rng};
    use rand::SeedableRng;

    /// A seed for the RNG used to generate test cases.
    const SEED: [u8; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
    /// The difference used in the analytic differentiation to test the libraries differentiation.
    const DIFF: f64 = 1e-7;
    /// The maximal allowed error in the results.
    const ALLOWED_ERROR: f64 = 1e-3;

    /// Asserts that two floating point numbers are close to each other.
    /// Tests that the ratio of the difference and the average is smaller than the allowed value.
    fn assert_close(a: f64, b: f64) {
        if a != b {
            let error = (a - b).abs() * 2. / (a.abs() + b.abs());
            assert!(error < ALLOWED_ERROR, "Values are not close: a={} b={} error={}", a, b, error);
        }
    }

    /// Tests a generic unary function.
    fn test_unary(func: impl Fn(Node) -> Node) {
        let mut rng = StdRng::from_seed(SEED);
        for _ in 0..100 {
            let v1: f64 = rng.gen::<f64>() * 100. - 50.;
            let v2: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v1 * (1. - DIFF);

            let node1 = Node::from_data(&[v1]);
            let node2 = Node::from_data(&[v2]);

            let mapped_1 = func(node1.clone());
            let mapped_2 = func(node2.clone());

            let grad = mapped_1.derive().get(&node1).unwrap().data()[0];

            assert_close(grad * (v2 - v1), mapped_2.data()[0] - mapped_1.data()[0]);
        }
    }

    #[test]
    fn test_exp() {
        test_unary(|node|node.exp());
    }
    #[test]
    fn test_cos() {
        test_unary(|node|node.cos());
    }
    #[test]
    fn test_sin() {
        test_unary(|node|node.sin());
    }
    #[test]
    fn test_signum() {
        test_unary(|node|node.signum());
    }
    #[test]
    fn test_abs() {
        test_unary(|node|node.abs());
    }
    #[test]
    fn test_ln() {
        test_unary(|node| if node.data()[0] > 0. {node.ln()} else {node});
    }
    #[test]
    fn test_pow() {
        for i in -5..5 {
            test_unary(|node|node.powi(i));
        }
    }
    #[test]
    fn test_neg() {
        test_unary(|node| (&node).neg());
    }
}
