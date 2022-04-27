/// Implementation of unary functions for the node.
/// To make implementing unary functions simpler,
/// the trait DerivableOp allows easy definition of derivable functions,
/// which can then be used with UnaryComp.
use crate::comps::Computation;
use crate::node::{Node, NodeRef};

/// A trait for derivable functions.
/// Used to more easily implement pointwise functions on arrays.
pub trait DerivableOp {
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
        0.
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
            sign_flip: !self.sign_flip,
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
            sign_flip: self.sign_flip,
        }
    }
}

/// A computation handling generic differentiable functions applied pointwise to arrays.
pub struct UnaryComp<Op: DerivableOp> {
    /// The parent node.
    src: NodeRef,
    /// The function applied to the array.
    op: Op,
}

impl<Op: 'static + DerivableOp> UnaryComp<Op> {
    fn apply(node: NodeRef, op: Op) -> NodeRef {
        let data: Vec<f64> = node.data.iter().map(|f| op.apply(f)).collect();
        let comp = Box::new(UnaryComp {
            src: node.clone(),
            op,
        });
        let res = Node::from_comp(&data, comp, node.alloc.clone());
        res
    }
}

impl<Op: DerivableOp + 'static> Computation for UnaryComp<Op> {
    fn sources(&self) -> Vec<NodeRef> {
        vec![self.src.clone()]
    }

    fn derivatives(&self, res_grads: NodeRef) -> Vec<NodeRef> {
        vec![&UnaryComp::apply(self.src.clone(), self.op.derivative()) * &res_grads]
    }
}

/// An implementation of the standard f64 functions to floats.
impl NodeRef {
    pub fn sin(&self) -> NodeRef {
        UnaryComp::apply(self.clone(), SinFunc { sign_flip: false })
    }
    pub fn cos(&self) -> NodeRef {
        UnaryComp::apply(self.clone(), CosFunc { sign_flip: false })
    }
    pub fn exp(&self) -> NodeRef {
        UnaryComp::apply(self.clone(), ExpFunc {})
    }
    pub fn powi(&self, power: i32) -> NodeRef {
        UnaryComp::apply(self.clone(), PowiFunc { power, coef: 1 })
    }
    pub fn signum(&self) -> NodeRef {
        UnaryComp::apply(self.clone(), SignumFunc {})
    }
    pub fn abs(&self) -> NodeRef {
        UnaryComp::apply(self.clone(), AbsFunc {})
    }
    pub fn ln(&self) -> NodeRef {
        UnaryComp::apply(self.clone(), LnFunc {})
    }
}

mod tests {}
