use std::ops::Neg;
use itertools::izip;
/// Implementation of unary functions for the array.
/// To make implementing unary functions simpler,
/// the trait DerivableOp allows easy definition of derivable functions,
/// which can then be used with UnaryComp.
use crate::computation::Computation;
use crate::array::DArray;

/// A trait for derivable functions.
/// Used to more easily implement pointwise functions on arrays.
pub trait DerivableOp : Clone + 'static {
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

impl Neg for &DArray {
    type Output = DArray;
    fn neg(self) -> Self::Output {
        self.map(NegFunc {})
    }
}
impl Neg for DArray {
    type Output = DArray;
    fn neg(self) -> Self::Output {
        self.map(NegFunc {})
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
    /// The parent array.
    src: DArray,
    /// The function applied to the array.
    op: Op,
}

impl<Op: DerivableOp> UnaryComp<Op> {
    /// Initializes a new unary computation object.
    pub fn new(src: DArray, op: Op) -> UnaryComp<Op> {
        UnaryComp {src, op}
    }
}

impl<Op: DerivableOp> Computation for UnaryComp<Op> {
    fn sources(&self) -> Vec<DArray> {
        vec![self.src.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        vec![self.src.map(self.op.derivative()) * res_grads]
    }

    fn apply(&self, res_array: &mut [f64]) {
        for (res, data) in izip!(res_array.iter_mut(), self.src.data().iter()) {
            *res += self.op.apply(data);
        }
    }

    fn len(&self) -> usize {
        self.src.len()
    }
}

/// An implementation of the standard f64 functions to floats.
impl DArray {
    pub fn sin(&self) -> DArray {
        self.map(SinFunc { sign_flip: false })
    }
    pub fn cos(&self) -> DArray {
        self.map(CosFunc { sign_flip: false })
    }
    pub fn exp(&self) -> DArray {
        self.map(ExpFunc {})
    }
    pub fn powi(&self, power: i32) -> DArray {
        self.map(PowiFunc { power, coef: 1 })
    }
    pub fn signum(&self) -> DArray {
        self.map(SignumFunc {})
    }
    pub fn abs(&self) -> DArray {
        self.map(AbsFunc {})
    }
    pub fn ln(&self) -> DArray {
        self.map(LnFunc {})
    }
}

/// A function testing if the value is larger than some contant.
#[derive(Copy, Clone, PartialEq)]
struct GtFunc {
    val: f64,
}

impl DerivableOp for GtFunc {
    type Derivative = ZeroFunc;

    fn apply(&self, src: &f64) -> f64 {
        if *src > self.val {
            1.
        } else {
            0.
        }
    }

    fn derivative(&self) -> Self::Derivative {
        ZeroFunc {}
    }
}

/// The pointwise maximum function.
#[derive(Copy, Clone, PartialEq)]
struct MaxFunc {
    val: f64,
}

impl DerivableOp for MaxFunc {
    type Derivative = GtFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.max(self.val)
    }

    fn derivative(&self) -> Self::Derivative {
        GtFunc { val: self.val }
    }
}


/// A function testing if the value is larger than some contant.
#[derive(Copy, Clone, PartialEq)]
struct LtFunc {
    val: f64,
}

impl DerivableOp for LtFunc {
    type Derivative = ZeroFunc;

    fn apply(&self, src: &f64) -> f64 {
        if *src < self.val {
            1.
        } else {
            0.
        }
    }

    fn derivative(&self) -> Self::Derivative {
        ZeroFunc {}
    }
}

/// The pointwise maximum function.
#[derive(Copy, Clone, PartialEq)]
struct MinFunc {
    val: f64,
}

impl DerivableOp for MinFunc {
    type Derivative = GtFunc;

    fn apply(&self, src: &f64) -> f64 {
        src.min(self.val)
    }

    fn derivative(&self) -> Self::Derivative {
        GtFunc { val: self.val }
    }
}

impl DArray {
    /// Performs the pointwise maximum function.
    pub fn max(&self, val: f64) -> DArray {
        self.map(MaxFunc { val })
    }
    /// Performs the pointwise minimum function.
    pub fn min(&self, val: f64) -> DArray {
        self.map(MinFunc { val })
    }
    /// Returns an array with ones where the original value is larger than the given value and 0 otherwise.
    pub fn gt(&self, val: f64) -> DArray {
        self.map(GtFunc { val })
    }
    /// Returns an array with ones where the original value is smaller than the given value and 0 otherwise.
    pub fn lt(&self, val: f64) -> DArray {
        self.map(LtFunc { val })
    }
}


/// Tests for the unary functions.
#[cfg(test)]
mod tests {
    use std::ops::Neg;
    use crate::array::DArray;
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
    fn test_unary(func: impl Fn(DArray) -> DArray) {
        let mut rng = StdRng::from_seed(SEED);
        for _ in 0..100 {
            let v1: f64 = rng.gen::<f64>() * 100. - 50.;
            let v2: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v1 * (1. - DIFF);

            let array1 = DArray::from(v1);
            let array2 = DArray::from(v2);

            let mapped_1 = func(array1.clone());
            let mapped_2 = func(array2.clone());

            let grad = mapped_1.derive().get(&array1).unwrap().data()[0];

            assert_close(grad * (v2 - v1), mapped_2.data()[0] - mapped_1.data()[0]);
        }
    }

    #[test]
    fn test_exp() {
        test_unary(|array|array.exp());
    }
    #[test]
    fn test_cos() {
        test_unary(|array|array.cos());
    }
    #[test]
    fn test_sin() {
        test_unary(|array|array.sin());
    }
    #[test]
    fn test_signum() {
        test_unary(|array|array.signum());
    }
    #[test]
    fn test_abs() {
        test_unary(|array|array.abs());
    }
    #[test]
    fn test_ln() {
        test_unary(|array| if array.data()[0] > 0. {array.ln()} else {array});
    }
    #[test]
    fn test_pow() {
        for i in -5..5 {
            test_unary(|array|array.powi(i));
        }
    }
    #[test]
    fn test_neg() {
        test_unary(|array| (&array).neg());
    }
}
