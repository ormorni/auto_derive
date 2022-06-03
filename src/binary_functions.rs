use std::ops::{Add, Div, Mul, Neg, Sub};
use itertools::izip;

use crate::computation::{Computation, ComputationType};
use crate::index_functions::expand_array;
use crate::array::{DArray, DArrayRef};

/// A computation handling pointwise addition of two arrays.
#[derive(Clone)]
struct AddComp {
    p1: DArray,
    p2: DArray,
}

impl<'t> AddComp {
    fn new(p1: DArray, p2: DArray) -> AddComp {
        assert_eq!(p1.len(), p2.len());
        AddComp {p1, p2}
    }
}

impl Computation for AddComp {
    fn sources(&self) -> Vec<DArray> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        vec![res_grads.clone(), res_grads.clone()]
    }

    fn len(&self) -> usize {
        self.p1.len()
    }

    fn apply(&self, res_array: &mut [f64]) {
        if self.p1.is_initialized() {
            let data = self.p1.data();
            for i in 0..self.len() {
                res_array[i] += data[i];
            }
        } else {
            self.p1.comp().apply(res_array);
        }
        if self.p2.is_initialized() {
            let data = self.p2.data();
            for i in 0..self.len() {
                res_array[i] += data[i];
            }
        } else {
            self.p2.comp().apply(res_array);
        }
    }

    fn get_type(&self) -> ComputationType {
        ComputationType::Add
    }

    /// Applies addition on a zero array. Propagates the "apply on zero" to an uninitialized array if possible.
    fn apply_on_zero(&self, res_array: &mut [f64]) {
        match (self.p1.is_initialized(), self.p2.is_initialized()) {
            (true, true) => {
                for (res, i, j) in izip!(res_array.iter_mut(), self.p1.data(), self.p2.data()) {
                    *res += i + j;
                }
            }
            (true, false) => {
                self.p2.comp().apply_on_zero(res_array);
                for (res, i) in izip!(res_array.iter_mut(), self.p1.data()) {
                    *res += i;
                }
            }
            (false, true) => {
                self.p1.comp().apply_on_zero(res_array);
                for (res, i) in izip!(res_array.iter_mut(), self.p2.data()) {
                    *res += i;
                }
            }
            (false, false) => {
                self.p1.comp().apply_on_zero(res_array);
                self.p2.comp().apply(res_array);
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
struct AddScalarComp {
    non_scalar: DArray,
    scalar: DArray,
}

impl AddScalarComp {
    fn new(p1: DArray, p2: DArray) -> AddScalarComp {
        if p1.is_scalar() {
            return AddScalarComp {non_scalar: p2, scalar: p1};
        }
        if p2.is_scalar() {
            return AddScalarComp {non_scalar: p1, scalar: p2};
        }
        panic!("AddScalarComp created with no scalars!")
    }
}

impl Computation for AddScalarComp {
    fn sources(&self) -> Vec<DArray> {
        vec![self.non_scalar.clone(), self.scalar.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        vec![res_grads.clone(), res_grads.sum()]
    }

    fn len(&self) -> usize {
        self.non_scalar.len()
    }

    fn apply(&self, res_array: &mut [f64]) {
        self.non_scalar.comp().apply(res_array);
        let c = self.scalar.data()[0];
        for v in res_array.iter_mut() {
            *v += c;
        }
    }

    fn apply_on_zero(&self, res_array: &mut [f64]) {
        self.non_scalar.comp().apply_on_zero(res_array);
        let c = self.scalar.data()[0];
        for v in res_array.iter_mut() {
            *v += c;
        }
    }

    fn get_type(&self) -> ComputationType {
        ComputationType::Add
    }
}

impl <Other: Into<DArray>> Add<Other> for &DArray {
    type Output = DArray;
    fn add(self, rhs: Other) -> Self::Output {
        let rhs = rhs.into();
        if self.is_scalar() || rhs.is_scalar() {
            DArray::from(AddScalarComp::new(self.clone(), rhs))
        } else {
            DArray::from(AddComp::new(self.clone(), rhs))
        }
    }
}

impl <Other: Into<DArray>> Add<Other> for DArray {
    type Output = DArray;
    fn add(self, rhs: Other) -> Self::Output {
        let rhs = rhs.into();
        if self.is_scalar() ^ rhs.is_scalar() {
            DArray::from(AddScalarComp::new(self.clone(), rhs))
        } else {
            DArray::from(AddComp::new(self, rhs))
        }
    }
}

impl <OtherNeg : Into<DArray>, Other: Neg<Output = OtherNeg>>  Sub<Other> for &DArray {
    type Output = DArray;
    fn sub(self, rhs: Other) -> Self::Output {
        self.clone() + (-rhs)
    }
}
impl <OtherNeg : Into<DArray>, Other: Neg<Output = OtherNeg>>  Sub<Other> for DArray {
    type Output = DArray;
    fn sub(self, rhs: Other) -> Self::Output {
        self + (-rhs)
    }
}

/// A computation handling pointwise multiplication of two arrays.
#[derive(Clone)]
struct MulComp {
    p1: DArray,
    p2: DArray,
}

impl<'t> MulComp {
    fn new(p1: DArray, p2: DArray) -> MulComp {
        assert_eq!(p1.len(), p2.len());
        MulComp {p1, p2}
    }
}

impl Computation for MulComp {
    fn sources(&self) -> Vec<DArray> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        vec![
            &self.p2 * &res_grads,
            &self.p1 * &res_grads,
        ]
    }

    fn len(&self) -> usize {
        self.p1.len()
    }

    fn apply(&self, res_array: &mut [f64]) {
        assert_eq!(res_array.len(), self.len());
        for (res, p1, p2) in izip!(res_array.iter_mut(), self.p1.data().iter(), self.p2.data().iter()) {
            *res += p1 * p2;
        }
    }


    fn get_type(&self) -> ComputationType {
        ComputationType::Binary
    }

    fn apply_on_zero(&self, res_array: &mut [f64]) {
        if !self.p1.is_initialized() {
            self.p1.comp().apply_on_zero(res_array);
            for (v1, v2) in izip!(res_array.iter_mut(), self.p2.data().iter()) {
                *v1 *= v2;
            }
        } else {
            self.p2.comp().apply_on_zero(res_array);
            for (v1, v2) in izip!(res_array.iter_mut(), self.p1.data().iter()) {
                *v1 *= v2;
            }
        }
    }
}


#[derive(Clone, Eq, PartialEq)]
struct MulScalarComp {
    non_scalar: DArray,
    scalar: DArray,
}

impl MulScalarComp {
    fn new(p1: DArray, p2: DArray) -> MulScalarComp {
        if p1.is_scalar() {
            return MulScalarComp {non_scalar: p2, scalar: p1};
        }
        if p2.is_scalar() {
            return MulScalarComp {non_scalar: p1, scalar: p2};
        }
        panic!("MulScalarComp created with no scalars!")
    }
}

impl Computation for MulScalarComp {
    fn sources(&self) -> Vec<DArray> {
        vec![self.non_scalar.clone(), self.scalar.clone()]
    }

    fn derivatives(&self, res_grads: DArray) -> Vec<DArray> {
        vec![res_grads.clone() * self.scalar.clone(), (res_grads * self.non_scalar.clone()).sum()]
    }

    fn len(&self) -> usize {
        self.non_scalar.len()
    }

    fn apply(&self, res_array: &mut [f64]) {
        self.non_scalar.comp().apply(res_array);
        let c = self.scalar.data()[0];
        for v in res_array.iter_mut() {
            *v += c;
        }
    }

    fn get_type(&self) -> ComputationType {
        ComputationType::Binary
    }

    fn apply_on_zero(&self, res_array: &mut [f64]) {
        self.non_scalar.comp().apply_on_zero(res_array);
        let c = self.scalar.data()[0];
        for v in res_array.iter_mut() {
            *v *= c;
        }
    }
}

impl <Other: DArrayRef> Mul<Other> for &DArray {
    type Output = DArray;

    fn mul(self, rhs: Other) -> Self::Output {
        let rhs = rhs.into();
        if self.is_scalar() ^ rhs.is_scalar() {
            DArray::from(MulScalarComp::new(self.clone(), rhs))
        } else {
            DArray::from(MulComp::new(self.clone(), rhs))
        }
    }
}
impl <Other: DArrayRef> Mul<Other> for DArray {
    type Output = DArray;

    fn mul(self, rhs: Other) -> Self::Output {
        let rhs = rhs.into();
        if self.is_scalar() ^ rhs.is_scalar() {
            DArray::from(MulScalarComp::new(self, rhs))
        } else {
            DArray::from(MulComp::new(self, rhs))
        }
    }
}

impl <Other: DArrayRef> Div<Other> for DArray {
    type Output = DArray;

    fn div(self, rhs: Other) -> Self::Output {
        self * rhs.into().powi(-1)
    }
}
impl <Other: DArrayRef> Div<Other> for &DArray {
    type Output = DArray;

    fn div(self, rhs: Other) -> Self::Output {
        self * rhs.into().powi(-1)
    }
}


#[cfg(test)]
mod tests {
    use crate::array::DArray;
    use crate::test_utils::*;


    /// Tests a generic binary function.
    fn test_binary(func: impl Fn(DArray, DArray) -> DArray) {
        let mut rng = StdRng::from_seed(SEED);
        for i in 0..100 {
            let v1: f64 = rng.gen::<f64>() * 100. - 50.;
            let v2: f64 = rng.gen::<f64>() * 100. - 50.;
            let d1: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v1 * (1. - DIFF);
            let d2: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v2 * (1. - DIFF);

            let array1 = DArray::from(v1);;
            let diff1 = DArray::from(d1);;
            let array2;
            let diff2;
            if i % 2 == 0 {
                array2 = DArray::from(v2);
                diff2 = DArray::from(d2);
            } else {
                array2 = DArray::from(vec![v2, v2]);
                diff2 = DArray::from(vec![d2, d2]);
            }

            let calc = func(array1.clone(), array2.clone());
            let calc_d1 = func(diff1.clone(), array2.clone());
            let calc_d2 = func(array1.clone(), diff2.clone());

            let grad_map = func(array1.clone(), array2.clone()).index(0).derive();

            let grad1 = grad_map.get(&array1).unwrap().data()[0];
            let grad2 = grad_map.get(&array2).unwrap().data()[0];

            assert_close(grad1 * (d1 - v1), calc_d1.data()[0] - calc.data()[0]);
            assert_close(grad2 * (d2 - v2), calc_d2.data()[0] - calc.data()[0]);
        }
    }

    #[test]
    fn test_add() {
        test_binary(|array1, array2| array1 + array2);
    }
    #[test]
    fn test_sub() {
        test_binary(|array1, array2| array1 - array2);
    }
    #[test]
    fn test_mul() {
        test_binary(|array1, array2| array1 * array2);
    }
    #[test]
    fn test_div() {
        test_binary(|array1, array2| array1 / array2);
    }
}
