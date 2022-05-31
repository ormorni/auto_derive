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
        let p1 = expand_array(p1, &p2);
        let p2 = expand_array(p2, &p1);

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
        self.p1.comp().apply(res_array);
        self.p2.comp().apply(res_array);
    }

    fn get_type(&self) -> ComputationType {
        ComputationType::Add
    }
}

impl <Other: Into<DArray>> Add<Other> for &DArray {
    type Output = DArray;
    fn add(self, rhs: Other) -> Self::Output {
        DArray::from(AddComp::new(self.clone(), rhs.into()))
    }
}
impl <Other: Into<DArray>> Add<Other> for DArray {
    type Output = DArray;
    fn add(self, rhs: Other) -> Self::Output {
        DArray::from(AddComp::new(self, rhs.into()))
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
        let p1 = expand_array(p1, &p2);
        let p2 = expand_array(p2, &p1);
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
}

impl <Other: DArrayRef> Mul<Other> for &DArray {
    type Output = DArray;

    fn mul(self, rhs: Other) -> Self::Output {
        DArray::from(MulComp::new(self.clone(), rhs.into()))
    }
}
impl <Other: DArrayRef> Mul<Other> for DArray {
    type Output = DArray;

    fn mul(self, rhs: Other) -> Self::Output {
        DArray::from(MulComp::new(self, rhs.into()))
    }
}



impl <Other: Into<DArray>> Div<Other> for DArray {
    type Output = DArray;

    fn div(self, rhs: Other) -> Self::Output {
        DArray::from(MulComp::new(self, rhs.into().powi(-1)))
    }
}
impl <Other: Into<DArray>> Div<Other> for &DArray {
    type Output = DArray;

    fn div(self, rhs: Other) -> Self::Output {
        DArray::from(MulComp::new(self.clone(), rhs.into().powi(-1)))
    }
}


#[cfg(test)]
mod tests {
    use crate::array::DArray;
    use crate::test_utils::*;


    /// Tests a generic binary function.
    fn test_binary(func: impl Fn(DArray, DArray) -> DArray) {
        let mut rng = StdRng::from_seed(SEED);
        for _ in 0..100 {
            let v1: f64 = rng.gen::<f64>() * 100. - 50.;
            let v2: f64 = rng.gen::<f64>() * 100. - 50.;
            let d1: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v1 * (1. - DIFF);
            let d2: f64 = (rng.gen::<f64>() * 100. - 50.) * DIFF + v2 * (1. - DIFF);

            let array1 = DArray::from(v1);
            let array2 = DArray::from(v2);
            let diff1 = DArray::from(d1);
            let diff2 = DArray::from(d2);

            let calc = func(array1.clone(), array2.clone());
            let calc_d1 = func(diff1.clone(), array2.clone());
            let calc_d2 = func(array1.clone(), diff2.clone());

            let grad_map = func(array1.clone(), array2.clone()).derive();

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
