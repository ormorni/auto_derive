use crate::node::Node;

/// A trait representing the computations which were used to generate nodes in the computation graph.
/// Used to perform the backward propagation.
pub trait Computation {
    /// Returns a vector of the parent nodes involved in the computation.
    fn sources(&self) -> Vec<Node>;
    /// Calculates the derivatives of the computation by each of the parent nodes.
    fn derivatives(&self, res_grads: Node) -> Vec<Node>;
    /// Calculates the function and adds the result to the given array.
    fn apply(&self, res_array: &mut [f64]);
    /// The length of the result array.
    fn len(&self) -> usize;
}

/// A computation that does nothing.
/// Used for nodes which are initialized from raw data, and are not computed.
#[derive(Copy, Clone)]
pub struct NullComp {
}

impl Computation for NullComp {
    fn sources(&self) -> Vec<Node> {
        vec![]
    }

    fn derivatives(&self, _: Node) -> Vec<Node> {
        vec![]
    }

    fn apply(&self, _: &mut [f64]) {}

    /// The null computation can't generate a length, and can't reproduce its data.
    fn len(&self) -> usize {
        panic!()
    }
}

#[derive(Clone)]
pub struct FromDataComp {
    pub(crate) data: Vec<f64>,
}

impl Computation for FromDataComp {
    fn sources(&self) -> Vec<Node> {
        vec![]
    }

    fn derivatives(&self, _: Node) -> Vec<Node> {
        vec![]
    }

    fn apply(&self, res: &mut [f64]) {
        assert_eq!(self.data.len(), res.len());
        for i in 0..self.len() {
            res[i] += self.data[i];
        }
    }

    /// The null computation can't generate a length, and can't reproduce its data.
    fn len(&self) -> usize {
        self.data.len()
    }
}


