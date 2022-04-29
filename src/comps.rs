use crate::node::Node;

/// A trait representing the computations which were used to generate nodes in the computation graph.
/// Used to perform the backward propagation.
pub trait Computation {
    /// Returns a vector of the parent nodes involved in the computation.
    fn sources(&self) -> Vec<Node>;
    /// Calculates the derivatives of the computation by each of the parent nodes.
    fn derivatives(&self, res_grads: Node) -> Vec<Node>;
}

/// A computation that does nothing.
/// Used for nodes which are initialized from raw data, and are not computed.
#[derive(Copy, Clone)]
pub struct NullComp {}

impl Computation for NullComp {
    fn sources(&self) -> Vec<Node> {
        vec![]
    }

    fn derivatives(&self, _: Node) -> Vec<Node> {
        vec![]
    }
}


