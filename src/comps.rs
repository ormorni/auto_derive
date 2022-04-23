use std::ops::{Add, Mul};
use itertools::izip;
use crate::{Node, NodeRef};

pub trait Computation {
    fn sources(&self) -> Vec<NodeRef>;
    fn backpropagate(&self, res_grads: NodeRef) -> Vec<NodeRef>;
}

#[derive(Copy, Clone)]
pub struct NullComp {
}

impl Computation for NullComp {
    fn sources(&self) -> Vec<NodeRef> {
        vec![]
    }

    fn backpropagate(&self, _: NodeRef) -> Vec<NodeRef> {
        vec![]
    }
}

#[derive(Clone)]
struct AddComp {
    p1: NodeRef,
    p2: NodeRef,
}

impl <'t> AddComp {
    fn apply(p1: &NodeRef, p2: &NodeRef) -> NodeRef {
        assert_eq!(p1.borrow().len(), p2.borrow().len());
        let data: Vec<f64> = izip!(p1.borrow().data.iter(), p2.borrow().data.iter()).map(|(v1, v2)|v1 + v2).collect();
        let comp = Box::new(AddComp {p1: p1.clone(), p2: p2.clone()});
        let node = Node::from_comp(&data, comp, p1.borrow().alloc.clone());
        node
    }
}

impl Computation for AddComp {
    fn sources(&self) -> Vec<NodeRef> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn backpropagate(&self, res_grads: NodeRef) -> Vec<NodeRef> {
        vec![res_grads.clone(), res_grads.clone()]
    }
}

impl Add for &NodeRef {
    type Output = NodeRef;

    fn add(self, rhs: Self) -> Self::Output {
        AddComp::apply(self, rhs)
    }
}


#[derive(Clone)]
struct MulComp {
    p1: NodeRef,
    p2: NodeRef,
}

impl <'t> MulComp {
    fn apply(p1: &NodeRef, p2: &NodeRef) -> NodeRef {
        assert_eq!(p1.borrow().len(), p2.borrow().len());
        let data: Vec<f64> = izip!(p1.borrow().data.iter(), p2.borrow().data.iter()).map(|(v1, v2)|v1 * v2).collect();
        let comp = Box::new(MulComp {p1: p1.clone(), p2: p2.clone()});
        let node = Node::from_comp(&data, comp, p1.borrow().alloc.clone());
        node
    }
}

impl Computation for MulComp {
    fn sources(&self) -> Vec<NodeRef> {
        vec![self.p1.clone(), self.p2.clone()]
    }

    fn backpropagate(&self, res_grads: NodeRef) -> Vec<NodeRef> {
        vec![MulComp::apply(&self.p2, &res_grads), MulComp::apply(&self.p1, &res_grads)]
    }
}

impl Mul for &NodeRef {
    type Output = NodeRef;

    fn mul(self, rhs: Self) -> Self::Output {
        MulComp::apply(self, rhs)
    }
}