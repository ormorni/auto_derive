use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::rc::Rc;
use fxhash::FxHashMap;
use itertools::izip;
use crate::node::{Node, NodeRef};

type Map<K, V> = FxHashMap<K, V>;


/// A computation graph context.
/// Used to store the computation graph and perform derivations on it.
pub struct Context {
    pub nodes: Rc<RefCell<Vec<NodeRef>>>,
}

impl Context {
    /// Initializes an new computation context.
    pub fn new() -> Self {
        Context {nodes: Rc::new(RefCell::new(Vec::new()))}
    }

    /// Calculates the derivative of the target value with respect to all intermediates in the computation graph.
    pub fn derive(&self, res: NodeRef) -> Map<NodeRef, NodeRef> {
        assert_eq!(res.len(), 1, "Derivatives are supported only for Nodes with a single element! Real len is {}", res.len());

        // Initializing derivation data structures.
        let mut grads = Map::default();
        grads.insert(res.clone(), Node::from_data(&[1.], res.alloc.clone()));

        let mut stack = BinaryHeap::new();
        stack.push(res);

        while let Some(node) = stack.pop() {
            let node_grads = grads.get(&node).unwrap();
            let sources = node.comp.sources();
            let source_grads = node.comp.derivatives(node_grads.clone());

            for (source, grad) in izip!(sources.iter(), source_grads.iter()) {
                let old_grad = grads.get(source);
                if old_grad.is_none() {
                    grads.insert(source.clone(), grad.clone());
                    stack.push(source.clone());
                } else {
                    let old_grad = old_grad.unwrap();
                    let new_grad = old_grad + grad;
                    grads.insert(source.clone(), new_grad);
                }
            }
        }

        grads
    }
}