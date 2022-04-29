use crate::comps::{Computation, NullComp};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use fxhash::FxHashMap;
use itertools::izip;

type Map<K, V> = FxHashMap<K, V>;


/// A wrapper over a float array, which dynamically creates a computation graph.
/// The computation graph can then be used to automatically calculate derivatives of complex functions
/// using backward propagation.
/// It is never used directly, and only used inside a NodeRef.
struct NodeInternal {
    /// The data stored by the node.
    data: Vec<f64>,
    /// The computation used to calculate the node. Tracks the computation graph.
    comp: Box<dyn Computation>,
    /// An allocator used to allocate new nodes of the computation graph.
    alloc: Rc<RefCell<Vec<Node>>>,
    /// An ID, used to easily sort the nodes by order of creation.
    id: usize,
}

impl NodeInternal {

}

/// A struct proving a comfortable handle for the actual nodes.
#[derive(Clone)]
pub struct Node {
    node: Rc<NodeInternal>,
}

impl Node {
    /// A constructor for node references from a raw node.
    /// Used only in the Node's constructors.
    pub(crate) fn new(node: NodeInternal) -> Self {
        Node {
            node: Rc::new(node),
        }
    }

    /// Initializes a node from a slice of floats.
    pub fn from_data(data: &[f64]) -> Node {
        Node::from_comp(data, Box::new(NullComp {}), Rc::new(RefCell::new(Vec::new())))
    }

    pub fn from_data_and_node(data: &[f64], node: &Node) -> Node {
        Node::from_comp(data, Box::new(NullComp {}), node.alloc().clone())
    }

    /// Initializes a node from a slice of floats and the computation used to calculate it.
    pub fn from_comp(
        data: &[f64],
        comp: Box<dyn Computation>,
        alloc: Rc<RefCell<Vec<Node>>>,
    ) -> Node {
        let mut nodes = (*alloc).borrow_mut();
        let len = nodes.len();

        (*nodes).push(Node::new(NodeInternal {
            data: data.to_vec(),
            comp,
            alloc: alloc.clone(),
            id: len,
        }));

        (*nodes)[len].clone()
    }

    pub fn len(&self) -> usize {
        self.node.data.len()
    }

    pub fn data(&self) -> &Vec<f64> {
        &self.node.data
    }

    pub fn comp(&self) -> &Box<dyn Computation> {
        &self.node.comp
    }

    pub fn alloc(&self) -> Rc<RefCell<Vec<Node>>> {
        self.node.alloc.clone()
    }

    /// Calculates the derivative of the target value with respect to all intermediates in the computation graph.
    pub fn derive(&self) -> Map<Node, Node> {
        assert_eq!(
            self.len(),
            1,
            "Derivatives are supported only for Nodes with a single element! Real len is {}",
            self.len()
        );

        // Initializing derivation data structures.
        let mut grads = Map::default();
        grads.insert(self.clone(), Node::from_data_and_node(&[1.], self));

        let mut stack = BinaryHeap::new();
        stack.push(self.clone());

        while let Some(node) = stack.pop() {
            let node_grads = grads.get(&node).unwrap();
            let sources = node.comp().sources();
            let source_grads = node.comp().derivatives(node_grads.clone());

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

impl Eq for NodeInternal {}

impl PartialEq<Self> for NodeInternal {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd<Self> for NodeInternal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeInternal {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl Hash for NodeInternal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl Eq for Node {}

impl PartialEq<Self> for Node {
    fn eq(&self, other: &Self) -> bool {
        self.node.deref().eq(&*other.node.deref())
    }
}

impl PartialOrd<Self> for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.node.deref().partial_cmp(&*other.node.deref())
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node.deref().cmp(other.node.deref())
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.deref().hash(state)
    }
}
