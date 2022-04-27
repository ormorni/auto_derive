use std::cell::RefCell;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use crate::comps::{Computation, NullComp};


/// A wrapper over a float array, which dynamically creates a computation graph.
/// The computation graph can then be used to automatically calculate derivatives of complex functions
/// using backward propagation.
/// It is never used directly, and only used inside a NodeRef.
pub struct Node {
    /// The data stored by the node.
    pub data: Vec<f64>,
    /// The computation used to calculate the node. Tracks the computation graph.
    pub comp: Box<dyn Computation>,
    /// An allocator used to allocate new nodes of the computation graph.
    pub alloc: Rc<RefCell<Vec<NodeRef>>>,
    /// An ID, used to easily sort the nodes by order of creation.
    id: usize,
}

impl Node {
    /// Initializes a node from a slice of floats.
    pub fn from_data(data: &[f64], alloc: Rc<RefCell<Vec<NodeRef>>>) -> NodeRef {
        Node::from_comp(data, Box::new(NullComp {}), alloc)
    }

    /// Initializes a node from a slice of floats and the computation used to calculate it.
    pub fn from_comp(data: &[f64], comp: Box<dyn Computation>, alloc: Rc<RefCell<Vec<NodeRef>>>) -> NodeRef {
        let mut nodes = (*alloc).borrow_mut();
        let len = nodes.len();

        (*nodes).push(NodeRef::new(Node {
            data: data.to_vec(),
            comp,
            alloc: alloc.clone(),
            id: len,
        }));

        (*nodes)[len].clone()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// A struct proving a comfortable handle for the actual nodes.
#[derive(Clone)]
pub struct NodeRef {
    node: Rc<Node>
}

impl NodeRef {
    /// A constructor for node references from a raw node.
    /// Used only in the Node's constructors.
    pub(crate) fn new(node: Node) -> Self {
        NodeRef {node: Rc::new(node)}
    }
}

impl Deref for NodeRef {
    type Target = Node;

    fn deref(&self) -> &Self::Target {
        self.node.deref()
    }
}

impl Eq for Node {}

impl PartialEq<Self> for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl PartialOrd<Self> for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl Eq for NodeRef {}

impl PartialEq<Self> for NodeRef {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(&*other.deref())
    }
}

impl PartialOrd<Self> for NodeRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.deref().partial_cmp(&*other.deref())
    }
}

impl Ord for NodeRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.deref().cmp(other.deref())
    }
}

impl Hash for NodeRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state)
    }
}


