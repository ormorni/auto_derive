use std::cell::{Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use crate::Rf;
use crate::comps::{Computation, NullComp};

pub struct Node {
    pub data: Vec<f64>,
    pub comp: Box<dyn Computation>,
    pub alloc: Rf<Vec<NodeRef>>,
    id: usize,
}

impl Node {
    pub fn from_data(data: &[f64], alloc: Rf<Vec<NodeRef>>) -> NodeRef {
        Node::from_comp(data, Box::new(NullComp {}), alloc)
    }

    pub fn from_comp(data: &[f64], comp: Box<dyn Computation>, alloc: Rf<Vec<NodeRef>>) -> NodeRef {
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

#[derive(Clone)]
pub struct NodeRef {
    node: Rc<Node>
}

impl NodeRef {
    pub fn new(node: Node) -> Self {
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


