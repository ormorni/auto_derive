mod node;
mod comps;

use std::borrow::Borrow;
use std::cell::{Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::rc::Rc;
use fxhash::FxHashMap;
use itertools::izip;
use crate::node::{Node, NodeRef};

type Rf<T> = Rc<RefCell<T>>;
type Map<K, V> = FxHashMap<K, V>;


pub struct Context {
    nodes: Rf<Vec<NodeRef>>,
}
impl Context {
    fn new() -> Self {
        Context {nodes: Rc::new(RefCell::new(Vec::new()))}
    }

    fn derive(&self, res: NodeRef) -> Map<NodeRef, NodeRef> {
        assert_eq!(res.borrow().len(), 1);

        // Initializing derivation data structures.
        let mut grads = Map::default();
        grads.insert(res.clone(), Node::from_data(&[1.], res.borrow().alloc.clone()));

        let mut stack = BinaryHeap::new();
        stack.push(res);

        while let Some(node) = stack.pop() {
            let node_grads = grads.get(&node).unwrap();
            let sources = node.borrow().comp.sources();
            let source_grads = node.borrow().comp.backpropagate(node_grads.clone());

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


#[cfg(test)]
mod tests {
    use std::borrow::Borrow;
    use crate::{Context, Node};

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }


    #[test]
    fn main() {
        let ctx = Context::new();

        let mut root_1 = Node::from_data(&[6.], ctx.nodes.clone());
        let mut root_2 = Node::from_data(&[4.], ctx.nodes.clone());

        let mul_node = &root_1 * &root_2;
        let add_node = &mul_node + &root_1;
        let grads = ctx.derive(add_node.clone());

        println!("{:?}", &add_node.borrow().data);
        println!("{:?}", &grads.get(&root_1).unwrap().borrow().data);
        println!("{:?}", &grads.get(&root_2).unwrap().borrow().data);
    }
}