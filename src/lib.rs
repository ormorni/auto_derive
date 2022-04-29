extern crate core;

pub mod comps;
pub mod node;
pub mod unary_functions;
pub mod binary_functions;
pub mod index_functions;

pub use crate::node::Node;

#[cfg(test)]
mod tests {
    use crate::index_functions::IndexComp;
    use crate::Node;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn main() {
        let root_1 = Node::from_data(&[-6., 1.]);
        let root_2 = Node::from_data(&[4., 2.]);
        let mul_node = &root_1 * &root_2;
        let add_node = &mul_node + &root_1;
        let res = IndexComp::map_indices(&add_node, (0..2).map(|i| (i, 0)), 1);
        let grads = res.derive();
        println!("res: {:?}", &res.data());

        println!("{:?}", &add_node.data());
        println!("{:?}", &grads.get(&root_1).unwrap().data());
        println!("{:?}", &grads.get(&root_2).unwrap().data());

        let c = res.sin();
        println!("c={:?}", &c.data());
        println!("c'={:?}", c.derive().get(&res).unwrap().data());
    }
}
