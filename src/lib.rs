#![feature(once_cell)]

pub mod comps;
pub mod node;
pub mod unary_functions;

#[cfg(test)]
mod tests {
    use std::lazy::Lazy;
    use crate::comps::IndexComp;
    use crate::node::Node;

    #[test]
    fn it_works() {
        let l: Lazy<i32> = Lazy::new(||5);
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn main() {
        let root_1 = Node::from_data(&[-6., 1.]);
        let root_2 = Node::from_data_and_node(&[4., 2.], &root_1);

        let mul_node = &root_1 * &root_2;
        let add_node = &mul_node + &root_1;
        let res = IndexComp::apply(&add_node, (0..2).map(|i| (i, 0)), 1);
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
