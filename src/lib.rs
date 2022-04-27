pub mod comps;
pub mod context;
pub mod node;
pub mod unary_functions;

#[cfg(test)]
mod tests {
    use crate::comps::IndexComp;
    use crate::context::Context;
    use crate::node::Node;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn main() {
        let ctx = Context::new();

        let root_1 = Node::from_data(&[-6., 1.], ctx.nodes.clone());
        let root_2 = Node::from_data(&[4., 2.], ctx.nodes.clone());

        let mul_node = &root_1 * &root_2;
        let add_node = &mul_node + &root_1;
        let res = IndexComp::apply(&add_node, (0..2).map(|i| (i, 0)), 1);
        let grads = ctx.derive(res.clone());
        println!("res: {:?}", &res.data);

        println!("{:?}", &add_node.data);
        println!("{:?}", &grads.get(&root_1).unwrap().data);
        println!("{:?}", &grads.get(&root_2).unwrap().data);

        let c = res.sin();
        println!("c={:?}", &c.data);
        println!("c'={:?}", &ctx.derive(c.clone()).get(&res).unwrap().data);
    }
}
