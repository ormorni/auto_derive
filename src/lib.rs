extern crate core;

pub mod computation;
pub mod array;
pub mod unary_functions;
pub mod binary_functions;
pub mod index_functions;
mod test_utils;

pub use crate::array::DArray;
pub use crate::index_functions::IndexComp;

#[cfg(test)]
mod tests {
    use crate::IndexComp;
    use crate::DArray;

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }

    #[test]
    fn main() {
        let root_1 = DArray::from(vec![-6., 1.]);
        let root_2 = DArray::from(vec![4., 2.]);
        let mul_array = &root_1 * &root_2;
        let add_array = &mul_array + &root_1;
        let res = IndexComp::map_indices(&add_array, (0..2).map(|i| (i, 0)), 1);
        let grads = res.derive();
        println!("res: {:?}", &res.data());

        println!("{:?}", &add_array.data());
        println!("{:?}", &grads.get(&root_1).unwrap().data());
        println!("{:?}", &grads.get(&root_2).unwrap().data());

        let c = res.sin();
        println!("c={:?}", &c.data());
        println!("c'={:?}", c.derive().get(&res).unwrap().data());
    }
}
