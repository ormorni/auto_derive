#[cfg(feature = "benchmarks")]
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use rand::distributions::Uniform;
use auto_derive::DArray;

pub const SEED: [u8; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];


/// Tests that the derivative of complex random rational functions are evaluated correctly.
fn random_graph(seed: &[u8; 32]) -> (DArray, DArray) {
    let mut rng = StdRng::from_seed(*seed);

    let mut arr = vec![];
    for _ in 0..10 {
        let rand_vec = Vec::from_iter((&mut rng).sample_iter(Uniform::new(-0.5, 0.5)).take(1000));
        arr.push(DArray::from(rand_vec));
    }

    let root = arr[0].clone();

    for _ in 0..100 {
        let p1: usize = rng.gen_range(0..arr.len());
        let p2: usize = rng.gen_range(0..arr.len());
        let op: usize = rng.gen_range(0..3);

        match op {
            0 => {arr[p1] = &arr[p1] + &arr[p2]},
            1 => {arr[p1] = &arr[p1] * &arr[p2]},
            2 => {arr[p1] = &arr[p1] / &arr[p2]},
            _ => panic!()
        }
    }

    (root, arr[0].index(0))
}

fn computation(seed: &[u8; 32]) -> f64 {
    let (_, res) = random_graph(seed);
    res.data()[0]
}

fn derivation(seed: &[u8; 32]) -> DArray {
    let (root, res) = random_graph(seed);
    res.derive().get(&root).unwrap().clone()
}

#[cfg(feature = "benchmarks")]
fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("graph_generation", |b| b.iter(|| random_graph(black_box(&SEED))));
    c.bench_function("graph_evaluation", |b| b.iter(|| computation(black_box(&SEED))));
    c.bench_function("graph_derivation", |b| b.iter(|| derivation(black_box(&SEED))));
}

#[cfg(feature = "benchmarks")]
criterion_group!(benches, criterion_benchmark);
#[cfg(feature = "benchmarks")]
criterion_main!(benches);

#[cfg(not(feature = "benchmarks"))]
fn main() {
}
