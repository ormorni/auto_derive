/// Reexporting RNG imports.
pub use rand::prelude::{StdRng, Rng};
pub use rand::SeedableRng;
pub use test_utils::*;

#[allow(dead_code)]
pub mod test_utils {
    /// The seed used for random number generation in tests.
    pub const SEED: [u8; 32] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31];
    /// The difference used for numeric differentiation.
    pub const DIFF: f64 = 1e-7;
    /// The allowed error between expected and observed results.
    pub const ALLOWED_ERROR: f64 = 1e-3;

    /// Asserts that two floating point numbers are close to each other.
    /// Tests that the ratio of the difference and the average is smaller than the allowed value.
    pub fn assert_close(a: f64, b: f64) {
        if a != b {
            let error = (a - b).abs() * 2. / (a.abs() + b.abs());
            assert!(error < ALLOWED_ERROR, "Values are not close: a={} b={} error={}", a, b, error);
        }
    }
}
