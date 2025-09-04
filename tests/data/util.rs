use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub fn rng() -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(0xDEADBEEF)
}

pub fn increasing<T: From<usize> + Copy>(len: usize) -> Vec<T> {
    (0..len).map(Into::into).collect()
}
