//! A small, fast, non-cryptographic hasher (the FxHash algorithm used inside rustc) for n-gram counting.
//! Keys are n-gram slices of token ids or code points, never untrusted lookups that need DoS resistance:
//! every map is built from the caller's own data and dropped at the end of the call.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes([
                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
            ]));
        }
        for &b in chunks.remainder() {
            self.add(u64::from(b));
        }
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Counts of the contiguous n-grams (as slices) of `seq`.
pub fn ngram_counts(seq: &[u32], n: usize) -> FxMap<&[u32], u32> {
    let mut m: FxMap<&[u32], u32> = FxMap::default();
    if n == 0 || seq.len() < n {
        return m;
    }
    m.reserve(seq.len() - n + 1);
    for w in seq.windows(n) {
        *m.entry(w).or_insert(0) += 1;
    }
    m
}
