//! Class labels.

use std::fmt::Debug;
use std::hash::Hash;

/// A class label: anything ordered, hashable and printable (integers, `bool`, `char`, strings).
///
/// `default_positive` picks the positive class of a binary task when none is given: for numbers,
/// `1` when every label is in {−1, 0, 1} (as EvalSuite's Python API); for `bool`, `true`; otherwise none, and
/// binary metrics then ask for an explicit `pos_label`.
pub trait Label: Clone + Ord + Hash + Debug + Send + Sync {
    fn default_positive(labels: &[Self]) -> Option<Self>;
}

macro_rules! int_label {
    ($($t:ty),*) => {$(
        impl Label for $t {
            #[allow(unused_comparisons)]
            fn default_positive(labels: &[Self]) -> Option<Self> {
                labels
                    .iter()
                    .all(|&l| l == 0 || l == 1 || (l as i128) == -1)
                    .then_some(1)
            }
        }
    )*};
}
int_label!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl Label for bool {
    fn default_positive(_: &[Self]) -> Option<Self> {
        Some(true)
    }
}

impl Label for char {
    fn default_positive(_: &[Self]) -> Option<Self> {
        None
    }
}

impl Label for String {
    fn default_positive(_: &[Self]) -> Option<Self> {
        None
    }
}

impl Label for &str {
    fn default_positive(_: &[Self]) -> Option<Self> {
        None
    }
}
