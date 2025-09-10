#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// `inv_assert*` fire in debug builds and whenever the `strict-invariants`
/// feature is enabled; they compile to no-ops otherwise.
#[cfg(any(debug_assertions, feature = "strict-invariants"))]
macro_rules! inv_assert {
    ($($tt:tt)*) => { assert!($($tt)*); };
}
#[cfg(any(debug_assertions, feature = "strict-invariants"))]
macro_rules! inv_assert_eq {
    ($($tt:tt)*) => { assert_eq!($($tt)*); };
}
#[cfg(not(any(debug_assertions, feature = "strict-invariants")))]
macro_rules! inv_assert {
    ($($tt:tt)*) => {
        ()
    };
}
#[cfg(not(any(debug_assertions, feature = "strict-invariants")))]
macro_rules! inv_assert_eq {
    ($($tt:tt)*) => {
        ()
    };
}
pub(crate) use inv_assert;
pub(crate) use inv_assert_eq;

#[inline]
pub fn count_pairs<P: Copy + Eq + std::hash::Hash>(
    it: impl IntoIterator<Item = (P, P)>,
) -> HashMap<(P, P), u32> {
    let mut m = HashMap::new();
    for e in it {
        *m.entry(e).or_insert(0) += 1;
    }
    m
}

#[inline]
pub fn assert_no_dups_per_src<P, D>(out: &HashMap<P, Vec<(P, D)>>)
where
    P: Copy + Eq + std::hash::Hash + std::fmt::Debug,
    D: std::fmt::Debug,
{
    for (src, vec) in out {
        let mut seen = HashSet::with_capacity(vec.len());
        for (dst, _) in vec {
            let fresh = seen.insert(*dst);
            inv_assert!(fresh, "duplicate edge detected: src={src:?} dst={dst:?}");
        }
    }
}

#[inline]
pub fn counts_equal<P>(
    a: &HashMap<(P, P), u32>,
    b: &HashMap<(P, P), u32>,
    label_a: &str,
    label_b: &str,
) where
    P: Copy + Eq + std::hash::Hash + std::fmt::Debug,
{
    inv_assert_eq!(
        a.len(),
        b.len(),
        "edge multiset cardinality mismatch ({label_a} vs {label_b})",
    );
    for (k, va) in a {
        let Some(vb) = b.get(k) else {
            inv_assert!(
                false,
                "edge present in {label_a} but missing in {label_b}: {k:?}"
            );
            continue;
        };
        inv_assert_eq!(
            va,
            vb,
            "edge multiplicity mismatch for {k:?}: {label_a}={va}, {label_b}={vb}"
        );
    }
}

#[cfg(any(debug_assertions, feature = "strict-invariants"))]
macro_rules! debug_invariants {
    ($s:expr) => {
        $s.debug_assert_invariants();
    };
}

#[cfg(not(any(debug_assertions, feature = "strict-invariants")))]
macro_rules! debug_invariants {
    ($s:expr) => {
        ()
    };
}

pub(crate) use debug_invariants;
