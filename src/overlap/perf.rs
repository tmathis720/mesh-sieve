// Iteration order of these aliases is **not** relied upon for determinism; all
// public mutation paths sort their inputs before insertion.

#[cfg(all(feature = "fast-hash", not(feature = "deterministic-order")))]
pub type FastSet<T> = ahash::AHashSet<T>;

#[cfg(feature = "deterministic-order")]
pub type FastSet<T> = std::collections::BTreeSet<T>;

#[cfg(not(any(feature = "fast-hash", feature = "deterministic-order")))]
pub type FastSet<T> = std::collections::HashSet<T>;

#[cfg(all(feature = "fast-hash", not(feature = "deterministic-order")))]
pub type FastMap<K, V> = ahash::AHashMap<K, V>;

#[cfg(feature = "deterministic-order")]
pub type FastMap<K, V> = std::collections::BTreeMap<K, V>;

#[cfg(not(any(feature = "fast-hash", feature = "deterministic-order")))]
pub type FastMap<K, V> = std::collections::HashMap<K, V>;
