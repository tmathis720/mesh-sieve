// feature-gate the whole refine module
#![cfg(feature = "data_refine")]
pub mod helpers;
pub mod sieved_array;
pub mod delta;

// re-export the main pieces at the top level:
pub use helpers::{restrict_closure, restrict_star, restrict_closure_vec, restrict_star_vec};
pub use sieved_array::SievedArray;
pub use delta::Delta;
pub type Sifter = Vec<(crate::topology::point::PointId, crate::topology::arrow::Orientation)>;
