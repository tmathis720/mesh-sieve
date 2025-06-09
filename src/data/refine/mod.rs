pub mod delta;
pub mod helpers;
pub mod sieved_array;

// re-export the main pieces at the top level:
pub use delta::Delta;
pub use helpers::{restrict_closure, restrict_closure_vec, restrict_star, restrict_star_vec};
pub use sieved_array::SievedArray;
pub type Sifter = Vec<(
    crate::topology::point::PointId,
    crate::topology::arrow::Orientation,
)>;
