//! Data module: section and atlas
#![warn(missing_docs)]

pub mod atlas;
pub mod bundle;
pub mod section;

mod _debug_invariants;
pub mod refine;

pub(crate) use _debug_invariants::DebugInvariants;
