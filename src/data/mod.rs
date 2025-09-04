//! Data module: section and atlas

pub mod atlas;
pub mod section;
pub mod bundle;

pub mod refine;
mod _debug_invariants;

pub(crate) use _debug_invariants::DebugInvariants;
