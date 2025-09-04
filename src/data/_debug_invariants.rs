use crate::mesh_error::MeshSieveError;

/// Trait for validating data structure invariants.
pub trait DebugInvariants {
    /// Assert invariants in debug builds or when `check-invariants` feature is enabled.
    fn debug_assert_invariants(&self);

    /// Validate invariants and return the first error encountered.
    fn validate_invariants(&self) -> Result<(), MeshSieveError>;
}

/// Helper macro to run a fallible check and panic on error when invariant
/// checking is enabled.
#[macro_export]
macro_rules! data_debug_assert_ok {
    ($expr:expr, $($ctx:tt)*) => {
        #[cfg(any(debug_assertions, feature = "check-invariants"))]
        if let Err(e) = $expr {
            panic!(concat!("[data invariants] ", $($ctx)*, ": {}"), e);
        }
    };
}
