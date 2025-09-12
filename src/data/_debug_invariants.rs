#[macro_export]
#[deprecated(note = "Use debug_invariants! macro")]
macro_rules! data_debug_assert_ok {
    ($($t:tt)*) => { $crate::debug_invariants!($($t)*); };
}
