pub mod sieve_trait;
pub mod in_memory;

// Re-export the core trait and in‐memory impl at top level
pub use sieve_trait::Sieve;
pub use in_memory::InMemorySieve;
