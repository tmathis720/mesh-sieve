use super::sieve_trait::Sieve;

/// Extension methods for common edge queries.
///
/// Provides degree counts and optional helpers built on top of the core
/// [`Sieve`] trait without modifying it.
pub trait SieveQueryExt: Sieve {
    /// Returns the out-degree (number of outgoing arrows) of `p`.
    ///
    /// Default implementation counts by iteration.
    fn out_degree(&self, p: Self::Point) -> usize {
        self.cone(p).count()
    }

    /// Returns the in-degree (number of incoming arrows) of `p`.
    ///
    /// Default implementation counts by iteration.
    fn in_degree(&self, p: Self::Point) -> usize {
        self.support(p).count()
    }
}
