use super::sieve_trait::Sieve;

/// Bulk arrow insertion helpers for [`Sieve`] implementations.
///
/// These methods allow inserting many edges in one pass while
/// pre-reserving capacities and invalidating caches only once.
pub trait SieveBuildExt: Sieve {
    /// Insert many arrows at once.
    ///
    /// For repeated `(src,dst)` pairs the last payload wins.
    fn add_arrows_from<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = (Self::Point, Self::Point, Self::Payload)>;

    /// Insert many arrows, deduplicating identical `(src,dst)` pairs prior to
    /// insertion. If duplicates are present in the input, the last payload wins.
    fn add_arrows_dedup_from<I>(&mut self, edges: I)
    where
        I: IntoIterator<Item = (Self::Point, Self::Point, Self::Payload)>;
}
