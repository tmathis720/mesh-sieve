//! Concrete traversal iterators for Sieve topologies.
//!
//! Two families are provided:
//! - **Value-based**: [`ClosureIter`], [`StarIter`], [`ClosureBothIter`]
//!   which visit `(Point, Payload)` pairs and therefore clone payloads during
//!   traversal.
//! - **Borrow-based**: [`ClosureIterRef`], [`StarIterRef`],
//!   [`ClosureBothIterRef`] which require [`SieveRef`] and operate purely on
//!   points, cloning no payloads.
//!
//! All iterators are depth-first with a `stack` + `seen` set and deterministic
//! "first seen wins" semantics. Deterministic constructors are available via
//! the `*_sorted` and `*_sorted_neighbors` constructors.
//!
//! Use via the helper methods on [`Sieve`]: `closure_iter*`, `star_iter*`,
//! or their `_ref` counterparts when `S: SieveRef`.

use std::collections::HashSet;

use super::sieve_ref::SieveRef;
use super::sieve_trait::Sieve;

#[derive(Copy, Clone, Debug)]
enum Dir {
    Down,
    Up,
    Both,
}

#[derive(Copy, Clone, Debug)]
enum NeighborOrder {
    AsIs,
    Sorted,
}

/// Depth-first traversal iterator with deterministic "first seen wins" semantics.
struct TraversalIter<'a, S: Sieve> {
    sieve: &'a S,
    stack: Vec<S::Point>,
    seen: HashSet<S::Point>,
    dir: Dir,
    norder: NeighborOrder,
}

impl<'a, S> TraversalIter<'a, S>
where
    S: Sieve,
{
    fn new<I>(sieve: &'a S, seeds: I, dir: Dir) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        Self::new_with_norder(sieve, seeds, dir, NeighborOrder::AsIs)
    }

    fn new_with_norder<I>(sieve: &'a S, seeds: I, dir: Dir, norder: NeighborOrder) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let stack: Vec<S::Point> = seeds.into_iter().collect();
        let seen: HashSet<S::Point> = stack.iter().copied().collect();
        Self {
            sieve,
            stack,
            seen,
            dir,
            norder,
        }
    }

    fn with_stack(sieve: &'a S, stack: Vec<S::Point>, dir: Dir) -> Self {
        let seen: HashSet<S::Point> = stack.iter().copied().collect();
        Self {
            sieve,
            stack,
            seen,
            dir,
            norder: NeighborOrder::AsIs,
        }
    }

    #[inline]
    fn push_down(&mut self, p: S::Point) {
        match self.norder {
            NeighborOrder::AsIs => {
                for (q, _) in self.sieve.cone(p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
            NeighborOrder::Sorted => {
                let mut buf: Vec<_> = self.sieve.cone(p).map(|(q, _)| q).collect();
                buf.sort_unstable();
                for q in buf.into_iter().rev() {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
        }
    }

    #[inline]
    fn push_up(&mut self, p: S::Point) {
        match self.norder {
            NeighborOrder::AsIs => {
                for (q, _) in self.sieve.support(p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
            NeighborOrder::Sorted => {
                let mut buf: Vec<_> = self.sieve.support(p).map(|(q, _)| q).collect();
                buf.sort_unstable();
                for q in buf.into_iter().rev() {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
        }
    }
}

impl<'a, S> Iterator for TraversalIter<'a, S>
where
    S: Sieve,
{
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(p) = self.stack.pop() {
            match self.dir {
                Dir::Down => self.push_down(p),
                Dir::Up => self.push_up(p),
                Dir::Both => {
                    self.push_down(p);
                    self.push_up(p);
                }
            }
            Some(p)
        } else {
            None
        }
    }
}

/// Downward closure iterator.
pub struct ClosureIter<'a, S: Sieve>(TraversalIter<'a, S>);
/// Upward star iterator.
pub struct StarIter<'a, S: Sieve>(TraversalIter<'a, S>);
/// Bidirectional closure iterator.
pub struct ClosureBothIter<'a, S: Sieve>(TraversalIter<'a, S>);

impl<'a, S: Sieve> ClosureIter<'a, S> {
    #[inline]
    pub fn new<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        ClosureIter(TraversalIter::new(sieve, seeds, Dir::Down))
    }

    /// Deterministic variant: seeds are sorted and deduped.
    #[inline]
    pub fn new_sorted<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<S::Point> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        ClosureIter(TraversalIter::with_stack(sieve, stack, Dir::Down))
    }

    /// Fully deterministic variant sorting neighbors on expansion.
    #[inline]
    pub fn new_sorted_neighbors<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<S::Point> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        let seen: HashSet<S::Point> = stack.iter().copied().collect();
        ClosureIter(TraversalIter {
            sieve,
            stack,
            seen,
            dir: Dir::Down,
            norder: NeighborOrder::Sorted,
        })
    }
}

impl<'a, S: Sieve> StarIter<'a, S> {
    #[inline]
    pub fn new<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        StarIter(TraversalIter::new(sieve, seeds, Dir::Up))
    }

    /// Deterministic variant: seeds are sorted and deduped.
    #[inline]
    pub fn new_sorted<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<S::Point> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        StarIter(TraversalIter::with_stack(sieve, stack, Dir::Up))
    }

    /// Fully deterministic variant sorting neighbors on expansion.
    #[inline]
    pub fn new_sorted_neighbors<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<S::Point> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        let seen: HashSet<S::Point> = stack.iter().copied().collect();
        StarIter(TraversalIter {
            sieve,
            stack,
            seen,
            dir: Dir::Up,
            norder: NeighborOrder::Sorted,
        })
    }
}

impl<'a, S: Sieve> ClosureBothIter<'a, S> {
    #[inline]
    pub fn new<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        ClosureBothIter(TraversalIter::new(sieve, seeds, Dir::Both))
    }

    /// Deterministic variant: seeds are sorted and deduped.
    #[inline]
    pub fn new_sorted<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<S::Point> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        ClosureBothIter(TraversalIter::with_stack(sieve, stack, Dir::Both))
    }

    /// Fully deterministic variant sorting neighbors on expansion.
    #[inline]
    pub fn new_sorted_neighbors<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<S::Point> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        let seen: HashSet<S::Point> = stack.iter().copied().collect();
        ClosureBothIter(TraversalIter {
            sieve,
            stack,
            seen,
            dir: Dir::Both,
            norder: NeighborOrder::Sorted,
        })
    }
}

impl<'a, S: Sieve> Iterator for ClosureIter<'a, S> {
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, S: Sieve> Iterator for StarIter<'a, S> {
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, S: Sieve> Iterator for ClosureBothIter<'a, S> {
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

/// Depth-first traversal iterator (borrow-based, point-only).
struct TraversalIterRef<'a, S: Sieve + SieveRef> {
    sieve: &'a S,
    stack: Vec<S::Point>,
    seen: HashSet<S::Point>,
    dir: Dir,
    norder: NeighborOrder,
}

impl<'a, S> TraversalIterRef<'a, S>
where
    S: Sieve + SieveRef,
{
    #[inline]
    fn with_stack(sieve: &'a S, stack: Vec<S::Point>, dir: Dir, norder: NeighborOrder) -> Self {
        let seen = stack.iter().copied().collect();
        Self {
            sieve,
            stack,
            seen,
            dir,
            norder,
        }
    }

    #[inline]
    fn push_down(&mut self, p: S::Point) {
        match self.norder {
            NeighborOrder::AsIs => {
                for q in SieveRef::cone_points(self.sieve, p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
            NeighborOrder::Sorted => {
                let mut buf: Vec<_> = SieveRef::cone_points(self.sieve, p).collect();
                buf.sort_unstable();
                for q in buf.into_iter().rev() {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
        }
    }

    #[inline]
    fn push_up(&mut self, p: S::Point) {
        match self.norder {
            NeighborOrder::AsIs => {
                for q in SieveRef::support_points(self.sieve, p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
            NeighborOrder::Sorted => {
                let mut buf: Vec<_> = SieveRef::support_points(self.sieve, p).collect();
                buf.sort_unstable();
                for q in buf.into_iter().rev() {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
        }
    }
}

impl<'a, S> Iterator for TraversalIterRef<'a, S>
where
    S: Sieve + SieveRef,
{
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let p = self.stack.pop()?;
        match self.dir {
            Dir::Down => self.push_down(p),
            Dir::Up => self.push_up(p),
            Dir::Both => {
                self.push_down(p);
                self.push_up(p);
            }
        }
        Some(p)
    }
}

/// Downward closure iterator (borrow-based).
pub struct ClosureIterRef<'a, S: Sieve + SieveRef>(TraversalIterRef<'a, S>);
/// Upward star iterator (borrow-based).
pub struct StarIterRef<'a, S: Sieve + SieveRef>(TraversalIterRef<'a, S>);
/// Bidirectional closure iterator (borrow-based).
pub struct ClosureBothIterRef<'a, S: Sieve + SieveRef>(TraversalIterRef<'a, S>);

impl<'a, S> ClosureIterRef<'a, S>
where
    S: Sieve + SieveRef,
{
    #[inline]
    pub fn new_ref<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let stack: Vec<_> = seeds.into_iter().collect();
        ClosureIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Down,
            NeighborOrder::AsIs,
        ))
    }

    /// Deterministic: seeds sorted/deduped.
    #[inline]
    pub fn new_ref_sorted<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<_> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        ClosureIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Down,
            NeighborOrder::AsIs,
        ))
    }

    /// Deterministic + neighbor-sorted expansion.
    #[inline]
    pub fn new_ref_sorted_neighbors<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<_> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        ClosureIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Down,
            NeighborOrder::Sorted,
        ))
    }
}

impl<'a, S> StarIterRef<'a, S>
where
    S: Sieve + SieveRef,
{
    #[inline]
    pub fn new_ref<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let stack: Vec<_> = seeds.into_iter().collect();
        StarIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Up,
            NeighborOrder::AsIs,
        ))
    }

    #[inline]
    pub fn new_ref_sorted<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<_> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        StarIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Up,
            NeighborOrder::AsIs,
        ))
    }

    #[inline]
    pub fn new_ref_sorted_neighbors<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<_> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        StarIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Up,
            NeighborOrder::Sorted,
        ))
    }
}

impl<'a, S> ClosureBothIterRef<'a, S>
where
    S: Sieve + SieveRef,
{
    #[inline]
    pub fn new_ref<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let stack: Vec<_> = seeds.into_iter().collect();
        ClosureBothIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Both,
            NeighborOrder::AsIs,
        ))
    }

    #[inline]
    pub fn new_ref_sorted<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<_> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        ClosureBothIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Both,
            NeighborOrder::AsIs,
        ))
    }

    #[inline]
    pub fn new_ref_sorted_neighbors<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let mut stack: Vec<_> = seeds.into_iter().collect();
        stack.sort_unstable();
        stack.dedup();
        ClosureBothIterRef(TraversalIterRef::with_stack(
            sieve,
            stack,
            Dir::Both,
            NeighborOrder::Sorted,
        ))
    }
}

impl<'a, S: Sieve + SieveRef> Iterator for ClosureIterRef<'a, S> {
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, S: Sieve + SieveRef> Iterator for StarIterRef<'a, S> {
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, S: Sieve + SieveRef> Iterator for ClosureBothIterRef<'a, S> {
    type Item = S::Point;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
