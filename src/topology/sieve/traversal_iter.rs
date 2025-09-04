//! Concrete traversal iterators for Sieve topologies.
//!
//! These provide stack+seen traversal without dynamic dispatch.
//! Use via `Sieve::closure_iter`, `star_iter`, or `closure_both_iter`.

use std::collections::HashSet;

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
                #[cfg(feature = "sieve_point_only")]
                for q in self.sieve.cone_points(p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
                #[cfg(not(feature = "sieve_point_only"))]
                for (q, _) in self.sieve.cone(p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
            NeighborOrder::Sorted => {
                #[cfg(feature = "sieve_point_only")]
                {
                    let mut buf: Vec<_> = self.sieve.cone_points(p).collect();
                    buf.sort_unstable();
                    for q in buf.into_iter().rev() {
                        if self.seen.insert(q) {
                            self.stack.push(q);
                        }
                    }
                }
                #[cfg(not(feature = "sieve_point_only"))]
                {
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
    }

    #[inline]
    fn push_up(&mut self, p: S::Point) {
        match self.norder {
            NeighborOrder::AsIs => {
                #[cfg(feature = "sieve_point_only")]
                for q in self.sieve.support_points(p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
                #[cfg(not(feature = "sieve_point_only"))]
                for (q, _) in self.sieve.support(p) {
                    if self.seen.insert(q) {
                        self.stack.push(q);
                    }
                }
            }
            NeighborOrder::Sorted => {
                #[cfg(feature = "sieve_point_only")]
                {
                    let mut buf: Vec<_> = self.sieve.support_points(p).collect();
                    buf.sort_unstable();
                    for q in buf.into_iter().rev() {
                        if self.seen.insert(q) {
                            self.stack.push(q);
                        }
                    }
                }
                #[cfg(not(feature = "sieve_point_only"))]
                {
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
