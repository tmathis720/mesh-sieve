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

/// Depth-first traversal iterator with deterministic "first seen wins" semantics.
struct TraversalIter<'a, S: Sieve> {
    sieve: &'a S,
    stack: Vec<S::Point>,
    seen: HashSet<S::Point>,
    dir: Dir,
}

impl<'a, S> TraversalIter<'a, S>
where
    S: Sieve,
{
    fn new<I>(sieve: &'a S, seeds: I, dir: Dir) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        let stack: Vec<S::Point> = seeds.into_iter().collect();
        let seen: HashSet<S::Point> = stack.iter().copied().collect();
        Self { sieve, stack, seen, dir }
    }

    #[inline]
    fn push_down(&mut self, p: S::Point) {
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

    #[inline]
    fn push_up(&mut self, p: S::Point) {
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
}

impl<'a, S: Sieve> StarIter<'a, S> {
    #[inline]
    pub fn new<I>(sieve: &'a S, seeds: I) -> Self
    where
        I: IntoIterator<Item = S::Point>,
    {
        StarIter(TraversalIter::new(sieve, seeds, Dir::Up))
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

