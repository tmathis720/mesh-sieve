//! Strata computation utilities for sieves.
//!
//! This module provides [`StrataCache`], a structure for storing precomputed height, depth, and strata
//! information for points in a sieve, as well as [`compute_strata`], a function to compute these values
//! for any sieve instance.
//!
//! # Errors
//! Returns `Err(MeshSieveError::MissingPointInCone(p))` if a `cone` points to `p` not in `points()`,
//! or `Err(MeshSieveError::CycleDetected)` if the topology contains a cycle.

use crate::topology::sieve::Sieve;
use crate::mesh_error::MeshSieveError;
use std::collections::HashMap;

/// Precomputed stratum information for a sieve.
///
/// Stores the height, depth, and strata (points at each height) for efficient queries,
/// as well as the diameter (maximum height) of the structure.
#[derive(Clone, Debug)]
pub struct StrataCache<P> {
    /// Map from point to its height (distance from any zero-in-degree source).
    pub height:   HashMap<P,u32>,
    /// Map from point to its depth (distance down to any zero-out-degree sink).
    pub depth:    HashMap<P,u32>,
    /// Vectors of points at each height: `strata[h] = points at height h`.
    pub strata:   Vec<Vec<P>>,
    /// Maximum height (diameter) of the sieve.
    pub diameter: u32,
}

impl<P: Copy + Eq + std::hash::Hash + Ord> StrataCache<P> {
    /// Create a new, empty `StrataCache`.
    pub fn new() -> Self {
        Self { height: HashMap::new(), depth: HashMap::new(), strata: Vec::new(), diameter: 0 }
    }
}

/// Compute strata information for any sieve instance on-the-fly (no cache).
///
/// Returns a [`StrataCache`] containing height, depth, strata, and diameter information for all points.
///
/// # Errors
/// Returns `Err(MeshSieveError::MissingPointInCone(p))` if a `cone` points to `p` not in `points()`,
/// or `Err(MeshSieveError::CycleDetected)` if the topology contains a cycle.
pub fn compute_strata<S>(
    s: &mut S
) -> Result<StrataCache<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: Copy + Eq + std::hash::Hash + Ord + std::fmt::Debug,
{
    // 1) collect in-degrees over s.points()
    let mut in_deg = HashMap::new();
    for p in s.points() {
        in_deg.entry(p).or_insert(0);
        for (q,_) in s.cone(p) { *in_deg.entry(q).or_insert(0) += 1; }
    }
    // 2) Kahn’s topo sort
    let mut stack: Vec<_> = in_deg.iter().filter(|&(_,d)| *d==0).map(|(&p,_)| p).collect();
    let mut topo = Vec::new();
    while let Some(p) = stack.pop() {
        topo.push(p);
        for (q,_) in s.cone(p) {
            let d = in_deg
                .get_mut(&q)
                .ok_or_else(|| MeshSieveError::MissingPointInCone(format!("{:?}", q)))?;
            *d -= 1;
            if *d==0 { stack.push(q); }
        }
    }
    // 3) detect cycles
    if topo.len() != in_deg.len() {
        return Err(MeshSieveError::CycleDetected);
    }
    // 4) heights
    let mut height = HashMap::new();
    for &p in &topo {
        let h = s.support(p)
                .map(|(pred,_)| height.get(&pred).copied().unwrap_or(0))
                .max().map_or(0, |m| m+1);
        height.insert(p,h);
    }
    let max_h = *height.values().max().unwrap_or(&0);
    let mut strata = vec![Vec::new(); (max_h+1) as usize];
    for (&p,&h) in &height { strata[h as usize].push(p) }
    // 5) depths
    let mut depth = HashMap::new();
    for &p in topo.iter().rev() {
        let d = s.cone(p)
                .map(|(succ,_)| depth.get(&succ).copied().unwrap_or(0))
                .max().map_or(0, |m| m+1);
        depth.insert(p,d);
    }
    Ok(StrataCache { height, depth, strata, diameter: max_h })
}
