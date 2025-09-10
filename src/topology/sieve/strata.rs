//! Strata computation utilities for sieves.
//!
//! This module provides [`StrataCache`], a structure for storing precomputed height, depth, and strata
//! information for points in a sieve, as well as [`compute_strata`], a function to compute these values
//! for any sieve instance.
//!
//! # Errors
//! * [`MeshSieveError::MissingPointInCone`]: an arrow references a point **not present in
//!   `base_points ∪ cap_points`**. The error aggregates all such points (examples provided) and is
//!   raised before any topological pass.
//! * [`MeshSieveError::CycleDetected`]: the topology contains a cycle.

use crate::mesh_error::MeshSieveError;
use crate::topology::bounds::PointLike;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

/// Precomputed stratum information for a sieve.
///
/// Stores the height, depth, and strata (points at each height) for efficient queries,
/// as well as the diameter (maximum height) of the structure.
#[derive(Clone, Debug)]
pub struct StrataCache<P> {
    /// Map from point to its height (distance from any zero-in-degree source).
    pub height: HashMap<P, u32>,
    /// Map from point to its depth (distance down to any zero-out-degree sink).
    pub depth: HashMap<P, u32>,
    /// Vectors of points at each height: `strata[h] = points at height h`.
    pub strata: Vec<Vec<P>>,
    /// Maximum height (diameter) of the sieve.
    pub diameter: u32,

    /// Deterministic global ordering of points (height-major, then point order).
    pub chart_points: Vec<P>, // index -> point
    /// Reverse lookup from point to chart index.
    pub chart_index: HashMap<P, usize>, // point -> index
}

impl<P: PointLike> StrataCache<P> {
    /// Create a new, empty `StrataCache`.
    pub fn new() -> Self {
        Self {
            height: HashMap::new(),
            depth: HashMap::new(),
            strata: Vec::new(),
            diameter: 0,
            chart_points: Vec::new(),
            chart_index: HashMap::new(),
        }
    }

    /// Index of `p` in the chart, if present.
    #[inline]
    pub fn index_of(&self, p: P) -> Option<usize> {
        self.chart_index.get(&p).copied()
    }

    /// Point at chart index `i`.
    #[inline]
    pub fn point_at(&self, i: usize) -> P {
        self.chart_points[i]
    }

    /// Total number of points in the chart.
    #[inline]
    pub fn len(&self) -> usize {
        self.chart_points.len()
    }
}

impl<P: PointLike> Default for StrataCache<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute strata information on-the-fly (no cache).
///
/// Returns a [`StrataCache`] containing height, depth, strata, and diameter information for all points.
///
/// ## Complexity
/// - Time: **O(|V| + |E|)** (Kahn topological sort + forward/backward passes)
/// - Space: **O(|V| + |E|)** for intermediate degree maps and result vectors.
///
/// # Errors
/// * [`MeshSieveError::MissingPointInCone`]: any arrow `(src → dst)` references a point not present
///   in `base_points ∪ cap_points`. The error aggregates all such points and is raised before any
///   topological pass.
/// * [`MeshSieveError::CycleDetected`]: the topology contains a cycle.
pub fn compute_strata<S>(s: &S) -> Result<StrataCache<S::Point>, MeshSieveError>
where
    S: Sieve,
    S::Point: PointLike,
{
    use std::collections::{HashMap, HashSet};

    // 0) Authoritative vertex set: V = base ∪ cap
    let mut in_deg: HashMap<S::Point, u32> = HashMap::new();
    for p in s.base_points() {
        in_deg.entry(p).or_insert(0);
    }
    for p in s.cap_points() {
        in_deg.entry(p).or_insert(0);
    }

    // 1) Validate edges against V and accumulate in-degrees
    //    We check both directions to catch backends that under-report one role.
    let mut missing: HashSet<S::Point> = HashSet::new();

    // 1a) cone: sources must be in V by construction; ensure all dst ∈ V
    let sources: Vec<_> = in_deg.keys().copied().collect();
    for p in sources {
        for (q, _) in s.cone(p) {
            if let Some(d) = in_deg.get_mut(&q) {
                *d += 1;
            } else {
                missing.insert(q);
            }
        }
    }

    // 1b) support: destinations must be in V; ensure all src ∈ V
    let caps: Vec<_> = in_deg.keys().copied().collect();
    for q in caps {
        for (src, _) in s.support(q) {
            if !in_deg.contains_key(&src) {
                missing.insert(src);
            }
        }
    }

    if !missing.is_empty() {
        let mut examples: Vec<_> = missing.iter().copied().collect();
        examples.sort_unstable();
        examples.truncate(8);
        return Err(MeshSieveError::MissingPointInCone(format!(
            "Topology references points not declared in base_points∪cap_points; examples: {examples:?} ({} missing total)",
            missing.len()
        )));
    }

    // 2) Kahn’s topological sort on V using the validated in-degrees
    let mut stack: Vec<_> = in_deg
        .iter()
        .filter_map(|(&p, &d)| (d == 0).then_some(p))
        .collect();

    let mut topo = Vec::with_capacity(in_deg.len());
    while let Some(p) = stack.pop() {
        topo.push(p);
        for (q, _) in s.cone(p) {
            // safe: all q ∈ V by validation above
            let d = in_deg.get_mut(&q).unwrap();
            *d -= 1;
            if *d == 0 {
                stack.push(q);
            }
        }
    }

    if topo.len() != in_deg.len() {
        return Err(MeshSieveError::CycleDetected);
    }

    // 3) Heights (same as before)
    let mut height = HashMap::new();
    for &p in &topo {
        let h = s
            .support(p)
            .map(|(pred, _)| height.get(&pred).copied().unwrap_or(0))
            .max()
            .map_or(0, |m| m + 1);
        height.insert(p, h);
    }
    let max_h = *height.values().max().unwrap_or(&0);
    let mut strata = vec![Vec::new(); (max_h + 1) as usize];
    for (&p, &h) in &height {
        strata[h as usize].push(p);
    }

    // 4) Depths (same as before)
    let mut depth = HashMap::new();
    for &p in topo.iter().rev() {
        let d = s
            .cone(p)
            .map(|(succ, _)| depth.get(&succ).copied().unwrap_or(0))
            .max()
            .map_or(0, |m| m + 1);
        depth.insert(p, d);
    }

    // 5) Deterministic chart (same as before)
    for lev in &mut strata {
        lev.sort_unstable();
    }
    let mut chart_points = Vec::with_capacity(height.len());
    for lev in &strata {
        chart_points.extend(lev.iter().copied());
    }
    let mut chart_index = HashMap::with_capacity(chart_points.len());
    for (i, p) in chart_points.iter().copied().enumerate() {
        chart_index.insert(p, i);
    }

    Ok(StrataCache {
        height,
        depth,
        strata,
        diameter: max_h,
        chart_points,
        chart_index,
    })
}
