//! Bundle: Combines mesh topology, DOF storage, and data transfer rules.
//!
//! A `Bundle` ties together:
//! 1. A **vertical stack** of mesh points → DOF points (`stack`),  
//! 2. A **field section** storing per-point data (`section`),  
//! 3. A **delta** strategy (`delta`) for refining/assembling data.
//!
//! This abstraction supports push (refine) and pull (assemble) of data
//! across mesh hierarchy levels, as described in Knepley & Karpeev (2009).

use crate::topology::point::PointId;
use crate::topology::stack::{InMemoryStack, Stack};
use crate::data::section::Section;
use crate::overlap::delta::CopyDelta;
use crate::topology::sieve::Sieve;

/// `Bundle<V, D>` packages a mesh‐to‐DOF stack, a data section, and a `Delta`-type.
///
/// - `V`: underlying data type stored at each DOF (e.g., `f64`, `i32`, …).
/// - `D`: `Delta<V>` implementation guiding how data moves (defaults to `CopyDelta`).
///
/// # Fields
/// - `stack`: vertical arrows from base mesh points → cap (DOF) points,
///    carrying an `Orientation` payload if needed.
/// - `section`: contiguous storage of data `V` for each point in the atlas.
/// - `delta`: rules for extracting (`restrict`) and merging (`fuse`) values.
pub struct Bundle<V, D = CopyDelta> {
    /// Vertical connectivity: base points → cap (DOF) points.
    pub stack: InMemoryStack<PointId, PointId, crate::topology::arrow::Orientation>,
    /// Field data storage, indexed by `PointId`.
    pub section: Section<V>,
    /// Delta strategy for refine/assemble operations.
    pub delta: D,
}

impl<V, D> Bundle<V, D>
where
    V: Clone + Default,
    // `D` must implement the Delta trait for `V`, with `Part = V`.
    D: crate::overlap::delta::Delta<V, Part = V>,
{
    /// **Refine**: push data *down* the stack (base → cap).
    ///
    /// For each base point in the transitive closure of `bases`:
    /// 1. Read its current value slice (`restrict`) → `base_vals`.
    /// 2. Lift to each cap point (DOF) via `stack.lift`.
    /// 3. Call `D::restrict` to extract the part to send.
    /// 4. Overwrite the cap’s slice with that part.
    ///
    /// # Example
    /// ```ignore
    /// // Propagate coarse solution values to refined DOF points.
    /// bundle.refine(mesh_cells_iter);
    /// ```
    pub fn refine(&mut self, bases: impl IntoIterator<Item = PointId>) {
        // Collect actions first to avoid mutable aliasing on `section`.
        let mut actions = Vec::new();
        for b in self.stack.base().closure(bases) {
            // Clone the base’s slice of values.
            let base_vals = self.section.restrict(b).to_vec();
            // Collect all cap points lifted from base `b`.
            let caps: Vec<_> = self.stack.lift(b).collect();
            actions.push((base_vals, caps));
        }

        // Execute actions: for each cap, overwrite its slice.
        for (base_vals, caps) in actions {
            for (cap, _payload) in caps {
                // Extract the part to send along the arrow.
                let part = D::restrict(&base_vals[0]);
                let cap_vals = self.section.restrict_mut(cap);
                if !cap_vals.is_empty() {
                    cap_vals[0] = part;
                }
            }
        }
    }

    /// **Assemble**: pull data *up* the stack (cap → base) using `delta`.
    ///
    /// For each base point in the closure of `bases`:
    /// 1. Gather all cap points via `stack.lift`.
    /// 2. Read each cap’s slice (`restrict`) → `cap_vals`.
    /// 3. Accumulate back into the base slice:
    ///    `D::fuse(&mut base_vals[0], incoming_part)`.
    ///
    /// # Example
    /// ```ignore
    /// // Gather refined DOF contributions back to coarse mesh.
    /// bundle.assemble(mesh_cells_iter);
    /// ```
    pub fn assemble(&mut self, bases: impl IntoIterator<Item = PointId>) {
        // Collect actions first to avoid borrow conflicts on `section`.
        let mut actions = Vec::new();
        for b in self.stack.base().closure(bases) {
            // Identify all caps attached to base `b`.
            let caps: Vec<_> = self.stack.lift(b).map(|(cap, _)| cap).collect();
            // Clone each cap’s slice of values.
            let cap_vals: Vec<_> = caps
                .iter()
                .map(|&cap| self.section.restrict(cap).to_vec())
                .collect();
            actions.push((b, cap_vals));
        }

        // Execute fuse operations: accumulate into each base.
        for (b, cap_vals_vec) in actions {
            let base_vals = self.section.restrict_mut(b);
            for cap_vals in cap_vals_vec.iter() {
                if !cap_vals.is_empty() {
                    // Merge with the delta strategy.
                    D::fuse(&mut base_vals[0], D::restrict(&cap_vals[0]));
                }
            }
        }
    }

    /// Iterate over `(cap_point, &[V])` pairs for all DOFs attached to base `p`.
    ///
    /// Yields each cap point and an immutable view into its data slice.
    ///
    /// # Example
    /// ```ignore
    /// for (dof_pt, values) in bundle.dofs(cell_pt) {
    ///     // use values[..] for computation…
    /// }
    /// ```
    pub fn dofs<'a>(
        &'a self,
        p: PointId
    ) -> impl Iterator<Item = (PointId, &'a [V])> + 'a {
        self.stack.lift(p)
            // Map each cap point to its data slice.
            .map(move |(cap, _)| (cap, self.section.restrict(cap)))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::arrow::Orientation;
    use crate::overlap::delta::CopyDelta;
    use crate::data::atlas::Atlas;
    #[test]
    fn bundle_basic_refine_and_assemble() {
        let mut atlas = Atlas::default();
        atlas.insert(PointId::new(1), 1);
        atlas.insert(PointId::new(2), 1);
        atlas.insert(PointId::new(101), 1); // cap DOF for 1
        atlas.insert(PointId::new(102), 1); // cap DOF for 2
        let mut section = Section::<i32>::new(atlas.clone());
        section.set(PointId::new(1), &[10]);
        section.set(PointId::new(2), &[20]);
        let mut stack = InMemoryStack::<PointId, PointId, Orientation>::new();
        stack.add_arrow(PointId::new(1), PointId::new(101), Orientation::Forward);
        stack.add_arrow(PointId::new(2), PointId::new(102), Orientation::Forward);
        let mut bundle = Bundle { stack, section, delta: CopyDelta };
        // Refine: push base values to cap
        bundle.refine([PointId::new(1), PointId::new(2)]);
        assert_eq!(bundle.section.restrict(PointId::new(101)), &[10]);
        assert_eq!(bundle.section.restrict(PointId::new(102)), &[20]);
        // Assemble: pull cap values back to base
        bundle.section.set(PointId::new(101), &[30]);
        bundle.section.set(PointId::new(102), &[40]);
        bundle.assemble([PointId::new(1), PointId::new(2)]);
        assert_eq!(bundle.section.restrict(PointId::new(1)), &[30]);
        assert_eq!(bundle.section.restrict(PointId::new(2)), &[40]);
    }
}
