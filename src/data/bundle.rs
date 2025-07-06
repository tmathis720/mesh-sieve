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
    D: crate::overlap::delta::Delta<V, Part = V>,
{
    /// **Refine**: push data *down* the stack (base → cap).
    ///
    /// # Errors
    /// Returns an error if any point is missing in the underlying Section.
    pub fn refine(&mut self, bases: impl IntoIterator<Item = PointId>) -> Result<(), crate::mesh_error::MeshSieveError> {
        let mut actions = Vec::new();
        for b in self.stack.base().closure(bases) {
            let base_vals = self.section.try_restrict(b)?.to_vec();
            let caps: Vec<_> = self.stack.lift(b).collect();
            actions.push((base_vals, caps));
        }
        for (base_vals, caps) in actions {
            for (cap, orientation) in caps {
                let mut oriented_vals = base_vals.clone();
                if let crate::topology::arrow::Orientation::Reverse = orientation {
                    oriented_vals.reverse();
                }
                let cap_vals = self.section.try_restrict_mut(cap)?;
                let n = cap_vals.len().min(oriented_vals.len());
                cap_vals[..n].clone_from_slice(&oriented_vals[..n]);
            }
        }
        Ok(())
    }

    /// **Assemble**: pull data *up* the stack (cap → base) using `delta`.
    ///
    /// # Errors
    /// Returns an error if any point is missing in the underlying Section.
    pub fn assemble(&mut self, bases: impl IntoIterator<Item = PointId>) -> Result<(), crate::mesh_error::MeshSieveError> {
        let mut actions = Vec::new();
        for b in self.stack.base().closure(bases) {
            let caps: Vec<_> = self.stack.lift(b).map(|(cap, _)| cap).collect();
            let cap_vals: Vec<_> = caps
                .iter()
                .map(|&cap| self.section.try_restrict(cap).map(|sl| sl.to_vec()))
                .collect::<Result<_,_>>()?;
            actions.push((b, cap_vals));
        }
        for (b, cap_vals_vec) in actions {
            let base_vals = self.section.try_restrict_mut(b)?;
            for cap_vals in cap_vals_vec {
                if !cap_vals.is_empty() {
                    D::fuse(&mut base_vals[0], D::restrict(&cap_vals[0]));
                }
            }
        }
        Ok(())
    }

    /// Iterate over `(cap_point, &[V])` pairs for all DOFs attached to base `p`.
    ///
    /// # Errors
    /// Returns an error for any cap point missing in the underlying Section.
    pub fn dofs<'a>(
        &'a self,
        p: PointId
    ) -> impl Iterator<Item = Result<(PointId, &'a [V]), crate::mesh_error::MeshSieveError>> + 'a {
        self.stack.lift(p)
            .map(move |(cap, _)| self.section.try_restrict(cap).map(|sl| (cap, sl)))
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
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(2).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap(); // cap DOF for 1
        atlas.try_insert(PointId::new(102).unwrap(), 1).unwrap(); // cap DOF for 2
        let mut section = Section::<i32>::new(atlas.clone());
        section.try_set(PointId::new(1).unwrap(), &[10]).unwrap();
        section.try_set(PointId::new(2).unwrap(), &[20]).unwrap();
        let mut stack = InMemoryStack::<PointId, PointId, Orientation>::new();
        let _ = stack.add_arrow(PointId::new(1).unwrap(), PointId::new(101).unwrap(), Orientation::Forward);
        let _ = stack.add_arrow(PointId::new(2).unwrap(), PointId::new(102).unwrap(), Orientation::Forward);
        let mut bundle = Bundle { stack, section, delta: CopyDelta };
        // Refine: push base values to cap
        bundle.refine([PointId::new(1).unwrap(), PointId::new(2).unwrap()]).unwrap();
        assert_eq!(bundle.section.try_restrict(PointId::new(101).unwrap()).unwrap(), &[10]);
        assert_eq!(bundle.section.try_restrict(PointId::new(102).unwrap()).unwrap(), &[20]);
        // Assemble: pull cap values back to base
        bundle.section.try_set(PointId::new(101).unwrap(), &[30]).unwrap();
        bundle.section.try_set(PointId::new(102).unwrap(), &[40]).unwrap();
        bundle.assemble([PointId::new(1).unwrap(), PointId::new(2).unwrap()]).unwrap();
        assert_eq!(bundle.section.try_restrict(PointId::new(1).unwrap()).unwrap(), &[30]);
        assert_eq!(bundle.section.try_restrict(PointId::new(2).unwrap()).unwrap(), &[40]);
    }
    #[test]
    fn empty_bundle_noop() {
        let atlas = Atlas::default();
        let section = Section::<i32>::new(atlas.clone());
        let stack = InMemoryStack::<PointId,PointId,Orientation>::new();
        let mut bundle = Bundle { stack, section, delta: CopyDelta };
        // Should not panic, nothing to do
        bundle.refine(std::iter::empty::<PointId>()).unwrap();
        bundle.assemble(std::iter::empty::<PointId>()).unwrap();
    }

    #[test]
    fn multiple_dofs_only_first_moved() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 2).unwrap();
        let mut section = Section::<i32>::new(atlas.clone());
        section.try_set(PointId::new(1).unwrap(), &[10,20]).unwrap();
        let mut stack = InMemoryStack::<PointId,PointId,Orientation>::new();
        let _ = stack.add_arrow(PointId::new(1).unwrap(), PointId::new(101).unwrap(), Orientation::Forward);
        let mut bundle = Bundle { stack, section, delta: CopyDelta };
        bundle.refine([PointId::new(1).unwrap()]).unwrap();
        let vals = bundle.section.try_restrict(PointId::new(101).unwrap()).unwrap();
        // Both slots should be copied
        assert_eq!(vals, &[10, 20]);
    }

    #[test]
    fn reverse_orientation_refine() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 2).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 2).unwrap();
        let mut section = Section::<i32>::new(atlas.clone());
        section.try_set(PointId::new(1).unwrap(), &[1,2]).unwrap();
        let mut stack = InMemoryStack::<PointId,PointId,Orientation>::new();
        let _ = stack.add_arrow(PointId::new(1).unwrap(), PointId::new(101).unwrap(), Orientation::Reverse);
        let mut bundle = Bundle { stack, section, delta: CopyDelta };
        bundle.refine([PointId::new(1).unwrap()]).unwrap();
        // Should get reversed [2,1]
        assert_eq!(bundle.section.try_restrict(PointId::new(101).unwrap()).unwrap(), &[2,1]);
    }

    #[test]
    fn assemble_with_add_delta() {
        use crate::overlap::delta::AddDelta;
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(102).unwrap(), 1).unwrap();
        let mut section = Section::<i32>::new(atlas.clone());
        section.try_set(PointId::new(101).unwrap(), &[5]).unwrap();
        section.try_set(PointId::new(102).unwrap(), &[7]).unwrap();
        let mut stack = InMemoryStack::<PointId,PointId,Orientation>::new();
        let _ = stack.add_arrow(PointId::new(1).unwrap(), PointId::new(101).unwrap(), Orientation::Forward);
        let _ = stack.add_arrow(PointId::new(1).unwrap(), PointId::new(102).unwrap(), Orientation::Forward);
        let mut bundle = Bundle { stack, section, delta: AddDelta };
        bundle.assemble([PointId::new(1).unwrap()]).unwrap();
        // base receives sum 5+7
        assert_eq!(bundle.section.try_restrict(PointId::new(1).unwrap()).unwrap(), &[12]);
    }

    #[test]
    fn dofs_iterator() {
        let mut atlas = Atlas::default();
        atlas.try_insert(PointId::new(1).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(101).unwrap(), 1).unwrap();
        atlas.try_insert(PointId::new(102).unwrap(), 1).unwrap();
        let mut section = Section::<i32>::new(atlas.clone());
        section.try_set(PointId::new(101).unwrap(), &[8]).unwrap();
        section.try_set(PointId::new(102).unwrap(), &[9]).unwrap();
        let mut stack = InMemoryStack::<PointId,PointId,Orientation>::new();
        let _ = stack.add_arrow(PointId::new(1).unwrap(), PointId::new(101).unwrap(), Orientation::Forward);
        let _ = stack.add_arrow(PointId::new(1).unwrap(), PointId::new(102).unwrap(), Orientation::Forward);
        let bundle = Bundle { stack, section, delta: CopyDelta };
        let vec: Vec<_> = bundle.dofs(PointId::new(1).unwrap()).collect();
        let mut vec = vec.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
        vec.sort_by_key(|(cap,_)| cap.get());
        assert_eq!(vec, vec![
            (PointId::new(101).unwrap(), &[8][..]),
            (PointId::new(102).unwrap(), &[9][..]),
        ]);
    }

    #[test]
    fn refine_unknown_base_errors() {
        let atlas = Atlas::default();
        let section = Section::<i32>::new(atlas.clone());
        let stack = InMemoryStack::<PointId,PointId,Orientation>::new();
        let mut bundle = Bundle { stack, section, delta: CopyDelta };
        let err = bundle.refine([PointId::new(999).unwrap()]).unwrap_err();
        assert!(matches!(err, crate::mesh_error::MeshSieveError::PointNotInAtlas(pid) if pid.get() == 999));
    }
}
