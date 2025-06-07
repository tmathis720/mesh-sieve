use crate::topology::point::PointId;
use crate::topology::stack::{InMemoryStack, Stack};
use crate::data::section::Section;
use crate::overlap::delta::CopyDelta;
use crate::topology::sieve::Sieve;

/// A bundle links mesh topology to DOF set and packages field storage.
pub struct Bundle<V, D = CopyDelta> {
    pub stack: InMemoryStack<PointId, PointId, crate::topology::arrow::Orientation>,
    pub section: Section<V>,
    pub delta: D,
}

impl<V, D> Bundle<V, D>
where
    V: Clone + Default,
    D: crate::overlap::delta::Delta<V, Part = V>,
{
    /// Push data *down* the stack (base → cap)
    pub fn refine(&mut self, bases: impl IntoIterator<Item = PointId>) {
        // Collect all (b, base_val, lifted caps) first to avoid borrow conflicts
        let mut actions = Vec::new();
        for b in self.stack.base().closure(bases) {
            let base_vals = self.section.restrict(b).to_vec();
            let caps: Vec<_> = self.stack.lift(b).collect();
            actions.push((base_vals, caps));
        }
        for (base_vals, caps) in actions {
            for (cap, _payload) in caps {
                let part = D::restrict(&base_vals[0]);
                let cap_vals = self.section.restrict_mut(cap);
                if !cap_vals.is_empty() {
                    cap_vals[0] = part;
                }
            }
        }
    }
    /// Pull data *up* the stack (cap → base) using delta
    pub fn assemble(&mut self, bases: impl IntoIterator<Item = PointId>) {
        // Collect all (b, cap_vals) first to avoid borrow conflicts
        let mut actions = Vec::new();
        for b in self.stack.base().closure(bases) {
            let caps: Vec<_> = self.stack.lift(b).map(|(cap, _)| cap).collect();
            let cap_vals: Vec<_> = caps.iter().map(|&cap| self.section.restrict(cap).to_vec()).collect();
            actions.push((b, cap_vals));
        }
        for (b, cap_vals_vec) in actions {
            let base_vals = self.section.restrict_mut(b);
            for cap_vals in cap_vals_vec.iter() {
                if !cap_vals.is_empty() {
                    D::fuse(&mut base_vals[0], D::restrict(&cap_vals[0]));
                }
            }
        }
    }
    /// Iterate DOFs attached to a base point
    pub fn dofs<'a>(&'a self, p: PointId) -> impl Iterator<Item = (PointId, &'a [V])> + 'a {
        self.stack.lift(p).map(move |(cap, _)| (cap, self.section.restrict(cap)))
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
