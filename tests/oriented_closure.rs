use mesh_sieve::section::{Section, section_on_closure_o};
use mesh_sieve::topology::orientation::Sign;
use mesh_sieve::topology::sieve::InMemoryOrientedSieve;
use mesh_sieve::topology::sieve::oriented::OrientedSieve;

#[derive(Default)]
struct ToySection;
impl Section for ToySection {
    type Point = u32;
    fn dof(&self, _p: u32) -> usize {
        1
    }
    fn offset(&self, p: u32) -> usize {
        p as usize
    }
}

#[test]
fn oriented_closure_accumulates() {
    // Build: cell 0 -> edges 10,11 with orientations flip=false, flip=true;
    // edge 10 -> vertex 20 with flip=true. Accumulated orientation to 20
    // should be false ^ true = true.
    let mut s = InMemoryOrientedSieve::<u32, (), Sign>::new();
    s.add_arrow_o(0, 10, (), Sign(false));
    s.add_arrow_o(0, 11, (), Sign(true));
    s.add_arrow_o(10, 20, (), Sign(true));

    let cl = s.closure_o([0]);
    // Points appear sorted: 0,10,11,20 with accumulated orientations from seed 0
    assert_eq!(
        cl,
        vec![
            (0, Sign(false)),
            (10, Sign(false)),
            (11, Sign(true)),
            (20, Sign(true)),
        ]
    );

    // Combine with Section helper
    let sec = ToySection::default();
    let spans: Vec<_> = section_on_closure_o(&s, &sec, 0).collect();
    assert_eq!(
        spans,
        vec![
            (0, 1, Sign(false)),
            (10, 1, Sign(false)),
            (11, 1, Sign(true)),
            (20, 1, Sign(true)),
        ]
    );
}

#[test]
fn star_orientation_inverts_step() {
    // Reverse traversal should compose with inverse at each step.
    let mut s = InMemoryOrientedSieve::<u32, (), Sign>::new();
    s.add_arrow_o(7, 3, (), Sign(true)); // ori(7->3)=flip
    s.add_arrow_o(3, 1, (), Sign(true)); // ori(3->1)=flip

    // Star from leaf 1 should go up and compose inverses:
    // inv(flip)=flip, so orientation at 3 is flip; then another flip gives
    // identity at point 7.
    let st = s.star_o([1]);
    // sorted by point: (1,false), (3,true), (7,false)
    assert_eq!(
        st,
        vec![(1, Sign(false)), (3, Sign(true)), (7, Sign(false))]
    );
}
