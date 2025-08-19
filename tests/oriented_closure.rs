use mesh_sieve::topology::sieve::InMemoryOrientedSieve;
use mesh_sieve::topology::sieve::oriented::OrientedSieve;
use mesh_sieve::section::{Section, section_on_closure_o};

#[derive(Default)]
struct ToySection;
impl Section for ToySection {
    type Point = u32;
    fn dof(&self, _p: u32) -> usize { 1 }
    fn offset(&self, p: u32) -> usize { p as usize }
}

#[test]
fn oriented_closure_accumulates() {
    // Build: cell 0 -> edges 10,11 with orientations +1, -1; edge 10 -> vertex 20 with +1
    // Accumulated orientation to 20 should be (+1 compose +1) = +2
    let mut s = InMemoryOrientedSieve::<u32, (), i32>::new();
    s.add_arrow_o(0, 10, (),  1);
    s.add_arrow_o(0, 11, (), -1);
    s.add_arrow_o(10, 20, (), 1);

    let cl = s.closure_o([0]);
    // Points appear sorted: 0,10,11,20 with accumulated orientations from seed 0
    assert_eq!(cl, vec![(0,0), (10,1), (11,-1), (20,2)]);

    // Combine with Section helper
    let sec = ToySection::default();
    let spans: Vec<_> = section_on_closure_o(&s, &sec, 0).collect();
    assert_eq!(spans, vec![(0,1,0), (10,1,1), (11,1,-1), (20,1,2)]);
}

#[test]
fn star_orientation_inverts_step() {
    // Reverse traversal should invert the step orientation
    let mut s = InMemoryOrientedSieve::<u32, (), i32>::new();
    s.add_arrow_o(7, 3, (),  2); // ori(7->3)=+2
    s.add_arrow_o(3, 1, (), -1); // ori(3->1)=-1

    // Star from leaf 1 should go up and compose inverses:
    // inv(-1)=+1, then inv(+2)=-2 => total -1 at point 7
    let st = s.star_o([1]);
    // sorted by point: (1,0), (3,1), (7,-1)
    assert_eq!(st, vec![(1,0), (3,1), (7,-1)]);
}
