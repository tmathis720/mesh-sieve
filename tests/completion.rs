use mesh_sieve::algs::{communicator::RayonComm, completion::complete_section};
use mesh_sieve::data::{atlas::Atlas, section::Section};
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::point::PointId;

#[test]
fn ghost_update_self() {
    // This test simulates a “rank” that owns P0 and also ghosts P0:
    // i.e. it adds a self‐link so that complete_section does one send
    // and one receive, but both target the same rank (yourself).

    let mut ovlp = Overlap::default();
    let p0 = PointId::new(1);

    // Add a self‐overlap: owner and ghost are the same rank (0).
    ovlp.add_link(p0, /*remote_rank=*/ 0, p0);

    // Build a section with one value = 42
    let mut atlas = Atlas::default();
    atlas.insert(p0, 1);
    let mut sec = Section::<u32>::new(atlas);
    sec.set(p0, &[42]);

    let comm = RayonComm::new(0);
    let delta = CopyDelta;

    // Should complete without deadlock and leave the value intact.
    complete_section(&mut sec, &ovlp, &comm, &delta, 0, 1);

    assert_eq!(sec.restrict(p0)[0], 42);
}
