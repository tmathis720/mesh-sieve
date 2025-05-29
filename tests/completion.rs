use sieve_rs::algs::{
    communicator::RayonComm,
    completion::complete_section,
};
use sieve_rs::data::{atlas::Atlas, section::Section};
use sieve_rs::topology::point::PointId;
use sieve_rs::overlap::overlap::Overlap;
use sieve_rs::overlap::delta::CopyDelta;

#[test]
fn ghost_update_two_ranks() {
    // rank 0 owns P0; rank 1 needs a ghost copy
    let mut ovlp = Overlap::default();
    let p0 = PointId::new(1);
    ovlp.add_link(p0, 1, p0);   // local p0 -> remote rank1, same id

    // ------- set up Section on rank 0 -------
    let mut atlas0 = Atlas::default();
    atlas0.insert(p0, 1);
    let mut sec0 = Section::<u32>::new(atlas0);
    sec0.set(p0, &[99]);

    // ------- set up Section on rank 1 (ghost) -------
    let mut atlas1 = Atlas::default();
    atlas1.insert(p0, 1);   // ghost slot
    let mut sec1 = Section::<u32>::new(atlas1);

    let comm0 = RayonComm::new(0);
    let comm1 = RayonComm::new(1);

    // rank 1 posts receive first
    let delta = CopyDelta;
    complete_section(&mut sec1, &ovlp, &comm1, &delta, 1);
    complete_section(&mut sec0, &ovlp, &comm0, &delta, 0);

    assert_eq!(sec1.restrict(p0)[0], 99);
}
