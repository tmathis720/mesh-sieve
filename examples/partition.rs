//! Run with: cargo run --example partition --features metis-support
use mesh_sieve::algs::dual_graph::{DualGraph, build_dual};
#[cfg(feature = "metis-support")]
use mesh_sieve::algs::metis_partition::MetisPartition;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn main() {
    // 1) Build a tiny mesh & extract its cells
    let (mesh, cells) = {
        let mut s = InMemorySieve::<PointId, ()>::default();
        // Build a small test mesh
        s.add_arrow(PointId::new(10).unwrap(), PointId::new(1).unwrap(), ());
        s.add_arrow(PointId::new(10).unwrap(), PointId::new(2).unwrap(), ());
        s.add_arrow(PointId::new(11).unwrap(), PointId::new(2).unwrap(), ());
        s.add_arrow(PointId::new(11).unwrap(), PointId::new(3).unwrap(), ());
        (
            s,
            vec![PointId::new(10).unwrap(), PointId::new(11).unwrap()],
        )
    };

    // 2) Build dual graph
    let _dg: DualGraph = build_dual(&mesh, cells.clone());

    #[cfg(feature = "metis-support")]
    {
        // 3) Partition into 2 parts
        let p = _dg.metis_partition(2);
        println!("partition array = {:?}", p.part);
    }
}
