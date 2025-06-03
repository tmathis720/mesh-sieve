//! Run with: cargo run --example partition --features metis
use sieve_rs::algs::dual_graph::{build_dual, DualGraph};
#[cfg(feature = "metis-support")]
use sieve_rs::algs::metis_partition::MetisPartition;
use sieve_rs::topology::sieve::{InMemorySieve, Sieve};
use sieve_rs::topology::point::PointId;

fn main() {
    // 1) Build a tiny mesh & extract its cells
    let (mesh, cells) = {
      let mut s = InMemorySieve::<PointId,()>::default();
      // Build a small test mesh
      s.add_arrow(PointId::new(10), PointId::new(1), ());
      s.add_arrow(PointId::new(10), PointId::new(2), ());
      s.add_arrow(PointId::new(11), PointId::new(2), ());
      s.add_arrow(PointId::new(11), PointId::new(3), ());
      (s, vec![PointId::new(10), PointId::new(11)])
    };

    // 2) Build dual graph
    let dg: DualGraph = build_dual(&mesh, cells.clone());

    #[cfg(feature = "metis-support")] {
      // 3) Partition into 2 parts
      let p = dg.metis_partition(2);
      println!("partition array = {:?}", p.part);
    }
}
