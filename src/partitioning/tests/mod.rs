use super::*;
#[path = "partition_property_tests.rs"]
mod partition_property_tests;

#[derive(Debug)]
struct DummyGraph;
impl PartitionableGraph for DummyGraph {
    type VertexId = usize;
    type VertexParIter<'a> = rayon::vec::IntoIter<usize>;
    type NeighParIter<'a> = rayon::vec::IntoIter<usize>;
    type NeighIter<'a> = std::vec::IntoIter<usize>;

    fn vertices(&self) -> Self::VertexParIter<'_> {
        vec![0, 1, 2, 3].into_par_iter()
    }
    fn neighbors(&self, _v: Self::VertexId) -> Self::NeighParIter<'_> {
        Vec::new().into_par_iter()
    }
    fn neighbors_seq(&self, _v: Self::VertexId) -> Self::NeighIter<'_> {
        Vec::new().into_iter()
    }
    fn degree(&self, _v: Self::VertexId) -> usize {
        0
    }
    fn edges(&self) -> rayon::vec::IntoIter<(usize, usize)> {
        Vec::new().into_par_iter()
    }
}

#[test]
fn trivial_partition_compiles() {
    let g = DummyGraph;
    let cfg = PartitionerConfig {
        n_parts: 2,
        ..Default::default()
    };
    match partition(&g, &cfg) {
        Ok(pm) => assert_eq!(pm.len(), 4),
        Err(PartitionerError::NoPositiveMerge) => {}
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}
