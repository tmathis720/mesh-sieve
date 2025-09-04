use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::topology::point::PointId;
use rand::seq::SliceRandom;
use rand::Rng;

mod util;
use util::rng;

#[test]
fn atlas_random_inserts_then_removals_invariants_hold() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let mut rng = rng();

    let n = 500;
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let len = 1 + (rng.next_u32() as usize % 8);
        let pid = PointId::new((i as u64) + 1)?;
        let off = a.try_insert(pid, len)?;
        ids.push((pid, len, off));
    }

    let spans = a.atlas_map();
    let sum: usize = spans.iter().map(|&(_, l)| l).sum();
    assert_eq!(a.total_len(), sum);
    let mut expected = 0usize;
    for &(off, len) in &spans {
        assert_eq!(off, expected, "non-contiguous at off={off}, expected={expected}");
        expected += len;
    }

    let spans2: Vec<_> = a.iter_spans().collect();
    assert_eq!(spans, spans2);

    let mut to_remove: Vec<_> = ids.iter().map(|&(p, _, _)| p).collect();
    to_remove.shuffle(&mut rng);
    to_remove.truncate(n / 3);

    for p in to_remove {
        a.remove_point(p)?;
        let mut expected = 0usize;
        for &(off, len) in a.iter_spans() {
            assert_eq!(off, expected, "after removal: contiguity broken");
            expected += len;
        }
        assert_eq!(a.total_len(), expected);
    }

    Ok(())
}

#[test]
fn atlas_invariants_property_small() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = Atlas::default();
    let mut rng = rng();

    for step in 0..50 {
        if rng.next_u32() % 2 == 0 || a.total_len() == 0 {
            let pid = PointId::new((step + 1) as u64)?;
            let len = 1 + (rng.next_u32() as usize % 5);
            let _ = a.try_insert(pid, len)?;
        } else {
            if let Some(p) = a.points().next() {
                let _ = a.remove_point(p)?;
            }
        }
        let mut expected = 0usize;
        for &(off, len) in a.iter_spans() {
            assert_eq!(off, expected);
            expected += len;
        }
        assert_eq!(a.total_len(), expected);
    }
    Ok(())
}
