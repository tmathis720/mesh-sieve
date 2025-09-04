use crate::topology::sieve::{InMemoryOrientedSieve, InMemorySieve, Sieve};

#[test]
fn remove_point_scrubs_both_sides_and_preserves_presence() {
    let mut s = InMemorySieve::<u32, ()>::default();

    s.add_point(1);
    s.add_point(2);
    s.add_point(3);

    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());

    assert_eq!(s.cone(1).count(), 1);
    assert_eq!(s.support(3).count(), 1);

    s.remove_point(2);

    assert!(s.cone(1).next().is_none());
    assert!(s.support(3).next().is_none());

    assert!(s.adjacency_out.contains_key(&2));
    assert!(s.adjacency_in.contains_key(&2));
    assert!(s.adjacency_out.contains_key(&1));
    assert!(s.adjacency_in.contains_key(&3));
}

#[test]
fn remove_base_point_removes_outgoing_and_base_key() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(1, 3, ());
    s.add_arrow(4, 1, ());

    s.remove_base_point(1);

    assert!(!s.adjacency_out.contains_key(&1));
    for ins in s.adjacency_in.values() {
        assert!(ins.iter().all(|(src, _)| *src != 1));
    }
    assert!(s.adjacency_in.get(&1).is_some());
    assert_eq!(s.support(1).count(), 1);
}

#[test]
fn remove_cap_point_removes_incoming_and_cap_key() {
    let mut s = InMemorySieve::<u32, ()>::default();
    s.add_arrow(1, 2, ());
    s.add_arrow(3, 2, ());
    s.add_arrow(2, 5, ());

    s.remove_cap_point(2);

    assert!(!s.adjacency_in.contains_key(&2));
    for outs in s.adjacency_out.values() {
        assert!(outs.iter().all(|(dst, _)| *dst != 2));
    }
    assert!(s.adjacency_out.get(&2).is_some());
    assert_eq!(s.cone(2).count(), 1);
}

#[test]
fn oriented_remove_point_scrubs_both_sides_and_preserves_presence() {
    let mut s = InMemoryOrientedSieve::<u32, ()>::default();

    s.add_point(1);
    s.add_point(2);
    s.add_point(3);

    s.add_arrow(1, 2, ());
    s.add_arrow(2, 3, ());

    assert_eq!(s.cone(1).count(), 1);
    assert_eq!(s.support(3).count(), 1);

    s.remove_point(2);

    assert!(s.cone(1).next().is_none());
    assert!(s.support(3).next().is_none());

    assert!(s.adjacency_out.contains_key(&2));
    assert!(s.adjacency_in.contains_key(&2));
    assert!(s.adjacency_out.contains_key(&1));
    assert!(s.adjacency_in.contains_key(&3));
}
