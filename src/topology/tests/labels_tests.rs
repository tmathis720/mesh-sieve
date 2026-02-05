use crate::topology::labels::{LabelSet, propagate_label_set_closure, propagate_label_set_star};
use crate::topology::point::PointId;
use crate::topology::sieve::InMemorySieve;
use crate::topology::sieve::sieve_trait::Sieve;

#[test]
fn label_set_roundtrip() {
    let mut labels = LabelSet::new();
    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();

    assert_eq!(labels.set_label(p1, "boundary", 7), None);
    assert_eq!(labels.set_label(p2, "boundary", 7), None);
    assert_eq!(labels.set_label(p1, "material", 3), None);

    assert_eq!(labels.get_label(p1, "boundary"), Some(7));
    assert_eq!(labels.get_label(p2, "boundary"), Some(7));
    assert_eq!(labels.get_label(p1, "material"), Some(3));
    assert_eq!(labels.get_label(p2, "material"), None);

    let mut pts: Vec<_> = labels.points_with_label("boundary", 7).collect();
    pts.sort_unstable();
    assert_eq!(pts, vec![p1, p2]);
}

#[test]
fn label_set_strata_and_values() {
    let mut labels = LabelSet::new();
    let p1 = PointId::new(10).unwrap();
    let p2 = PointId::new(4).unwrap();
    let p3 = PointId::new(7).unwrap();
    let p4 = PointId::new(2).unwrap();

    labels.set_label(p1, "phase", 2);
    labels.set_label(p2, "phase", 1);
    labels.set_label(p3, "phase", 2);
    labels.set_label(p4, "phase", 1);

    assert_eq!(labels.stratum_size("phase", 2), 2);
    assert_eq!(labels.stratum_points("phase", 2), vec![p3, p1]);
    assert_eq!(labels.stratum_values("phase"), vec![1, 2]);
    assert_eq!(labels.values("phase"), vec![1, 2]);
}

#[test]
fn label_set_clear_label_value() {
    let mut labels = LabelSet::new();
    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    let p3 = PointId::new(3).unwrap();

    labels.set_label(p1, "boundary", 5);
    labels.set_label(p2, "boundary", 9);
    labels.set_label(p3, "boundary", 5);
    labels.set_label(p1, "material", 1);

    assert_eq!(labels.clear_label_value("boundary", 5), 2);
    assert_eq!(labels.get_label(p1, "boundary"), None);
    assert_eq!(labels.get_label(p3, "boundary"), None);
    assert_eq!(labels.get_label(p2, "boundary"), Some(9));
    assert_eq!(labels.values("boundary"), vec![9]);

    assert_eq!(labels.clear_label_value("boundary", 9), 1);
    assert_eq!(labels.values("boundary"), Vec::<i32>::new());
    assert_eq!(labels.get_label(p1, "material"), Some(1));
}

#[test]
fn label_set_range_queries_and_set_ops() {
    let mut left = LabelSet::new();
    let mut right = LabelSet::new();
    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    let p3 = PointId::new(3).unwrap();
    let p4 = PointId::new(4).unwrap();

    left.set_label(p1, "region", 1);
    left.set_label(p2, "region", 2);
    left.set_label(p3, "region", 3);
    right.set_label(p2, "region", 2);
    right.set_label(p4, "region", 2);

    assert_eq!(left.stratum_points_in_range("region", 2..=3), vec![p2, p3]);
    assert_eq!(left.stratum_union(&right, "region", 2), vec![p2, p4]);
    assert_eq!(left.stratum_intersection(&right, "region", 2), vec![p2]);
    assert_eq!(
        left.stratum_difference(&right, "region", 2),
        Vec::<PointId>::new()
    );
}

#[test]
fn label_propagation_closure_and_star() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = PointId::new(10).unwrap();
    let face = PointId::new(20).unwrap();
    let vertex = PointId::new(30).unwrap();

    sieve.add_arrow(cell, face, ());
    sieve.add_arrow(face, vertex, ());

    let mut labels = LabelSet::new();
    labels.set_label(face, "boundary", 1);
    labels.set_label(vertex, "seed", 9);

    let closure_labels = propagate_label_set_closure(&sieve, &labels);
    let mut boundary_pts = closure_labels.stratum_points("boundary", 1);
    boundary_pts.sort_unstable();
    assert_eq!(boundary_pts, vec![face, vertex]);

    let star_labels = propagate_label_set_star(&sieve, &labels);
    let mut seed_pts = star_labels.stratum_points("seed", 9);
    seed_pts.sort_unstable();
    assert_eq!(seed_pts, vec![cell, face, vertex]);
}
