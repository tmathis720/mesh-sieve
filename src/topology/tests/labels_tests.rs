use crate::topology::labels::LabelSet;
use crate::topology::point::PointId;

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
