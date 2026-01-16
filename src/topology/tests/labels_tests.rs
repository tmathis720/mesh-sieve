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
