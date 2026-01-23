use std::collections::HashMap;

use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinate_dm::CoordinateDM;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn build_coordinate_dm() -> CoordinateDM<f64, VecStorage<f64>> {
    let vertices = [pid(1), pid(2), pid(3), pid(4)];

    let mut atlas = Atlas::default();
    for vertex in vertices {
        atlas.try_insert(vertex, 2).unwrap();
    }

    let mut coords = Coordinates::try_new(2, atlas).unwrap();
    coords
        .try_restrict_mut(vertices[0])
        .unwrap()
        .copy_from_slice(&[0.0, 0.0]);
    coords
        .try_restrict_mut(vertices[1])
        .unwrap()
        .copy_from_slice(&[1.0, 0.0]);
    coords
        .try_restrict_mut(vertices[2])
        .unwrap()
        .copy_from_slice(&[1.0, 1.0]);
    coords
        .try_restrict_mut(vertices[3])
        .unwrap()
        .copy_from_slice(&[0.0, 1.0]);

    let mut labels = LabelSet::new();
    labels.set_label(vertices[0], "corner", 1);
    labels.set_label(vertices[3], "corner", 1);

    let mut dm = CoordinateDM::new(coords);
    dm.labels = Some(labels);
    dm
}

#[test]
fn coordinate_dm_reorder_points_preserves_data() {
    let coordinate_dm = build_coordinate_dm();
    let permutation = vec![pid(4), pid(2), pid(1), pid(3)];

    let reordered = coordinate_dm.reorder_points(&permutation).unwrap();

    let mut mapping = HashMap::new();
    for (idx, &old) in permutation.iter().enumerate() {
        mapping.insert(old, pid((idx + 1) as u64));
    }

    for &old in &[pid(1), pid(2), pid(3), pid(4)] {
        let new = mapping[&old];
        let old_slice = coordinate_dm.coordinates.try_restrict(old).unwrap();
        let new_slice = reordered.coordinates.try_restrict(new).unwrap();
        assert_eq!(old_slice, new_slice);
    }

    let old_labels = coordinate_dm.labels.as_ref().unwrap();
    let new_labels = reordered.labels.as_ref().unwrap();
    for (name, point, value) in old_labels.iter() {
        let new_point = mapping[&point];
        assert_eq!(new_labels.get_label(new_point, name), Some(value));
    }
}

#[test]
fn coordinate_dm_rename_points_preserves_order_and_data() {
    let coordinate_dm = build_coordinate_dm();
    let mut mapping = HashMap::new();
    mapping.insert(pid(1), pid(10));
    mapping.insert(pid(3), pid(11));

    let renamed = coordinate_dm.rename_points(&mapping).unwrap();

    let old_points = [pid(1), pid(2), pid(3), pid(4)];
    let new_points = [pid(10), pid(2), pid(11), pid(4)];

    for (&old, &new) in old_points.iter().zip(new_points.iter()) {
        let old_slice = coordinate_dm.coordinates.try_restrict(old).unwrap();
        let new_slice = renamed.coordinates.try_restrict(new).unwrap();
        assert_eq!(old_slice, new_slice);
    }

    let old_labels = coordinate_dm.labels.as_ref().unwrap();
    let new_labels = renamed.labels.as_ref().unwrap();
    for (name, point, value) in old_labels.iter() {
        let new_point = mapping.get(&point).copied().unwrap_or(point);
        assert_eq!(new_labels.get_label(new_point, name), Some(value));
    }

    assert_eq!(renamed.numbering.points(), new_points.as_slice());
}
