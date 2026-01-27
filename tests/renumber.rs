use std::collections::{BTreeMap, BTreeSet, HashMap};

use mesh_sieve::algs::renumber::{
    StratifiedOrdering, renumber_coordinate_dm, renumber_points, stratified_permutation,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinate_dm::CoordinateDM;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, MutableSieve, Sieve};

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn build_mesh(
) -> MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>> {
    let vertices = [pid(1), pid(2), pid(3), pid(4)];
    let cells = [pid(10), pid(11)];

    let mut sieve = InMemorySieve::default();
    for p in vertices.iter().chain(cells.iter()) {
        MutableSieve::add_point(&mut sieve, *p);
    }
    sieve.add_arrow(cells[0], vertices[0], ());
    sieve.add_arrow(cells[0], vertices[1], ());
    sieve.add_arrow(cells[1], vertices[1], ());
    sieve.add_arrow(cells[1], vertices[2], ());

    let mut atlas = Atlas::default();
    atlas.try_insert(vertices[0], 2).unwrap();
    atlas.try_insert(vertices[2], 1).unwrap();
    atlas.try_insert(cells[0], 3).unwrap();
    let mut section = Section::<f64, VecStorage<f64>>::new(atlas);
    section.try_set(vertices[0], &[1.0, 2.0]).unwrap();
    section.try_set(vertices[2], &[3.5]).unwrap();
    section.try_set(cells[0], &[4.0, 5.0, 6.0]).unwrap();

    let mut sections = BTreeMap::new();
    sections.insert("field".to_string(), section);

    let mut labels = LabelSet::new();
    labels.set_label(vertices[0], "bc", 1);
    labels.set_label(vertices[1], "bc", 1);
    labels.set_label(cells[0], "mat", 7);

    MeshData {
        sieve,
        coordinates: None,
        sections,
        mixed_sections: Default::default(),
        labels: Some(labels),
        cell_types: None,
        discretization: None,
    }
}

fn build_coordinate_dm(
) -> CoordinateDM<f64, VecStorage<f64>> {
    let vertices = [pid(1), pid(2), pid(3), pid(4)];

    let mut atlas = Atlas::default();
    for vertex in vertices {
        atlas.try_insert(vertex, 2).unwrap();
    }

    let mut coords = Coordinates::try_new(2, 2, atlas).unwrap();
    coords.try_restrict_mut(vertices[0]).unwrap().copy_from_slice(&[0.0, 0.0]);
    coords.try_restrict_mut(vertices[1]).unwrap().copy_from_slice(&[1.0, 0.0]);
    coords.try_restrict_mut(vertices[2]).unwrap().copy_from_slice(&[1.0, 1.0]);
    coords.try_restrict_mut(vertices[3]).unwrap().copy_from_slice(&[0.0, 1.0]);

    let mut labels = LabelSet::new();
    labels.set_label(vertices[0], "corner", 1);
    labels.set_label(vertices[3], "corner", 1);

    let numbering = mesh_sieve::data::coordinate_dm::CoordinateNumbering::from_points(
        coords.section().atlas().points(),
    );

    CoordinateDM {
        coordinates: coords,
        labels: Some(labels),
        discretization: None,
        numbering,
    }
}

#[test]
fn renumber_preserves_adjacency_and_sections() {
    let mesh = build_mesh();
    let permutation = vec![pid(11), pid(3), pid(1), pid(10), pid(2), pid(4)];

    let renumbered = renumber_points(&mesh, &permutation).unwrap();

    let mut mapping = HashMap::new();
    for (idx, &old) in permutation.iter().enumerate() {
        mapping.insert(old, pid((idx + 1) as u64));
    }

    for old_src in mesh.sieve.points() {
        let new_src = mapping[&old_src];
        let old_cone: BTreeSet<_> = mesh
            .sieve
            .cone_points(old_src)
            .map(|dst| mapping[&dst])
            .collect();
        let new_cone: BTreeSet<_> = renumbered.sieve.cone_points(new_src).collect();
        assert_eq!(old_cone, new_cone);
    }

    let old_section = mesh.sections.get("field").unwrap();
    let new_section = renumbered.sections.get("field").unwrap();
    for &old in &[pid(1), pid(3), pid(10)] {
        let new = mapping[&old];
        let old_slice = old_section.try_restrict(old).unwrap();
        let new_slice = new_section.try_restrict(new).unwrap();
        assert_eq!(old_slice, new_slice);
    }

    let old_labels = mesh.labels.as_ref().unwrap();
    let new_labels = renumbered.labels.as_ref().unwrap();
    for (name, point, value) in old_labels.iter() {
        let new_point = mapping[&point];
        assert_eq!(new_labels.get_label(new_point, name), Some(value));
    }
}

#[test]
fn stratified_permutation_orders_vertices_first() {
    let mesh = build_mesh();
    let perm = stratified_permutation(&mesh.sieve, StratifiedOrdering::VertexFirst).unwrap();

    let vertices = [pid(1), pid(2), pid(3), pid(4)];
    let cells = [pid(10), pid(11)];

    let max_vertex_idx = vertices
        .iter()
        .map(|p| perm.iter().position(|q| q == p).unwrap())
        .max()
        .unwrap();
    let min_cell_idx = cells
        .iter()
        .map(|p| perm.iter().position(|q| q == p).unwrap())
        .min()
        .unwrap();

    assert!(max_vertex_idx < min_cell_idx);
}

#[test]
fn renumber_coordinate_dm_preserves_coordinates() {
    let mesh = build_mesh();
    let coordinate_dm = build_coordinate_dm();
    let permutation = vec![pid(11), pid(3), pid(1), pid(10), pid(2), pid(4)];

    let renumbered = renumber_coordinate_dm(&mesh, &coordinate_dm, &permutation).unwrap();

    let mut mapping = HashMap::new();
    for (idx, &old) in permutation.iter().enumerate() {
        mapping.insert(old, pid((idx + 1) as u64));
    }

    for &old in &[pid(1), pid(2), pid(3), pid(4)] {
        let new = mapping[&old];
        let old_slice = coordinate_dm
            .coordinates
            .try_restrict(old)
            .unwrap();
        let new_slice = renumbered
            .coordinates
            .try_restrict(new)
            .unwrap();
        assert_eq!(old_slice, new_slice);
    }

    let old_labels = coordinate_dm.labels.as_ref().unwrap();
    let new_labels = renumbered.labels.as_ref().unwrap();
    for (name, point, value) in old_labels.iter() {
        let new_point = mapping[&point];
        assert_eq!(new_labels.get_label(new_point, name), Some(value));
    }
}
