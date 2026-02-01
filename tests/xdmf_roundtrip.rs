use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::xdmf::{XdmfReader, XdmfWriter};
use mesh_sieve::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn build_mesh(
) -> MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>> {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pid(20);
    for v in [pid(4), pid(5), pid(6)] {
        sieve.add_arrow(cell, v, ());
    }

    let mut coord_atlas = Atlas::default();
    for v in [pid(4), pid(5), pid(6)] {
        coord_atlas.try_insert(v, 2).unwrap();
    }
    let mut coords = Coordinates::try_new(2, 2, coord_atlas).unwrap();
    coords.try_restrict_mut(pid(4)).unwrap().copy_from_slice(&[0.0, 0.0]);
    coords.try_restrict_mut(pid(5)).unwrap().copy_from_slice(&[1.0, 0.0]);
    coords.try_restrict_mut(pid(6)).unwrap().copy_from_slice(&[0.0, 1.0]);

    let mut section_atlas = Atlas::default();
    for v in [pid(4), pid(5), pid(6)] {
        section_atlas.try_insert(v, 1).unwrap();
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(section_atlas);
    section.try_set(pid(4), &[5.0]).unwrap();
    section.try_set(pid(5), &[6.0]).unwrap();
    section.try_set(pid(6), &[7.0]).unwrap();

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(cell, &[CellType::Triangle]).unwrap();

    let mut labels = LabelSet::new();
    labels.set_label(pid(4), "corner", 2);
    labels.set_label(cell, "region", 9);

    let mut sections = std::collections::BTreeMap::new();
    sections.insert("pressure".to_string(), section);

    MeshData {
        sieve,
        coordinates: Some(coords),
        sections,
        mixed_sections: Default::default(),
        labels: Some(labels),
        cell_types: Some(cell_types),
        discretization: None,
    }
}

#[test]
fn xdmf_round_trip_preserves_topology_labels_and_sections() {
    let mesh = build_mesh();
    let writer = XdmfWriter::default();
    let mut output = Vec::new();
    writer.write(&mut output, &mesh).expect("write xdmf");

    let reader = XdmfReader::default();
    let round_tripped = reader.read(output.as_slice()).expect("read xdmf");

    let cell = pid(20);
    let nodes: Vec<_> = round_tripped.sieve.cone_points(cell).collect();
    assert_eq!(nodes, vec![pid(4), pid(5), pid(6)]);

    let coords = round_tripped.coordinates.as_ref().expect("coords");
    assert_eq!(coords.dimension(), 2);
    assert_eq!(coords.try_restrict(pid(5)).unwrap(), &[1.0, 0.0]);

    let cell_types = round_tripped.cell_types.as_ref().expect("cell types");
    assert_eq!(cell_types.try_restrict(cell).unwrap()[0], CellType::Triangle);

    let labels = round_tripped.labels.as_ref().expect("labels");
    assert_eq!(labels.get_label(pid(4), "corner"), Some(2));
    assert_eq!(labels.get_label(cell, "region"), Some(9));

    let section = round_tripped.sections.get("pressure").expect("section");
    assert_eq!(section.try_restrict(pid(6)).unwrap(), &[7.0]);
}
