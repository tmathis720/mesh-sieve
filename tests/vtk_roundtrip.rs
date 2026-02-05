use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::vtk::{VtkReader, VtkWriter};
use mesh_sieve::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn pid(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn build_mesh() -> MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>
{
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pid(10);
    for v in [pid(1), pid(2), pid(3)] {
        sieve.add_arrow(cell, v, ());
    }

    let mut coord_atlas = Atlas::default();
    for v in [pid(1), pid(2), pid(3)] {
        coord_atlas.try_insert(v, 2).unwrap();
    }
    let mut coords = Coordinates::try_new(2, 2, coord_atlas).unwrap();
    coords
        .try_restrict_mut(pid(1))
        .unwrap()
        .copy_from_slice(&[0.0, 0.0]);
    coords
        .try_restrict_mut(pid(2))
        .unwrap()
        .copy_from_slice(&[1.0, 0.0]);
    coords
        .try_restrict_mut(pid(3))
        .unwrap()
        .copy_from_slice(&[0.0, 1.0]);

    let mut section_atlas = Atlas::default();
    for v in [pid(1), pid(2), pid(3)] {
        section_atlas.try_insert(v, 1).unwrap();
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(section_atlas);
    section.try_set(pid(1), &[10.0]).unwrap();
    section.try_set(pid(2), &[20.0]).unwrap();
    section.try_set(pid(3), &[30.0]).unwrap();

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell, 1).unwrap();
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(cell, &[CellType::Triangle]).unwrap();

    let mut labels = LabelSet::new();
    labels.set_label(pid(1), "corner", 1);
    labels.set_label(cell, "region", 7);

    let mut sections = std::collections::BTreeMap::new();
    sections.insert("temperature".to_string(), section);

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
fn vtk_round_trip_preserves_topology_labels_and_sections() {
    let mesh = build_mesh();
    let writer = VtkWriter::default();
    let mut output = Vec::new();
    writer.write(&mut output, &mesh).expect("write vtk");

    let reader = VtkReader::default();
    let round_tripped = reader.read(output.as_slice()).expect("read vtk");

    let cell = pid(10);
    let nodes: Vec<_> = round_tripped.sieve.cone_points(cell).collect();
    assert_eq!(nodes, vec![pid(1), pid(2), pid(3)]);

    let coords = round_tripped.coordinates.as_ref().expect("coords");
    assert_eq!(coords.dimension(), 2);
    assert_eq!(coords.try_restrict(pid(2)).unwrap(), &[1.0, 0.0]);

    let cell_types = round_tripped.cell_types.as_ref().expect("cell types");
    assert_eq!(
        cell_types.try_restrict(cell).unwrap()[0],
        CellType::Triangle
    );

    let labels = round_tripped.labels.as_ref().expect("labels");
    assert_eq!(labels.get_label(pid(1), "corner"), Some(1));
    assert_eq!(labels.get_label(cell, "region"), Some(7));

    let section = round_tripped.sections.get("temperature").expect("section");
    assert_eq!(section.try_restrict(pid(3)).unwrap(), &[30.0]);
}
