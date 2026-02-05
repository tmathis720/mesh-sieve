use mesh_sieve::data::discretization::{
    Discretization, DiscretizationMetadata, FieldDiscretization, RegionKey,
};
use mesh_sieve::topology::cell_type::CellType;

#[test]
fn field_discretization_lookup_by_region() {
    let mut field = FieldDiscretization::new();
    let label_region = RegionKey::label("material", 1);
    let cell_region = RegionKey::cell_type(CellType::Hexahedron);
    let label_metadata = DiscretizationMetadata::new("lagrange_p1", "gauss2");
    let cell_metadata = DiscretizationMetadata::new("serendipity_p2", "gauss4");

    field.set_metadata(label_region.clone(), label_metadata.clone());
    field.set_metadata(cell_region.clone(), cell_metadata.clone());

    assert_eq!(field.metadata_for(&label_region), Some(&label_metadata));
    assert_eq!(field.metadata_for(&cell_region), Some(&cell_metadata));
    assert!(
        field
            .metadata_for(&RegionKey::label("material", 2))
            .is_none()
    );
}

#[test]
fn discretization_container_field_lookup() {
    let mut discretization = Discretization::new();
    let mut field = FieldDiscretization::new();
    let region = RegionKey::label("phase", 7);
    let metadata = DiscretizationMetadata::new("lagrange_p2", "gauss5");

    field.set_metadata(region.clone(), metadata.clone());
    discretization.insert_field("temperature", field);

    let field = discretization.field("temperature").expect("field missing");
    assert_eq!(field.metadata_for(&region), Some(&metadata));
}
