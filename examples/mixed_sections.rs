use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::mixed_section::MixedSectionStore;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::{
    gmsh::GmshReader, gmsh::GmshWriter, MeshData, SieveSectionReader, SieveSectionWriter,
};
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::{InMemorySieve, MutableSieve, Sieve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let p1 = PointId::new(1)?;
    let p2 = PointId::new(2)?;
    let elem = PointId::new(3)?;

    let mut sieve = InMemorySieve::default();
    MutableSieve::add_point(&mut sieve, p1);
    MutableSieve::add_point(&mut sieve, p2);
    MutableSieve::add_point(&mut sieve, elem);
    sieve.add_arrow(elem, p1, ());
    sieve.add_arrow(elem, p2, ());

    let mut coord_atlas = Atlas::default();
    coord_atlas.try_insert(p1, 2)?;
    coord_atlas.try_insert(p2, 2)?;
    let mut coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, coord_atlas)?;
    coords.section_mut().try_set(p1, &[0.0, 0.0])?;
    coords.section_mut().try_set(p2, &[1.0, 0.0])?;

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(elem, 1)?;
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(elem, &[CellType::Segment])?;

    let mut temp_atlas = Atlas::default();
    temp_atlas.try_insert(p1, 1)?;
    temp_atlas.try_insert(p2, 1)?;
    let mut temperature = Section::<f64, VecStorage<f64>>::new(temp_atlas);
    temperature.try_set(p1, &[300.0])?;
    temperature.try_set(p2, &[310.0])?;

    let mut mat_atlas = Atlas::default();
    mat_atlas.try_insert(elem, 1)?;
    let mut material_id = Section::<i32, VecStorage<i32>>::new(mat_atlas);
    material_id.try_set(elem, &[7])?;

    let mut mixed_sections = MixedSectionStore::default();
    mixed_sections.insert::<f64>("temperature", temperature);
    mixed_sections.insert::<i32>("material_id", material_id);

    let mut mesh = MeshData::new(sieve);
    mesh.coordinates = Some(coords);
    mesh.cell_types = Some(cell_types);
    mesh.mixed_sections = mixed_sections;

    let mut buffer = Vec::new();
    GmshWriter::default().write(&mut buffer, &mesh)?;

    let round_trip = GmshReader::default().read(buffer.as_slice())?;
    let temps = round_trip
        .mixed_sections
        .get::<f64>("temperature")
        .expect("temperature section missing");
    let mats = round_trip
        .mixed_sections
        .get::<i32>("material_id")
        .expect("material_id section missing");

    println!("temperature len = {}", temps.atlas().points().count());
    println!("material_id len = {}", mats.atlas().points().count());
    Ok(())
}
