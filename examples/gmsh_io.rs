use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::gmsh::GmshReader;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::topology::sieve::InMemorySieve;

fn main() -> Result<(), MeshSieveError> {
    let msh = r#"
$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
1
1 3 0 1 2 3 4
$EndElements
"#;

    let reader = GmshReader::default();
    let mesh = reader.read(msh.as_bytes())?;

    let _sieve: InMemorySieve<_, _> = mesh.sieve;
    let mut coordinates = mesh.coordinates.expect("Gmsh reader populates coordinates");

    for (point, values) in coordinates.section().iter() {
        println!("point {point:?} -> {values:?}");
    }

    if let Some((point, _)) = coordinates.section().iter().next() {
        coordinates
            .section_mut()
            .try_set(point, &[42.0, 0.0, 0.0])?;
    }

    Ok(())
}
