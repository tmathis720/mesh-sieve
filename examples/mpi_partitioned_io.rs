// cargo mpirun -n 2 --features mpi-support --example mpi_partitioned_io
// Demonstrates partitioned mesh I/O with overlap metadata using MPI.
#[cfg(feature = "mpi-support")]
use mesh_sieve::data::atlas::Atlas;
#[cfg(feature = "mpi-support")]
use mesh_sieve::data::coordinates::Coordinates;
#[cfg(feature = "mpi-support")]
use mesh_sieve::data::section::Section;
#[cfg(feature = "mpi-support")]
use mesh_sieve::data::storage::VecStorage;
#[cfg(feature = "mpi-support")]
use mesh_sieve::topology::cell_type::CellType;
#[cfg(feature = "mpi-support")]
use mesh_sieve::topology::point::PointId;
#[cfg(feature = "mpi-support")]
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
#[cfg(feature = "mpi-support")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use mesh_sieve::algs::communicator::{Communicator, MpiComm};
    use mesh_sieve::algs::distribute::{
        DistributionConfig, ProvidedPartition, distribute_with_overlap,
    };
    use mesh_sieve::io::gmsh::{GmshReader, GmshWriter};
    use mesh_sieve::io::{GatherPolicy, MeshData, read_partitioned_mesh, write_partitioned_mesh};
    use std::fs;
    use std::path::PathBuf;

    let comm = MpiComm::new()?;
    if comm.size() != 2 {
        eprintln!("This example expects exactly 2 MPI ranks");
        return Ok(());
    }

    let (global_mesh, cells) = build_global_mesh()?;

    let parts = vec![0usize, 1];
    let partitioner = ProvidedPartition { parts: &parts };
    let distributed = distribute_with_overlap(
        &global_mesh,
        &cells,
        &partitioner,
        DistributionConfig {
            overlap_depth: 1,
            synchronize_sections: true,
        },
        &comm,
    )?;

    let mut local_mesh = MeshData::new(distributed.sieve);
    local_mesh.coordinates = distributed.coordinates;
    local_mesh.sections = distributed.sections;
    local_mesh.mixed_sections = distributed.mixed_sections;
    local_mesh.labels = distributed.labels;
    local_mesh.cell_types = distributed.cell_types;
    local_mesh.discretization = distributed.discretization;

    let writer = GmshWriter::default();
    let reader = GmshReader::default();

    let local_dir = PathBuf::from("output/partitioned_io/local");
    if comm.rank() == 0 {
        let _ = fs::remove_dir_all(&local_dir);
    }
    comm.barrier();

    write_partitioned_mesh(
        &writer,
        &local_dir,
        "mesh",
        &local_mesh,
        Some(&distributed.overlap),
        &comm,
        GatherPolicy::LocalPieces,
    )?;
    comm.barrier();

    let local_loaded = read_partitioned_mesh(&reader, &local_dir, "mesh", &comm)?;
    print_overlap_summary("local", comm.rank(), &local_loaded.overlap);

    let bundle_dir = PathBuf::from("output/partitioned_io/bundle");
    if comm.rank() == 0 {
        let _ = fs::remove_dir_all(&bundle_dir);
    }
    comm.barrier();

    write_partitioned_mesh(
        &writer,
        &bundle_dir,
        "mesh",
        &local_mesh,
        Some(&distributed.overlap),
        &comm,
        GatherPolicy::GatherToRoot {
            max_total_bytes: 64_000,
        },
    )?;
    comm.barrier();

    let bundle_loaded = read_partitioned_mesh(&reader, &bundle_dir, "mesh", &comm)?;
    print_overlap_summary("bundle", comm.rank(), &bundle_loaded.overlap);

    Ok(())
}

#[cfg(feature = "mpi-support")]
fn build_global_mesh() -> Result<
    (
        mesh_sieve::io::MeshData<
            InMemorySieve<PointId, ()>,
            f64,
            VecStorage<f64>,
            VecStorage<CellType>,
        >,
        Vec<PointId>,
    ),
    Box<dyn std::error::Error>,
> {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell0 = PointId::new(100)?;
    let cell1 = PointId::new(101)?;
    let v1 = PointId::new(1)?;
    let v2 = PointId::new(2)?;
    let v3 = PointId::new(3)?;
    let v4 = PointId::new(4)?;

    sieve.add_arrow(cell0, v1, ());
    sieve.add_arrow(cell0, v2, ());
    sieve.add_arrow(cell0, v3, ());
    sieve.add_arrow(cell1, v2, ());
    sieve.add_arrow(cell1, v3, ());
    sieve.add_arrow(cell1, v4, ());

    let mut coord_atlas = Atlas::default();
    for vertex in [v1, v2, v3, v4] {
        coord_atlas.try_insert(vertex, 2)?;
    }
    let mut coords = Coordinates::try_new(2, 2, coord_atlas)?;
    coords.section_mut().try_set(v1, &[0.0, 0.0])?;
    coords.section_mut().try_set(v2, &[1.0, 0.0])?;
    coords.section_mut().try_set(v3, &[0.0, 1.0])?;
    coords.section_mut().try_set(v4, &[1.0, 1.0])?;

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell0, 1)?;
    cell_atlas.try_insert(cell1, 1)?;
    let mut cell_types = Section::new(cell_atlas);
    cell_types.try_set(cell0, &[CellType::Triangle])?;
    cell_types.try_set(cell1, &[CellType::Triangle])?;

    let mut mesh = mesh_sieve::io::MeshData::new(sieve);
    mesh.coordinates = Some(coords);
    mesh.cell_types = Some(cell_types);

    Ok((mesh, vec![cell0, cell1]))
}

#[cfg(feature = "mpi-support")]
fn print_overlap_summary(
    label: &str,
    rank: usize,
    overlap: &Option<mesh_sieve::overlap::overlap::Overlap>,
) {
    if let Some(overlap) = overlap {
        let neighbors: Vec<_> = overlap.neighbor_ranks().collect();
        for neighbor in neighbors {
            let link_count = overlap.links_to(neighbor).count();
            println!("[{label}] rank {rank} neighbor {neighbor} has {link_count} links",);
        }
    } else {
        println!("[{label}] rank {rank} has no overlap metadata");
    }
}

#[cfg(not(feature = "mpi-support"))]
fn main() {
    eprintln!("This example requires the 'mpi-support' feature to run.");
}
