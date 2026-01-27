use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::overlap::overlap::Overlap;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::refine_mesh_with_ownership;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};
use mesh_sieve::topology::validation::validate_overlap_ownership_topology;

fn pid(id: u64) -> PointId {
    PointId::new(id).expect("valid point id")
}

struct LocalSetup {
    sieve: InMemorySieve<PointId, ()>,
    ownership: PointOwnership,
    overlap: Overlap,
    cell_types: Section<CellType, VecStorage<CellType>>,
}

fn build_local_setup(rank: usize) -> Result<LocalSetup, MeshSieveError> {
    let (c0, c1) = (pid(10), pid(11));
    let (v1, v2, v3, v4) = (pid(1), pid(2), pid(3), pid(4));

    let (cell, vertices) = if rank == 0 {
        (c0, vec![v1, v2, v3])
    } else {
        (c1, vec![v2, v1, v4])
    };

    let mut sieve = InMemorySieve::<PointId, ()>::default();
    for v in &vertices {
        sieve.add_arrow(cell, *v, ());
    }

    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(cell, 1)?;
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(cell, &[CellType::Triangle])?;

    let mut ownership = PointOwnership::default();
    if rank == 0 {
        for p in [cell, v1, v2, v3] {
            ownership.set_from_owner(p, 0, rank)?;
        }
    } else {
        ownership.set_from_owner(cell, 1, rank)?;
        ownership.set_from_owner(v4, 1, rank)?;
        for p in [v1, v2] {
            ownership.set_from_owner(p, 0, rank)?;
        }
    }

    let mut overlap = Overlap::default();
    if rank == 0 {
        for p in [v1, v2] {
            overlap.try_add_link(p, 1, p)?;
        }
    } else {
        for p in [v1, v2] {
            overlap.try_add_link(p, 0, p)?;
        }
    }

    Ok(LocalSetup {
        sieve,
        ownership,
        overlap,
        cell_types,
    })
}

#[test]
fn refine_two_ranks_with_overlap_validation() {
    let mut handles = Vec::new();

    for rank in 0..2 {
        handles.push(std::thread::spawn(move || -> Result<(), MeshSieveError> {
            let mut local = build_local_setup(rank)?;
            let refined = refine_mesh_with_ownership(
                &mut local.sieve,
                &local.cell_types,
                &local.ownership,
                rank,
            )?;

            validate_overlap_ownership_topology(
                &refined.sieve,
                &refined.ownership,
                Some(&local.overlap),
                rank,
            )?;

            Ok(())
        }));
    }

    for handle in handles {
        handle.join().expect("thread panicked").unwrap();
    }
}
