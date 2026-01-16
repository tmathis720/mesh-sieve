use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::refine::SievedArray;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::refine_mesh;
use mesh_sieve::topology::sieve::InMemorySieve;

fn pid(i: u64) -> PointId {
    PointId::new(i).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Coarse mesh: single triangle cell 10 -> vertices 1,2,3.
    let mut coarse = InMemorySieve::<PointId, ()>::default();
    coarse.add_arrow(pid(10), pid(1), ());
    coarse.add_arrow(pid(10), pid(2), ());
    coarse.add_arrow(pid(10), pid(3), ());

    // Cell-type metadata: cell 10 is a triangle.
    let mut cell_atlas = Atlas::default();
    cell_atlas.try_insert(pid(10), 1)?;
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(pid(10), &[CellType::Triangle])?;

    // Refine the topology and build a coarse→fine mapping.
    let refined = refine_mesh(&mut coarse, &cell_types)?;

    // Coarse cell data (one value per cell).
    let mut coarse_values = SievedArray::<PointId, f64>::new(cell_types.atlas().clone());
    coarse_values.try_set(pid(10), &[1.0])?;

    // Build atlas for refined cells.
    let mut fine_atlas = Atlas::default();
    for (_, fine_cells) in &refined.cell_refinement {
        for (fine_cell, _) in fine_cells {
            fine_atlas.try_insert(*fine_cell, 1)?;
        }
    }

    let mut fine_values = SievedArray::<PointId, f64>::new(fine_atlas);
    fine_values.try_refine_with_sifter(&coarse_values, &refined.cell_refinement)?;

    // All refined cells inherit the coarse value.
    for (_, fine_cells) in refined.cell_refinement {
        for (fine_cell, _) in fine_cells {
            assert_eq!(fine_values.try_get(fine_cell)?[0], 1.0);
        }
    }

    Ok(())
}
