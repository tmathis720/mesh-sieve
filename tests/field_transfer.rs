use mesh_sieve::algs::field_transfer::transfer_section_by_refinement_map;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::refine_mesh;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn pt(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn cell_types_section(cells: &[(PointId, CellType)]) -> Section<CellType, VecStorage<CellType>> {
    let mut atlas = Atlas::default();
    for (cell, _) in cells {
        atlas.try_insert(*cell, 1).unwrap();
    }
    let mut section = Section::<CellType, VecStorage<CellType>>::new(atlas);
    for (cell, cell_type) in cells {
        section.try_set(*cell, &[*cell_type]).unwrap();
    }
    section
}

#[test]
fn transfer_scalar_from_coarse_to_refined_cells() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(10);
    for v in [pt(1), pt(2), pt(3)] {
        sieve.add_arrow(cell, v, ());
    }

    let cell_types = cell_types_section(&[(cell, CellType::Triangle)]);
    let refined = refine_mesh(&mut sieve, &cell_types).expect("triangle refinement should succeed");

    let mut atlas = Atlas::default();
    atlas.try_insert(cell, 1).unwrap();
    let mut coarse = Section::<f64, VecStorage<f64>>::new(atlas);
    coarse.try_set(cell, &[3.14]).unwrap();

    let fine = transfer_section_by_refinement_map(&coarse, &refined.cell_refinement)
        .expect("refinement map transfer should succeed");

    for (_coarse_cell, fine_cells) in refined.cell_refinement.iter() {
        for (fine_cell, _) in fine_cells {
            let data = fine.try_restrict(*fine_cell).unwrap();
            assert_eq!(data, &[3.14]);
        }
    }
}
