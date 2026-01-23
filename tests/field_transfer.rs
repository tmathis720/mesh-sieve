use mesh_sieve::algs::field_transfer::{
    transfer_section_by_nearest_cell_centroid, transfer_section_by_refinement_map,
};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::coordinates::Coordinates;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::{RefineOptions, refine_mesh, refine_mesh_with_options};
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

#[test]
fn transfer_scalar_by_cell_centroid_between_coarse_and_refined_meshes() {
    let mut sieve = InMemorySieve::<PointId, ()>::default();
    let cell = pt(20);
    let vertices = [pt(1), pt(2), pt(3)];
    for v in vertices {
        sieve.add_arrow(cell, v, ());
    }

    let cell_types = cell_types_section(&[(cell, CellType::Triangle)]);

    let mut coord_atlas = Atlas::default();
    for v in vertices {
        coord_atlas.try_insert(v, 2).unwrap();
    }
    let mut coarse_coords = Coordinates::<f64, VecStorage<f64>>::try_new(2, coord_atlas)
        .expect("coordinate atlas should be valid");
    coarse_coords.try_restrict_mut(vertices[0]).unwrap().copy_from_slice(&[0.0, 0.0]);
    coarse_coords.try_restrict_mut(vertices[1]).unwrap().copy_from_slice(&[1.0, 0.0]);
    coarse_coords.try_restrict_mut(vertices[2]).unwrap().copy_from_slice(&[0.0, 1.0]);

    let refined = refine_mesh_with_options(
        &mut sieve,
        &cell_types,
        Some(&coarse_coords),
        RefineOptions::default(),
    )
    .expect("triangle refinement with coordinates should succeed");
    let refined_coords = refined
        .coordinates
        .as_ref()
        .expect("refinement should generate coordinates");

    let mut atlas = Atlas::default();
    atlas.try_insert(cell, 1).unwrap();
    let mut coarse = Section::<f64, VecStorage<f64>>::new(atlas);
    coarse.try_set(cell, &[2.5]).unwrap();

    let fine_cells: Vec<_> = refined
        .cell_refinement
        .iter()
        .flat_map(|(_, fine_cells)| fine_cells.iter().map(|(cell_id, _)| *cell_id))
        .collect();

    let fine = transfer_section_by_nearest_cell_centroid(
        &coarse,
        &sieve,
        &coarse_coords,
        &refined.sieve,
        refined_coords,
        fine_cells,
    )
    .expect("nearest cell-centroid transfer should succeed");

    for (fine_cell, data) in fine.iter() {
        assert_eq!(data, &[2.5], "fine cell {fine_cell:?} should inherit value");
    }
}
