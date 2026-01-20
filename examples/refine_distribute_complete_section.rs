// cargo run --example refine_distribute_complete_section
use mesh_sieve::algs::communicator::RayonComm;
use mesh_sieve::algs::completion::complete_section_with_ownership;
use mesh_sieve::algs::distribute::{distribute_with_overlap, DistributionConfig, ProvidedPartition};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::constrained_section::ConstrainedSection;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::io::MeshData;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::ownership::PointOwnership;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::refine::refine_mesh_with_ownership;
use mesh_sieve::topology::sieve::{InMemorySieve, Sieve};

fn pid(id: u64) -> PointId {
    PointId::new(id).expect("valid point id")
}

fn build_refined_mesh_data() -> Result<
    (
        MeshData<InMemorySieve<PointId, ()>, f64, VecStorage<f64>, VecStorage<CellType>>,
        Vec<PointId>,
        Vec<usize>,
        PointId,
        f64,
    ),
    MeshSieveError,
> {
    let mut coarse = InMemorySieve::<PointId, ()>::default();
    let (c0, c1) = (pid(10), pid(11));
    let (v1, v2, v3, v4) = (pid(1), pid(2), pid(3), pid(4));

    for v in [v1, v2, v3] {
        coarse.add_arrow(c0, v, ());
    }
    for v in [v1, v3, v4] {
        coarse.add_arrow(c1, v, ());
    }

    let mut cell_atlas = Atlas::default();
    for p in [c0, c1] {
        cell_atlas.try_insert(p, 1)?;
    }
    let mut cell_types = Section::<CellType, VecStorage<CellType>>::new(cell_atlas);
    cell_types.try_set(c0, &[CellType::Triangle])?;
    cell_types.try_set(c1, &[CellType::Triangle])?;

    let mut ownership = PointOwnership::default();
    for p in coarse.points() {
        ownership.set_from_owner(p, 0, 0)?;
    }

    let refined = refine_mesh_with_ownership(&mut coarse, &cell_types, &ownership, 0)?;
    let new_point = refined
        .sieve
        .points()
        .max_by_key(|p| p.get())
        .expect("refined mesh has points");
    assert_eq!(refined.ownership.owner_or_err(new_point)?, 0);

    let mut field_atlas = Atlas::default();
    for p in refined.sieve.points() {
        field_atlas.try_insert(p, 1)?;
    }
    let mut section = Section::<f64, VecStorage<f64>>::new(field_atlas);
    for p in refined.sieve.points() {
        section.try_set(p, &[p.get() as f64])?;
    }

    let constrained_point = refined
        .sieve
        .points()
        .find(|p| refined.sieve.cone_points(*p).next().is_none())
        .expect("at least one vertex");
    let constrained_value = -10.0;
    let mut constrained = ConstrainedSection::new(section);
    constrained.insert_constraint(constrained_point, 0, constrained_value)?;
    constrained.apply_constraints()?;
    assert_eq!(
        constrained
            .section()
            .try_restrict(constrained_point)?[0],
        constrained_value
    );

    let mut mesh_data = MeshData::new(refined.sieve);
    mesh_data
        .sections
        .insert("temperature".to_string(), constrained.into_section());

    let mut cells: Vec<_> = mesh_data.sieve.base_points().collect();
    cells.sort_by_key(|p| p.get());
    let parts: Vec<_> = (0..cells.len()).map(|i| i % 2).collect();

    Ok((
        mesh_data,
        cells,
        parts,
        constrained_point,
        constrained_value,
    ))
}

fn main() -> Result<(), MeshSieveError> {
    let mut handles = Vec::new();

    for rank in 0..2 {
        handles.push(std::thread::spawn(move || -> Result<(), MeshSieveError> {
            let (mesh_data, cells, parts, constrained_point, constrained_value) =
                build_refined_mesh_data()?;
            let comm = RayonComm::new(rank, 2);
            let config = DistributionConfig {
                overlap_depth: 1,
                synchronize_sections: false,
            };

            let dist = distribute_with_overlap(
                &mesh_data,
                &cells,
                &ProvidedPartition { parts: &parts },
                config,
                &comm,
            )?;

            let mut local_section = dist
                .sections
                .get("temperature")
                .expect("section distributed")
                .clone();
            let ghost_points: Vec<_> = dist.ownership.ghost_points().collect();
            assert!(!ghost_points.is_empty());

            if let Some(&ghost_point) = ghost_points.first() {
                let before = local_section.try_restrict(ghost_point)?[0];
                let expected = mesh_data
                    .sections
                    .get("temperature")
                    .expect("global section")
                    .try_restrict(ghost_point)?[0];
                assert_ne!(before, expected);
            }

            complete_section_with_ownership::<f64, VecStorage<f64>, CopyDelta, _>(
                &mut local_section,
                &dist.overlap,
                &dist.ownership,
                &comm,
                rank,
            )?;

            for ghost_point in ghost_points {
                let expected = mesh_data
                    .sections
                    .get("temperature")
                    .expect("global section")
                    .try_restrict(ghost_point)?[0];
                assert_eq!(local_section.try_restrict(ghost_point)?[0], expected);
            }

            assert_eq!(
                local_section.try_restrict(constrained_point)?[0],
                constrained_value
            );

            Ok(())
        }));
    }

    for handle in handles {
        handle.join().expect("thread panicked")?;
    }

    Ok(())
}
