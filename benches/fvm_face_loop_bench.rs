use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use mesh_sieve::discretization::runtime::{CellGeometry, FaceGeometry, FluxStencil};
use mesh_sieve::physics::fvm::FvmInputs;
use mesh_sieve::topology::point::PointId;

fn pid(raw: u64) -> PointId {
    PointId::new(raw + 1).expect("nonzero PointId")
}

fn synthetic_inputs(n_internal: usize, n_boundary: usize) -> FvmInputs {
    let n_cells = n_internal + 2;
    let mut stencils = Vec::with_capacity(n_internal + n_boundary);
    let mut cell_geometry = Vec::with_capacity(n_cells);
    let mut face_geometry = Vec::with_capacity(n_internal + n_boundary);

    for c in 0..n_cells {
        let cpid = pid(c as u64);
        cell_geometry.push((
            cpid,
            CellGeometry {
                centroid: vec![c as f64, 0.0, 0.0],
                volume: 1.0,
            },
        ));
    }

    for i in 0..n_internal {
        let face = pid((100_000 + i) as u64);
        let left = pid(i as u64);
        let right = pid((i + 1) as u64);
        stencils.push(FluxStencil {
            face,
            left,
            right: Some(right),
        });
        face_geometry.push((
            face,
            FaceGeometry {
                face,
                centroid: vec![i as f64 + 0.5, 0.0, 0.0],
                normal: vec![1.0, 0.0, 0.0],
                area: 1.0,
                neighbors: vec![left, right],
            },
        ));
    }

    for i in 0..n_boundary {
        let face = pid((200_000 + i) as u64);
        let left = pid((i % n_cells) as u64);
        stencils.push(FluxStencil {
            face,
            left,
            right: None,
        });
        face_geometry.push((
            face,
            FaceGeometry {
                face,
                centroid: vec![i as f64, 1.0, 0.0],
                normal: vec![0.0, 1.0, 0.0],
                area: 1.0,
                neighbors: vec![left],
            },
        ));
    }

    FvmInputs::new(stencils, cell_geometry, face_geometry)
}

fn bench_fvm_face_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("fvm_face_loop");
    for &(internal, boundary) in &[(100_000usize, 10_000usize), (500_000, 50_000)] {
        let mut inputs = synthetic_inputs(internal, boundary);
        inputs.build_packed_cache();

        group.bench_with_input(
            BenchmarkId::new("lookup_maps", internal),
            &internal,
            |b, _| {
                b.iter(|| {
                    let mut accum = 0.0;
                    for s in inputs.internal_faces() {
                        let fg = inputs.face_metrics(s.face).unwrap();
                        let lc = inputs.cell_metrics(s.left).unwrap();
                        let rc = inputs.cell_metrics(s.right.unwrap()).unwrap();
                        accum += fg.area + lc.volume + rc.volume;
                    }
                    black_box(accum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("packed_cache", internal),
            &internal,
            |b, _| {
                let packed = inputs.packed().unwrap();
                b.iter(|| {
                    let mut accum = 0.0;
                    for s in &packed.internal_faces {
                        let fg = &inputs.face_geometry[s.face_geom_idx].1;
                        let lc = &inputs.cell_geometry[s.owner_cell_geom_idx].1;
                        let rc = &inputs.cell_geometry[s.neighbor_cell_geom_idx].1;
                        accum += fg.area + lc.volume + rc.volume;
                    }
                    black_box(accum);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_fvm_face_loop);
criterion_main!(benches);
