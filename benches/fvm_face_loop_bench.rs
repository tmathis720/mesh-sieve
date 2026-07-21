use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
#[cfg(feature = "cuda")]
use mesh_sieve::accelerator::{AcceleratorBackend, cuda::CudaBackend, cuda::CudaOptions};
use mesh_sieve::accelerator::{CpuBackend, DeviceFvmOperator, DeviceFvmState, PlanEpochs};
use mesh_sieve::discretization::runtime::{
    CellGeometry, FaceGeometry, FiniteVolumeMetadata, FluxStencil,
};
use mesh_sieve::physics::fvm::{
    BoundaryCondition, ConvectiveScheme, DiffusionSettings, FvBoundaryBranch, FvBoundaryPolicy,
    FvmInputs, FvmPackedCache, FvmSchemeSettings, LimiterOption, NonOrthogonalCorrectionMode,
    ReconstructionGradient, ReconstructionMode, ReconstructionSettings,
    UnsupportedBoundaryBehavior,
};
use mesh_sieve::topology::point::PointId;
use std::collections::{HashMap, HashSet};

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

        group.bench_with_input(
            BenchmarkId::new("packed_cache_reuse", internal),
            &internal,
            |b, _| {
                let mut cache = FvmPackedCache::default();
                b.iter(|| {
                    inputs.build_packed_cache_into(&mut cache);
                    let mut accum = 0.0;
                    for s in &cache.packed.internal_faces {
                        let fg = &inputs.face_geometry[s.face_geom_idx].1;
                        let lc = &inputs.cell_geometry[s.owner_cell_geom_idx].1;
                        let rc = &inputs.cell_geometry[s.neighbor_cell_geom_idx].1;
                        accum += fg.area + lc.volume + rc.volume;
                    }
                    black_box(accum);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("preindexed_triplets", internal),
            &internal,
            |b, _| {
                let face_idx = inputs.internal_face_geometry_indices();
                let owner_neigh = inputs.internal_owner_neighbor_indices();
                b.iter(|| {
                    let mut accum = 0.0;
                    for (i, (owner, neigh)) in owner_neigh.iter().enumerate() {
                        let fg = &inputs.face_geometry[face_idx[i]].1;
                        let lc = &inputs.cell_geometry[*owner].1;
                        let rc = &inputs.cell_geometry[*neigh].1;
                        accum += fg.area + lc.volume + rc.volume;
                    }
                    black_box(accum);
                });
            },
        );
    }
    group.finish();
}

fn bench_resident_fvm_operator(c: &mut Criterion) {
    let mut group = c.benchmark_group("fvm_resident_operator_cpu");
    for &(internal, boundary) in &[(10_000usize, 1_000usize), (100_000, 10_000)] {
        group.throughput(Throughput::Elements((internal + boundary) as u64));
        let inputs = synthetic_inputs(internal, boundary);
        let boundary_face_branches = inputs
            .boundary_faces()
            .map(|face| (face.face, FvBoundaryBranch::Inflow))
            .collect::<HashMap<_, _>>();
        let policy = FvBoundaryPolicy {
            boundary_face_branches,
            allowed_branches: HashSet::from([FvBoundaryBranch::Inflow]),
            convective_branch_hooks: HashMap::from([(
                FvBoundaryBranch::Inflow,
                BoundaryCondition::Dirichlet { value: 0.0 },
            )]),
            diffusive_branch_hooks: HashMap::from([(
                FvBoundaryBranch::Inflow,
                BoundaryCondition::Neumann { gradient: 0.0 },
            )]),
            unsupported_behavior: UnsupportedBoundaryBehavior::Error,
        };
        let schemes = FvmSchemeSettings {
            convective: ConvectiveScheme::Upwind,
            reconstruction: ReconstructionSettings {
                mode: ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
                limiter: LimiterOption::None,
            },
            diffusion: DiffusionSettings {
                diffusivity: 0.01,
                non_orthogonal_mode: NonOrthogonalCorrectionMode::FullyCorrected,
            },
        };
        let backend = CpuBackend;
        let operator = DeviceFvmOperator::compile(
            &backend,
            &inputs,
            FiniteVolumeMetadata::new(1),
            None,
            &policy,
            schemes,
            PlanEpochs::default(),
        )
        .unwrap();
        let mut state = DeviceFvmState::upload(
            &backend,
            &operator.plan,
            &vec![1.0_f64; operator.plan.cell_ids.len()],
            &vec![0.1_f64; operator.plan.face_count()],
            &vec![0.0_f64; operator.plan.boundary_face_count],
        )
        .unwrap();
        group.bench_with_input(
            BenchmarkId::new("evaluate_residual", internal),
            &internal,
            |b, _| {
                b.iter(|| {
                    operator
                        .evaluate_residual(&mut state, PlanEpochs::default())
                        .unwrap();
                    black_box(state.residual.as_slice());
                });
            },
        );
    }
    group.finish();
}

fn bench_fvm_operator_setup(c: &mut Criterion) {
    let mut group = c.benchmark_group("fvm_operator_setup_cpu");
    for &(internal, boundary) in &[(10_000usize, 1_000usize), (100_000, 10_000)] {
        let inputs = synthetic_inputs(internal, boundary);
        let boundary_face_branches = inputs
            .boundary_faces()
            .map(|face| (face.face, FvBoundaryBranch::Inflow))
            .collect::<HashMap<_, _>>();
        let policy = FvBoundaryPolicy {
            boundary_face_branches,
            allowed_branches: HashSet::from([FvBoundaryBranch::Inflow]),
            convective_branch_hooks: HashMap::from([(
                FvBoundaryBranch::Inflow,
                BoundaryCondition::Dirichlet { value: 0.0 },
            )]),
            diffusive_branch_hooks: HashMap::from([(
                FvBoundaryBranch::Inflow,
                BoundaryCondition::Neumann { gradient: 0.0 },
            )]),
            unsupported_behavior: UnsupportedBoundaryBehavior::Error,
        };
        let schemes = FvmSchemeSettings {
            convective: ConvectiveScheme::HighResolution {
                blend: 0.75,
                limiter: mesh_sieve::physics::fvm::SlopeLimiterFamily::VanLeer,
            },
            reconstruction: ReconstructionSettings {
                mode: ReconstructionMode::LeastSquaresWithGreenGaussFallback,
                limiter: LimiterOption::None,
            },
            diffusion: DiffusionSettings {
                diffusivity: 0.01,
                non_orthogonal_mode: NonOrthogonalCorrectionMode::FullyCorrected,
            },
        };
        let backend = CpuBackend;
        group.bench_with_input(BenchmarkId::new("compile", internal), &internal, |b, _| {
            b.iter(|| {
                black_box(
                    DeviceFvmOperator::compile(
                        &backend,
                        &inputs,
                        FiniteVolumeMetadata::new(4).with_reconstruction_order(2),
                        None,
                        &policy,
                        schemes.clone(),
                        PlanEpochs::default(),
                    )
                    .unwrap(),
                );
            });
        });
        let operator = DeviceFvmOperator::compile(
            &backend,
            &inputs,
            FiniteVolumeMetadata::new(4).with_reconstruction_order(2),
            None,
            &policy,
            schemes,
            PlanEpochs::default(),
        )
        .unwrap();
        let cells = vec![1.0_f64; operator.plan.cell_ids.len() * 4];
        let mass = vec![0.1_f64; operator.plan.face_count()];
        let boundary_values = vec![0.0_f64; operator.plan.boundary_face_count * 4];
        group.bench_with_input(
            BenchmarkId::new("state_upload", internal),
            &internal,
            |b, _| {
                b.iter(|| {
                    black_box(
                        DeviceFvmState::upload_components(
                            &backend,
                            &operator.plan,
                            4,
                            &cells,
                            &mass,
                            &boundary_values,
                        )
                        .unwrap(),
                    );
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_cuda_resident_operator(c: &mut Criterion) {
    let Ok(backend) = CudaBackend::new(CudaOptions::default()) else {
        eprintln!("CUDA unavailable; skipping CUDA FVM benchmarks");
        return;
    };
    let mut group = c.benchmark_group("fvm_resident_operator_cuda");
    for &(internal, boundary, components) in &[
        (10_000usize, 1_000usize, 1usize),
        (10_000, 1_000, 4),
        (100_000, 10_000, 4),
    ] {
        let inputs = synthetic_inputs(internal, boundary);
        let boundary_face_branches = inputs
            .boundary_faces()
            .map(|face| (face.face, FvBoundaryBranch::Inflow))
            .collect::<HashMap<_, _>>();
        let policy = FvBoundaryPolicy {
            boundary_face_branches,
            allowed_branches: HashSet::from([FvBoundaryBranch::Inflow]),
            convective_branch_hooks: HashMap::from([(
                FvBoundaryBranch::Inflow,
                BoundaryCondition::Dirichlet { value: 0.0 },
            )]),
            diffusive_branch_hooks: HashMap::from([(
                FvBoundaryBranch::Inflow,
                BoundaryCondition::Neumann { gradient: 0.0 },
            )]),
            unsupported_behavior: UnsupportedBoundaryBehavior::Error,
        };
        let schemes = FvmSchemeSettings {
            convective: ConvectiveScheme::Upwind,
            reconstruction: ReconstructionSettings {
                mode: ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
                limiter: LimiterOption::None,
            },
            diffusion: DiffusionSettings {
                diffusivity: 0.01,
                non_orthogonal_mode: NonOrthogonalCorrectionMode::FullyCorrected,
            },
        };
        group.bench_with_input(
            BenchmarkId::new(format!("compile_{components}c"), internal),
            &internal,
            |b, _| {
                b.iter(|| {
                    black_box(
                        DeviceFvmOperator::compile(
                            &backend,
                            &inputs,
                            FiniteVolumeMetadata::new(components),
                            None,
                            &policy,
                            schemes.clone(),
                            PlanEpochs::default(),
                        )
                        .unwrap(),
                    );
                });
            },
        );
        let operator = DeviceFvmOperator::compile(
            &backend,
            &inputs,
            FiniteVolumeMetadata::new(components),
            None,
            &policy,
            schemes,
            PlanEpochs::default(),
        )
        .unwrap();
        let cell_values = vec![1.0_f64; operator.plan.cell_ids.len() * components];
        let mass_flux = vec![0.1_f64; operator.plan.face_count()];
        let boundary_values = vec![0.0_f64; operator.plan.boundary_face_count * components];
        group.bench_with_input(
            BenchmarkId::new(format!("state_upload_{components}c"), internal),
            &internal,
            |b, _| {
                b.iter(|| {
                    black_box(
                        DeviceFvmState::upload_components(
                            &backend,
                            &operator.plan,
                            components,
                            &cell_values,
                            &mass_flux,
                            &boundary_values,
                        )
                        .unwrap(),
                    );
                });
            },
        );
        let mut state = DeviceFvmState::upload_components(
            &backend,
            &operator.plan,
            components,
            &cell_values,
            &mass_flux,
            &boundary_values,
        )
        .unwrap();
        backend
            .evaluate_residual(&operator, &mut state, PlanEpochs::default())
            .unwrap();
        backend.synchronize().unwrap();
        group.throughput(Throughput::Elements(
            ((internal + boundary) * components) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new(format!("evaluate_residual_{components}c"), internal),
            &internal,
            |b, _| {
                b.iter(|| {
                    backend
                        .evaluate_residual(&operator, &mut state, PlanEpochs::default())
                        .unwrap();
                    backend.synchronize_stream(0).unwrap();
                    black_box(&state.residual);
                });
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_resident_operator(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_fvm_face_loop,
    bench_resident_fvm_operator,
    bench_fvm_operator_setup,
    bench_cuda_resident_operator
);
criterion_main!(benches);
