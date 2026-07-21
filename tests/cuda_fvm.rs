#![cfg(feature = "cuda")]

use std::collections::{HashMap, HashSet};

#[cfg(feature = "cuda-cusparse")]
use mesh_sieve::accelerator::DeviceCsrMatrix;
use mesh_sieve::accelerator::cuda::{CudaBackend, CudaOptions};
use mesh_sieve::accelerator::{
    AcceleratorBackend, CpuBackend, DeviceFvmOperator, DeviceFvmPlan, DeviceFvmState,
    DeviceReduction, PlanEpochs, ScalarFluxScheme,
};
use mesh_sieve::discretization::runtime::{
    CellGeometry, FaceGeometry, FiniteVolumeMetadata, FluxStencil,
};
#[cfg(feature = "cuda-cusparse")]
use mesh_sieve::discretization::runtime::{ClosureDof, CsrPattern};
use mesh_sieve::physics::fvm::{
    BoundaryCondition, ConvectiveScheme, DiffusionSettings, FvBoundaryBranch, FvBoundaryPolicy,
    FvmInputs, FvmSchemeSettings, LimiterOption, NonOrthogonalCorrectionMode,
    ReconstructionGradient, ReconstructionMode, ReconstructionSettings, SlopeLimiterFamily,
    UnsupportedBoundaryBehavior,
};
use mesh_sieve::topology::point::PointId;

fn p(raw: u64) -> PointId {
    PointId::new(raw).unwrap()
}

fn inputs() -> FvmInputs {
    let c0 = p(1);
    let c1 = p(2);
    let face = p(3);
    FvmInputs::new(
        [FluxStencil {
            face,
            left: c0,
            right: Some(c1),
        }],
        vec![
            (
                c0,
                CellGeometry {
                    centroid: vec![0.0],
                    volume: 1.0,
                },
            ),
            (
                c1,
                CellGeometry {
                    centroid: vec![1.0],
                    volume: 1.0,
                },
            ),
        ],
        vec![(
            face,
            FaceGeometry {
                face,
                centroid: vec![0.5],
                normal: vec![1.0],
                area: 1.0,
                neighbors: vec![c0, c1],
            },
        )],
    )
}

#[test]
fn cuda_fvm_f32_and_f64_match_conservative_reference() {
    let backend = match CudaBackend::new(CudaOptions::default()) {
        Ok(backend) => backend,
        Err(error) if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") => {
            panic!("CUDA tests were required but initialization failed: {error}")
        }
        Err(error) => {
            eprintln!(
                "CUDA runtime unavailable ({error}); execution smoke test skipped; set \
                 MESH_SIEVE_RUN_CUDA_TESTS=1 to require execution"
            );
            return;
        }
    };
    let epochs = PlanEpochs::default();
    let plan = DeviceFvmPlan::compile(&backend, &inputs(), None, epochs).unwrap();

    let mut state64 =
        DeviceFvmState::upload(&backend, &plan, &[3.0_f64, 5.0], &[2.0], &[]).unwrap();
    backend
        .compute_face_fluxes(&plan, &mut state64, ScalarFluxScheme::Upwind, epochs)
        .unwrap();
    backend
        .assemble_cell_residuals(&plan, &mut state64, epochs)
        .unwrap();
    backend.synchronize().unwrap();
    assert_eq!(
        state64.download_residual(&backend).unwrap(),
        vec![6.0, -6.0]
    );

    let mut state32 =
        DeviceFvmState::upload(&backend, &plan, &[3.0_f32, 5.0], &[2.0], &[]).unwrap();
    backend
        .compute_face_fluxes(&plan, &mut state32, ScalarFluxScheme::Upwind, epochs)
        .unwrap();
    backend
        .assemble_cell_residuals(&plan, &mut state32, epochs)
        .unwrap();
    backend.synchronize().unwrap();
    assert_eq!(
        state32.download_residual(&backend).unwrap(),
        vec![6.0, -6.0]
    );
}

fn boundary_inputs() -> FvmInputs {
    let c0 = p(1);
    let c1 = p(2);
    let internal = p(3);
    let left = p(4);
    let right = p(5);
    FvmInputs::new(
        [
            FluxStencil {
                face: internal,
                left: c0,
                right: Some(c1),
            },
            FluxStencil {
                face: left,
                left: c0,
                right: None,
            },
            FluxStencil {
                face: right,
                left: c1,
                right: None,
            },
        ],
        vec![
            (
                c0,
                CellGeometry {
                    centroid: vec![0.0],
                    volume: 1.0,
                },
            ),
            (
                c1,
                CellGeometry {
                    centroid: vec![1.0],
                    volume: 1.0,
                },
            ),
        ],
        vec![
            (
                internal,
                FaceGeometry {
                    face: internal,
                    centroid: vec![0.5],
                    normal: vec![1.0],
                    area: 1.0,
                    neighbors: vec![c0, c1],
                },
            ),
            (
                left,
                FaceGeometry {
                    face: left,
                    centroid: vec![-0.5],
                    normal: vec![-1.0],
                    area: 1.0,
                    neighbors: vec![c0],
                },
            ),
            (
                right,
                FaceGeometry {
                    face: right,
                    centroid: vec![1.5],
                    normal: vec![1.0],
                    area: 1.0,
                    neighbors: vec![c1],
                },
            ),
        ],
    )
}

fn boundary_policy(condition: BoundaryCondition) -> FvBoundaryPolicy {
    FvBoundaryPolicy {
        boundary_face_branches: HashMap::from([
            (p(4), FvBoundaryBranch::Inflow),
            (p(5), FvBoundaryBranch::Outflow),
        ]),
        allowed_branches: HashSet::from([FvBoundaryBranch::Inflow, FvBoundaryBranch::Outflow]),
        convective_branch_hooks: HashMap::from([
            (FvBoundaryBranch::Inflow, condition),
            (FvBoundaryBranch::Outflow, condition),
        ]),
        diffusive_branch_hooks: HashMap::from([
            (FvBoundaryBranch::Inflow, condition),
            (FvBoundaryBranch::Outflow, condition),
        ]),
        unsupported_behavior: UnsupportedBoundaryBehavior::Error,
    }
}

#[test]
fn cuda_multicomponent_operator_is_deterministic() {
    let backend = match CudaBackend::new(CudaOptions::default()) {
        Ok(backend) => backend,
        Err(error) if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") => {
            panic!("CUDA tests were required but initialization failed: {error}")
        }
        Err(error) => {
            eprintln!(
                "CUDA runtime unavailable ({error}); deterministic operator test SKIPPED; set \
                 MESH_SIEVE_RUN_CUDA_TESTS=1 to require execution"
            );
            return;
        }
    };
    let policy = FvBoundaryPolicy {
        boundary_face_branches: Default::default(),
        allowed_branches: Default::default(),
        convective_branch_hooks: Default::default(),
        diffusive_branch_hooks: Default::default(),
        unsupported_behavior: UnsupportedBoundaryBehavior::Ignore,
    };
    let schemes = FvmSchemeSettings {
        convective: ConvectiveScheme::Upwind,
        reconstruction: ReconstructionSettings {
            mode: ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
            limiter: LimiterOption::None,
        },
        diffusion: DiffusionSettings {
            diffusivity: 0.0,
            non_orthogonal_mode: NonOrthogonalCorrectionMode::OrthogonalOnly,
        },
    };
    let epochs = PlanEpochs::default();
    let operator = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(2),
        None,
        &policy,
        schemes,
        epochs,
    )
    .unwrap();
    let mut state = DeviceFvmState::upload_components(
        &backend,
        operator.plan(),
        2,
        &[3.0_f64, 5.0, 30.0, 50.0],
        &[2.0],
        &[],
    )
    .unwrap();
    backend
        .evaluate_residual(&operator, &mut state, epochs)
        .unwrap();
    backend.synchronize().unwrap();
    let first = state.download_residual(&backend).unwrap();
    backend
        .evaluate_residual(&operator, &mut state, epochs)
        .unwrap();
    backend.synchronize().unwrap();
    let second = state.download_residual(&backend).unwrap();
    assert_eq!(first, vec![6.0, -6.0, 60.0, -60.0]);
    assert_eq!(first, second);
    if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") {
        for _ in 0..10_000 {
            backend
                .evaluate_residual(&operator, &mut state, epochs)
                .unwrap();
        }
        backend.synchronize().unwrap();
        assert_eq!(state.download_residual(&backend).unwrap(), first);
    }
}

#[test]
fn cuda_rejects_cross_backend_resources_before_transfer_or_launch() {
    let first = match CudaBackend::new(CudaOptions::default()) {
        Ok(backend) => backend,
        Err(error) if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") => {
            panic!("CUDA tests were required but initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("CUDA runtime unavailable ({error}); ownership test SKIPPED");
            return;
        }
    };
    let second = CudaBackend::new(CudaOptions::default()).unwrap();
    let buffer = first.upload(&[1.0_f64]).unwrap();
    let mut host = [0.0];
    assert!(matches!(
        second.download(&buffer, &mut host),
        Err(mesh_sieve::accelerator::AcceleratorError::BackendMismatch { .. })
    ));

    let epochs = PlanEpochs::default();
    let plan = DeviceFvmPlan::compile(&first, &inputs(), None, epochs).unwrap();
    let state = DeviceFvmState::upload(&first, &plan, &[1.0_f64, 2.0], &[0.0], &[]).unwrap();
    assert!(matches!(
        state.download_residual(&second),
        Err(mesh_sieve::accelerator::AcceleratorError::BackendMismatch { .. })
    ));
}

#[test]
fn cuda_boundary_limiter_and_diffusion_contracts_match_cpu() {
    let cuda = match CudaBackend::new(CudaOptions::default()) {
        Ok(backend) => backend,
        Err(error) if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") => {
            panic!("CUDA tests were required but initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("CUDA runtime unavailable ({error}); numerical parity test SKIPPED");
            return;
        }
    };
    let cpu = CpuBackend;
    let epochs = PlanEpochs::default();
    let cases = [
        BoundaryCondition::Dirichlet { value: 2.0 },
        BoundaryCondition::Neumann { gradient: 0.25 },
        BoundaryCondition::Robin {
            alpha: 2.0,
            beta: 0.5,
            gamma: 3.0,
        },
    ];
    for (case_index, condition) in cases.into_iter().enumerate() {
        for limiter in [
            SlopeLimiterFamily::Minmod,
            SlopeLimiterFamily::VanLeer,
            SlopeLimiterFamily::Superbee,
        ] {
            let settings = FvmSchemeSettings {
                convective: ConvectiveScheme::HighResolution {
                    blend: 0.8,
                    limiter,
                },
                reconstruction: ReconstructionSettings {
                    mode: ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
                    limiter: LimiterOption::None,
                },
                diffusion: DiffusionSettings {
                    diffusivity: 0.2,
                    non_orthogonal_mode: NonOrthogonalCorrectionMode::Deferred,
                },
            };
            let policy = boundary_policy(condition);
            let cuda_operator = DeviceFvmOperator::compile(
                &cuda,
                &boundary_inputs(),
                FiniteVolumeMetadata::new(1).with_reconstruction_order(2),
                None,
                &policy,
                settings.clone(),
                epochs,
            )
            .unwrap();
            let cpu_operator = DeviceFvmOperator::compile(
                &cpu,
                &boundary_inputs(),
                FiniteVolumeMetadata::new(1).with_reconstruction_order(2),
                None,
                &policy,
                settings,
                epochs,
            )
            .unwrap();
            let mut cuda_state = DeviceFvmState::upload(
                &cuda,
                cuda_operator.plan(),
                &[1.0_f64, 3.0],
                &[0.5, -0.25, 0.75],
                &[99.0, 99.0],
            )
            .unwrap();
            let mut cpu_state = DeviceFvmState::upload(
                &cpu,
                cpu_operator.plan(),
                &[1.0_f64, 3.0],
                &[0.5, -0.25, 0.75],
                &[99.0, 99.0],
            )
            .unwrap();
            cuda.evaluate_residual(&cuda_operator, &mut cuda_state, epochs)
                .unwrap();
            cuda.synchronize().unwrap();
            cpu_operator
                .evaluate_residual(&mut cpu_state, epochs)
                .unwrap();
            let actual = cuda_state.download_residual(&cuda).unwrap();
            let expected = cpu_state.download_residual(&cpu).unwrap();
            for (a, e) in actual.iter().zip(&expected) {
                assert!(
                    (a - e).abs() <= 1.0e-10,
                    "case {case_index} {limiter:?}: {actual:?} != {expected:?}"
                );
            }
        }
    }
}

#[test]
fn cuda_vector_reduction_event_and_multistream_paths_execute() {
    let backend = match CudaBackend::new(CudaOptions {
        stream_count: 2,
        enable_profiling: true,
        ..Default::default()
    }) {
        Ok(backend) => backend,
        Err(error) if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") => {
            panic!("CUDA tests were required but initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("CUDA runtime unavailable ({error}); vector/event test SKIPPED");
            return;
        }
    };
    let values: Vec<f64> = (1..=10_000).map(f64::from).collect();
    let input = backend.upload_on(1, &values).unwrap();
    let mut output = backend.allocate_on::<f64>(1, values.len()).unwrap();
    let start = backend.record_event(1).unwrap();
    backend.copy_on(1, &input, &mut output).unwrap();
    backend.axpy_on(1, 2.0, &input, &mut output).unwrap();
    let end = backend.record_event(1).unwrap();
    backend.wait_event(0, &end).unwrap();
    backend.synchronize_event(&end).unwrap();
    assert!(backend.elapsed_ms(&start, &end).unwrap() >= 0.0);

    let mut reduction = DeviceReduction::new(&backend, values.len()).unwrap();
    let sum = reduction.sum_on(&backend, 1, &input).unwrap();
    assert_eq!(sum, 50_005_000.0);
    let dot = reduction.dot_on(&backend, 1, &input, &input).unwrap();
    let expected_dot = values.iter().map(|value| value * value).sum::<f64>();
    assert_eq!(dot, expected_dot);
    assert_eq!(reduction.max_abs_on(&backend, 1, &input).unwrap(), 10_000.0);
    assert!(
        (reduction.l2_norm_on(&backend, 1, &input).unwrap() - expected_dot.sqrt()).abs() < 1.0e-9
    );

    let mask_values: Vec<u8> = (0..values.len())
        .map(|index| u8::from(index % 2 == 0))
        .collect();
    let mask = backend.upload_on(1, &mask_values).unwrap();
    backend.apply_mask_on(1, &mask, &mut output).unwrap();
    let mut actual = vec![0.0; values.len()];
    backend.download_on(1, &output, &mut actual).unwrap();
    for (index, value) in actual.into_iter().enumerate() {
        let expected = if index % 2 == 0 {
            3.0 * values[index]
        } else {
            0.0
        };
        assert_eq!(value, expected);
    }
}

#[cfg(feature = "cuda-cusparse")]
#[test]
fn cuda_cusparse_spmv_executes_in_required_lane() {
    let backend = match CudaBackend::new(CudaOptions::default()) {
        Ok(backend) => backend,
        Err(error) if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") => {
            panic!("CUDA tests were required but initialization failed: {error}")
        }
        Err(error) => {
            eprintln!("CUDA runtime unavailable ({error}); cuSPARSE test SKIPPED");
            return;
        }
    };
    let rows: Vec<_> = (1..=2)
        .map(|raw| ClosureDof {
            point: p(raw),
            local_dof: 0,
        })
        .collect();
    let pattern = CsrPattern {
        xadj: vec![0, 2, 4],
        adjncy: vec![rows[0], rows[1], rows[0], rows[1]],
        rows,
    };
    let mut matrix =
        DeviceCsrMatrix::from_pattern(&backend, &pattern, &[2.0_f64, -1.0, -1.0, 2.0]).unwrap();
    let x = backend.upload(&[1.0_f64, 3.0]).unwrap();
    let mut y = backend.allocate(2).unwrap();
    matrix.spmv(&backend, 1.0, &x, 0.0, &mut y).unwrap();
    backend.synchronize().unwrap();
    let mut actual = [0.0; 2];
    backend.download(&y, &mut actual).unwrap();
    assert_eq!(actual, [-1.0, 5.0]);
}
