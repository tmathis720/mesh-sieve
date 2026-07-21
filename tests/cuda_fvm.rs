#![cfg(feature = "cuda")]

use mesh_sieve::accelerator::cuda::{CudaBackend, CudaOptions};
use mesh_sieve::accelerator::{
    AcceleratorBackend, DeviceFvmOperator, DeviceFvmPlan, DeviceFvmState, PlanEpochs,
    ScalarFluxScheme,
};
use mesh_sieve::discretization::runtime::{
    CellGeometry, FaceGeometry, FiniteVolumeMetadata, FluxStencil,
};
use mesh_sieve::physics::fvm::{
    ConvectiveScheme, DiffusionSettings, FvBoundaryPolicy, FvmInputs, FvmSchemeSettings,
    LimiterOption, NonOrthogonalCorrectionMode, ReconstructionGradient, ReconstructionMode,
    ReconstructionSettings, UnsupportedBoundaryBehavior,
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

#[test]
fn cuda_multicomponent_operator_is_deterministic() {
    let backend = match CudaBackend::new(CudaOptions::default()) {
        Ok(backend) => backend,
        Err(error) if std::env::var("MESH_SIEVE_RUN_CUDA_TESTS").ok().as_deref() == Some("1") => {
            panic!("CUDA tests were required but initialization failed: {error}")
        }
        Err(_) => return,
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
        &operator.plan,
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
}
