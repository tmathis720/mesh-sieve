use std::collections::{HashMap, HashSet};

use mesh_sieve::accelerator::{
    AcceleratorBackend, AcceleratorError, CpuBackend, DeviceCsrMatrix, DeviceFvmOperator,
    DeviceFvmState, DeviceReduction, PlanEpochs,
};
use mesh_sieve::algs::assembly::GlobalCsrPattern;
use mesh_sieve::discretization::runtime::{
    CellGeometry, ClosureDof, CsrPattern, FaceGeometry, FiniteVolumeMetadata, FluxStencil,
};
use mesh_sieve::physics::fvm::{
    BoundaryCondition, ConvectiveScheme, DiffusionSettings, FvBoundaryBranch, FvBoundaryPolicy,
    FvmInputs, FvmSchemeSettings, LimiterOption, NonOrthogonalCorrectionMode,
    ReconstructionGradient, ReconstructionMode, ReconstructionSettings, SlopeLimiterFamily,
    UnsupportedBoundaryBehavior,
};
use mesh_sieve::topology::coastal::{
    BOUNDARY_CLASS_LABEL, BOUNDARY_ROLE_LABEL, BoundaryClass, OpenBoundaryRole, WET_DRY_MASK_LABEL,
    WetDryMask,
};
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;

fn point(raw: u64) -> PointId {
    PointId::new(raw).unwrap()
}

fn inputs() -> FvmInputs {
    let c0 = point(1);
    let c1 = point(2);
    let internal = point(10);
    let left = point(11);
    let right = point(12);
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
                    centroid: vec![0.0, 0.0],
                    volume: 1.0,
                },
            ),
            (
                c1,
                CellGeometry {
                    centroid: vec![1.0, 0.0],
                    volume: 1.0,
                },
            ),
        ],
        vec![
            (
                internal,
                FaceGeometry {
                    face: internal,
                    centroid: vec![0.5, 0.0],
                    normal: vec![1.0, 0.0],
                    area: 1.0,
                    neighbors: vec![c0, c1],
                },
            ),
            (
                left,
                FaceGeometry {
                    face: left,
                    centroid: vec![-0.5, 0.0],
                    normal: vec![-1.0, 0.0],
                    area: 1.0,
                    neighbors: vec![c0],
                },
            ),
            (
                right,
                FaceGeometry {
                    face: right,
                    centroid: vec![1.5, 0.0],
                    normal: vec![1.0, 0.0],
                    area: 1.0,
                    neighbors: vec![c1],
                },
            ),
        ],
    )
}

fn inputs_with_dimension(dimension: usize) -> FvmInputs {
    let mut result = inputs();
    for (_, geometry) in &mut result.cell_geometry {
        geometry.centroid.resize(dimension, 0.0);
        geometry.centroid.truncate(dimension);
    }
    for (_, geometry) in &mut result.face_geometry {
        geometry.centroid.resize(dimension, 0.0);
        geometry.centroid.truncate(dimension);
        geometry.normal.resize(dimension, 0.0);
        geometry.normal.truncate(dimension);
    }
    result
}

fn policy(left: BoundaryCondition, right: BoundaryCondition) -> FvBoundaryPolicy {
    FvBoundaryPolicy {
        boundary_face_branches: HashMap::from([
            (point(11), FvBoundaryBranch::Inflow),
            (point(12), FvBoundaryBranch::Outflow),
        ]),
        allowed_branches: HashSet::from([FvBoundaryBranch::Inflow, FvBoundaryBranch::Outflow]),
        convective_branch_hooks: HashMap::from([
            (FvBoundaryBranch::Inflow, left),
            (FvBoundaryBranch::Outflow, right),
        ]),
        diffusive_branch_hooks: HashMap::from([
            (FvBoundaryBranch::Inflow, left),
            (FvBoundaryBranch::Outflow, right),
        ]),
        unsupported_behavior: UnsupportedBoundaryBehavior::Error,
    }
}

fn schemes(
    convective: ConvectiveScheme,
    mode: ReconstructionMode,
    limiter: LimiterOption,
    diffusion_mode: NonOrthogonalCorrectionMode,
    diffusivity: f64,
) -> FvmSchemeSettings {
    FvmSchemeSettings {
        convective,
        reconstruction: ReconstructionSettings { mode, limiter },
        diffusion: DiffusionSettings {
            diffusivity,
            non_orthogonal_mode: diffusion_mode,
        },
    }
}

#[test]
fn multicomponent_operator_is_component_major_and_conservative() {
    let backend = CpuBackend;
    let operator = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(2),
        None,
        &policy(
            BoundaryCondition::Dirichlet { value: 10.0 },
            BoundaryCondition::Dirichlet { value: 20.0 },
        ),
        schemes(
            ConvectiveScheme::Upwind,
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
            LimiterOption::None,
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            0.0,
        ),
        PlanEpochs::default(),
    )
    .unwrap();
    let mut state = DeviceFvmState::upload_components(
        &backend,
        operator.plan(),
        2,
        &[3.0_f64, 5.0, 30.0, 50.0],
        &[2.0, -1.0, 0.5],
        &[10.0, 20.0, 100.0, 200.0],
    )
    .unwrap();
    operator
        .evaluate_residual(&mut state, PlanEpochs::default())
        .unwrap();
    assert_eq!(
        state.download_residual(&backend).unwrap(),
        vec![-4.0, -3.5, 50.0, -35.0]
    );
    state
        .upload_boundary_overrides(&backend, &[10.0, 20.0, 100.0, 200.0])
        .unwrap();
    operator
        .evaluate_residual(&mut state, PlanEpochs::default())
        .unwrap();
    assert_eq!(
        state.download_residual(&backend).unwrap(),
        vec![-4.0, -3.5, -40.0, -35.0]
    );
}

#[test]
fn explicit_face_and_cell_sources_are_gathered_without_component_bleed() {
    let backend = CpuBackend;
    let operator = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(2),
        None,
        &policy(
            BoundaryCondition::Neumann { gradient: 0.0 },
            BoundaryCondition::Neumann { gradient: 0.0 },
        ),
        schemes(
            ConvectiveScheme::Central,
            ReconstructionMode::LeastSquaresWithGreenGaussFallback,
            LimiterOption::Family(SlopeLimiterFamily::Minmod),
            NonOrthogonalCorrectionMode::Deferred,
            0.0,
        ),
        PlanEpochs::default(),
    )
    .unwrap();
    let mut state = DeviceFvmState::upload_components(
        &backend,
        operator.plan(),
        2,
        &[1.0_f64, 2.0, 10.0, 20.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 2.0, 10.0, 20.0],
    )
    .unwrap();
    state
        .upload_sources(
            &backend,
            &[2.0, 3.0, 4.0, 20.0, 30.0, 40.0],
            &[10.0, 20.0, 100.0, 200.0],
        )
        .unwrap();
    operator
        .evaluate_residual(&mut state, PlanEpochs::default())
        .unwrap();
    assert_eq!(
        state.download_residual(&backend).unwrap(),
        vec![15.0, 22.0, 150.0, 220.0]
    );
}

#[test]
fn schemes_limiters_diffusion_and_boundary_variants_are_deterministic() {
    let backend = CpuBackend;
    let convection = [
        ConvectiveScheme::Upwind,
        ConvectiveScheme::Central,
        ConvectiveScheme::BoundedLinear { blend: 0.4 },
        ConvectiveScheme::BlendUpwindCentral { blend: 0.6 },
        ConvectiveScheme::HighResolution {
            blend: 0.8,
            limiter: SlopeLimiterFamily::Superbee,
        },
    ];
    let boundaries = [
        BoundaryCondition::Dirichlet { value: 2.0 },
        BoundaryCondition::Neumann { gradient: 0.25 },
        BoundaryCondition::Robin {
            alpha: 2.0,
            beta: 0.5,
            gamma: 3.0,
        },
    ];
    for scheme in convection {
        for boundary in boundaries {
            for mode in [
                NonOrthogonalCorrectionMode::OrthogonalOnly,
                NonOrthogonalCorrectionMode::Deferred,
                NonOrthogonalCorrectionMode::FullyCorrected,
            ] {
                let operator = DeviceFvmOperator::compile(
                    &backend,
                    &inputs(),
                    FiniteVolumeMetadata::new(1).with_reconstruction_order(2),
                    None,
                    &policy(boundary, boundary),
                    schemes(
                        scheme,
                        ReconstructionMode::LeastSquaresWithGreenGaussFallback,
                        LimiterOption::Family(SlopeLimiterFamily::VanLeer),
                        mode,
                        0.2,
                    ),
                    PlanEpochs::default(),
                )
                .unwrap();
                let run = || {
                    let mut state = DeviceFvmState::upload(
                        &backend,
                        operator.plan(),
                        &[1.0_f64, 3.0],
                        &[0.5, -0.25, 0.75],
                        &[2.0, 4.0],
                    )
                    .unwrap();
                    operator
                        .evaluate_residual(&mut state, PlanEpochs::default())
                        .unwrap();
                    state.download_residual(&backend).unwrap()
                };
                let first = run();
                let second = run();
                assert_eq!(first, second);
                assert!(first.iter().all(|value| value.is_finite()));
            }
        }
    }
}

#[test]
fn singular_least_squares_requires_or_uses_fallback() {
    let backend = CpuBackend;
    let boundary = policy(
        BoundaryCondition::Dirichlet { value: 0.0 },
        BoundaryCondition::Dirichlet { value: 0.0 },
    );
    let strict = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(1).with_reconstruction_order(2),
        None,
        &boundary,
        schemes(
            ConvectiveScheme::Upwind,
            ReconstructionMode::GradientOnly(ReconstructionGradient::LeastSquares),
            LimiterOption::None,
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            0.0,
        ),
        PlanEpochs::default(),
    );
    assert!(matches!(strict, Err(AcceleratorError::InvalidPlan(_))));

    DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(1).with_reconstruction_order(2),
        None,
        &boundary,
        schemes(
            ConvectiveScheme::Upwind,
            ReconstructionMode::LeastSquaresWithGreenGaussFallback,
            LimiterOption::None,
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            0.0,
        ),
        PlanEpochs::default(),
    )
    .unwrap();
}

#[test]
fn f32_operator_supports_one_through_three_dimensions_and_stale_epochs() {
    let backend = CpuBackend;
    for dimension in 1..=3 {
        let epochs = PlanEpochs {
            topology: 4,
            atlas: 5,
            geometry: 6,
        };
        let operator = DeviceFvmOperator::compile(
            &backend,
            &inputs_with_dimension(dimension),
            FiniteVolumeMetadata::new(1).with_reconstruction_order(2),
            None,
            &policy(
                BoundaryCondition::Dirichlet { value: 1.0 },
                BoundaryCondition::Dirichlet { value: 2.0 },
            ),
            schemes(
                ConvectiveScheme::HighResolution {
                    blend: 0.7,
                    limiter: SlopeLimiterFamily::Minmod,
                },
                ReconstructionMode::LeastSquaresWithGreenGaussFallback,
                LimiterOption::Family(SlopeLimiterFamily::Superbee),
                NonOrthogonalCorrectionMode::FullyCorrected,
                0.1,
            ),
            epochs,
        )
        .unwrap();
        let mut state = DeviceFvmState::upload(
            &backend,
            operator.plan(),
            &[1.0_f32, 2.0],
            &[0.5, -0.25, 0.75],
            &[1.0, 2.0],
        )
        .unwrap();
        operator.evaluate_residual(&mut state, epochs).unwrap();
        assert!(
            state
                .download_residual(&backend)
                .unwrap()
                .iter()
                .all(|value| value.is_finite())
        );
        assert!(matches!(
            operator.evaluate_residual(
                &mut state,
                PlanEpochs {
                    geometry: 7,
                    ..epochs
                }
            ),
            Err(AcceleratorError::StaleGeometryPlan { .. })
        ));
    }
}

#[test]
fn wet_dry_mask_zeroes_every_component_of_a_dry_cell() {
    let backend = CpuBackend;
    let mut labels = LabelSet::new();
    labels.set_label(point(2), WET_DRY_MASK_LABEL, WetDryMask::Dry.code());
    let operator = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(2),
        Some(&labels),
        &policy(
            BoundaryCondition::Neumann { gradient: 0.0 },
            BoundaryCondition::Neumann { gradient: 0.0 },
        ),
        schemes(
            ConvectiveScheme::Upwind,
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
            LimiterOption::None,
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            0.0,
        ),
        PlanEpochs::default(),
    )
    .unwrap();
    let mut state = DeviceFvmState::upload_components(
        &backend,
        operator.plan(),
        2,
        &[1.0_f64, 2.0, 10.0, 20.0],
        &[3.0, 4.0, 5.0],
        &[1.0, 2.0, 10.0, 20.0],
    )
    .unwrap();
    state
        .upload_sources(&backend, &[0.0; 6], &[7.0, 8.0, 70.0, 80.0])
        .unwrap();
    operator
        .evaluate_residual(&mut state, PlanEpochs::default())
        .unwrap();
    let residual = state.download_residual(&backend).unwrap();
    assert_eq!(residual[1], 0.0);
    assert_eq!(residual[3], 0.0);
}

#[test]
fn boundary_labels_must_match_packed_policy_dispatch() {
    let backend = CpuBackend;
    let mut labels = LabelSet::new();
    labels.set_label(point(11), BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
    labels.set_label(
        point(11),
        BOUNDARY_ROLE_LABEL,
        OpenBoundaryRole::Outflow.code(),
    );
    let result = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(1),
        Some(&labels),
        &policy(
            BoundaryCondition::Neumann { gradient: 0.0 },
            BoundaryCondition::Neumann { gradient: 0.0 },
        ),
        schemes(
            ConvectiveScheme::Upwind,
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
            LimiterOption::None,
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            0.0,
        ),
        PlanEpochs::default(),
    );
    assert!(matches!(result, Err(AcceleratorError::InvalidPlan(_))));
}

#[test]
fn boundary_diffusion_matches_dirichlet_neumann_and_robin_source_semantics() {
    let backend = CpuBackend;
    let cases = [
        (BoundaryCondition::Dirichlet { value: 3.0 }, -4.0),
        (BoundaryCondition::Neumann { gradient: 0.25 }, -0.25),
        (
            BoundaryCondition::Robin {
                alpha: 2.0,
                beta: 0.5,
                gamma: 3.0,
            },
            -2.0,
        ),
    ];
    for (boundary, expected) in cases {
        let operator = DeviceFvmOperator::compile(
            &backend,
            &inputs(),
            FiniteVolumeMetadata::new(1),
            None,
            &policy(boundary, boundary),
            schemes(
                ConvectiveScheme::Upwind,
                ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
                LimiterOption::None,
                NonOrthogonalCorrectionMode::OrthogonalOnly,
                1.0,
            ),
            PlanEpochs::default(),
        )
        .unwrap();
        let mut state = DeviceFvmState::upload(
            &backend,
            operator.plan(),
            &[1.0_f64, 1.0],
            &[0.0, 0.0, 0.0],
            &[3.0, 3.0],
        )
        .unwrap();
        operator
            .evaluate_residual(&mut state, PlanEpochs::default())
            .unwrap();
        assert_eq!(state.download_residual(&backend).unwrap(), [expected; 2]);
    }
}

#[test]
fn boundary_refresh_updates_packed_dirichlet_values_without_state_overrides() {
    let backend = CpuBackend;
    let mut operator = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(1),
        None,
        &policy(
            BoundaryCondition::Dirichlet { value: 2.0 },
            BoundaryCondition::Dirichlet { value: 2.0 },
        ),
        schemes(
            ConvectiveScheme::Upwind,
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
            LimiterOption::None,
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            1.0,
        ),
        PlanEpochs::default(),
    )
    .unwrap();
    let mut state = DeviceFvmState::upload(
        &backend,
        operator.plan(),
        &[1.0_f64, 1.0],
        &[0.0, 0.0, 0.0],
        &[99.0, 99.0],
    )
    .unwrap();
    operator
        .evaluate_residual(&mut state, PlanEpochs::default())
        .unwrap();
    let before = state.download_residual(&backend).unwrap();
    operator
        .refresh_boundary_conditions(
            &backend,
            &policy(
                BoundaryCondition::Dirichlet { value: 4.0 },
                BoundaryCondition::Dirichlet { value: 4.0 },
            ),
        )
        .unwrap();
    operator
        .evaluate_residual(&mut state, PlanEpochs::default())
        .unwrap();
    let after = state.download_residual(&backend).unwrap();
    assert_ne!(before, after);
    assert_eq!(after, vec![-6.0, -6.0]);
}

#[test]
fn zero_beta_robin_diffusion_is_rejected() {
    let backend = CpuBackend;
    let robin = BoundaryCondition::Robin {
        alpha: 1.0,
        beta: 0.0,
        gamma: 2.0,
    };
    let result = DeviceFvmOperator::compile(
        &backend,
        &inputs(),
        FiniteVolumeMetadata::new(1),
        None,
        &policy(robin, robin),
        schemes(
            ConvectiveScheme::Upwind,
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss),
            LimiterOption::None,
            NonOrthogonalCorrectionMode::OrthogonalOnly,
            1.0,
        ),
        PlanEpochs::default(),
    );
    assert!(matches!(result, Err(AcceleratorError::InvalidPlan(_))));
}

#[test]
fn reductions_and_cpu_csr_spmv_validate_and_compute() {
    let backend = CpuBackend;
    let input = backend.upload(&[-1.0_f64, 2.0, -3.0]).unwrap();
    let other = backend.upload(&[-1.0_f64, 2.0, -3.0]).unwrap();
    let mut reduction = DeviceReduction::new(&backend, 3).unwrap();
    assert_eq!(reduction.sum(&input).unwrap(), -2.0);
    assert_eq!(reduction.dot(&input, &other).unwrap(), 14.0);
    assert_eq!(reduction.l2_norm(&input).unwrap(), 14.0_f64.sqrt());
    assert_eq!(reduction.max_abs(&input).unwrap(), 3.0);

    let d0 = ClosureDof {
        point: point(1),
        local_dof: 0,
    };
    let d1 = ClosureDof {
        point: point(2),
        local_dof: 0,
    };
    let pattern = CsrPattern {
        xadj: vec![0, 2, 4],
        adjncy: vec![d0, d1, d0, d1],
        rows: vec![d0, d1],
    };
    let matrix =
        DeviceCsrMatrix::from_pattern(&backend, &pattern, &[2.0_f64, 1.0, 3.0, 4.0]).unwrap();
    let x = backend.upload(&[1.0_f64, 2.0]).unwrap();
    let mut y = backend.upload(&[1.0_f64, 1.0]).unwrap();
    matrix.spmv(1.0, &x, 0.5, &mut y).unwrap();
    assert_eq!(y.as_slice(), &[4.5, 11.5]);

    let malformed = CsrPattern {
        xadj: vec![1, 1],
        adjncy: vec![],
        rows: vec![d0],
    };
    assert!(matches!(
        DeviceCsrMatrix::<f64, _>::from_pattern(&backend, &malformed, &[]),
        Err(AcceleratorError::InvalidPlan(_))
    ));

    let overflow = GlobalCsrPattern {
        xadj: vec![0, 1],
        adjncy: vec![usize::MAX],
        rows: vec![0],
    };
    assert!(matches!(
        DeviceCsrMatrix::<f64, _>::from_global_pattern(&backend, &overflow, &[1.0]),
        Err(AcceleratorError::IndexOverflow { .. })
    ));
}
