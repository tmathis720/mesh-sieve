use std::collections::HashMap;
use std::sync::Arc;

use mesh_sieve::accelerator::{
    AcceleratorError, CpuBackend, DeviceFvmPlan, DeviceFvmState, DeviceMeshPlan, DeviceSection,
    PlanEpochs, ScalarFluxScheme,
};
use mesh_sieve::data::CpuSection;
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::discretization::runtime::{CellGeometry, FaceGeometry, FluxStencil};
use mesh_sieve::physics::fvm::FvmInputs;
use mesh_sieve::topology::coastal::{WET_DRY_MASK_LABEL, WetDryMask};
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::FrozenSieveCsr;

fn point(raw: u64) -> PointId {
    PointId::new(raw).unwrap()
}

fn sample_inputs() -> FvmInputs {
    let c0 = point(1);
    let c1 = point(2);
    let internal = point(10);
    let left_boundary = point(11);
    let right_boundary = point(12);
    FvmInputs::new(
        [
            FluxStencil {
                face: internal,
                left: c0,
                right: Some(c1),
            },
            FluxStencil {
                face: left_boundary,
                left: c0,
                right: None,
            },
            FluxStencil {
                face: right_boundary,
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
                left_boundary,
                FaceGeometry {
                    face: left_boundary,
                    centroid: vec![-0.5],
                    normal: vec![-1.0],
                    area: 1.0,
                    neighbors: vec![c0],
                },
            ),
            (
                right_boundary,
                FaceGeometry {
                    face: right_boundary,
                    centroid: vec![1.5],
                    normal: vec![1.0],
                    area: 1.0,
                    neighbors: vec![c1],
                },
            ),
        ],
    )
}

#[test]
fn device_section_transfers_are_explicit_and_version_checked() {
    let mut atlas = Atlas::default();
    atlas.try_insert(point(1), 2).unwrap();
    atlas.try_insert(point(2), 1).unwrap();
    let mut section = CpuSection::<f64>::new(atlas);
    section.try_scatter_in_order(&[1.0, 2.0, 3.0]).unwrap();

    let backend = CpuBackend;
    let mut device = DeviceSection::upload_from(&backend, &section, 7).unwrap();
    assert_eq!(device.values.as_slice(), &[1.0, 2.0, 3.0]);

    section.try_scatter_in_order(&[4.0, 5.0, 6.0]).unwrap();
    device.refresh_values_from(&backend, &section, 7).unwrap();
    assert_eq!(device.values.as_slice(), &[4.0, 5.0, 6.0]);

    device
        .values
        .as_mut_slice()
        .copy_from_slice(&[7.0, 8.0, 9.0]);
    device.download_into(&backend, &mut section, 7).unwrap();
    assert_eq!(section.as_flat_slice(), &[7.0, 8.0, 9.0]);

    assert_eq!(
        device.validate(section.atlas().version(), 8),
        Err(AcceleratorError::StaleTopologyPlan {
            expected: 7,
            found: 8,
        })
    );

    section.try_add_point(point(3), 1).unwrap();
    assert_eq!(
        device.download_into(&backend, &mut section, 7),
        Err(AcceleratorError::StaleAtlasPlan {
            expected: device.atlas_version,
            found: section.atlas().version(),
        })
    );
}

#[test]
fn frozen_csr_compiles_without_uploading_hash_map() {
    let p0 = point(1);
    let p1 = point(2);
    let frozen = FrozenSieveCsr::<PointId, ()> {
        point_of: Arc::from([p0, p1]),
        index_of: HashMap::from([(p0, 0), (p1, 1)]),
        out_offsets: Arc::from([0, 1, 1]),
        out_dsts: Arc::from([1]),
        out_pay: Arc::from([()]),
        in_offsets: Arc::from([0, 0, 1]),
        in_srcs: Arc::from([0]),
        in_pay: Arc::from([()]),
    };
    let plan = DeviceMeshPlan::compile(&CpuBackend, &frozen, 12).unwrap();
    assert_eq!(plan.topology.point_ids.as_slice(), &[1, 2]);
    assert_eq!(plan.topology.cone_offsets.as_slice(), &[0, 1, 1]);
    assert_eq!(plan.index_of[&p1], 1);
    assert!(matches!(
        plan.validate_topology(13),
        Err(AcceleratorError::StaleTopologyPlan { .. })
    ));
}

fn run_reference<T: mesh_sieve::accelerator::FvmScalar>(
    cell: &[T],
    mass: &[T],
    boundary: &[T],
) -> Vec<T> {
    let backend = CpuBackend;
    let epochs = PlanEpochs {
        topology: 4,
        atlas: 2,
        geometry: 9,
    };
    let plan = DeviceFvmPlan::compile(&backend, &sample_inputs(), None, epochs).unwrap();
    let mut state = DeviceFvmState::upload(&backend, &plan, cell, mass, boundary).unwrap();
    plan.compute_face_fluxes(&mut state, ScalarFluxScheme::Upwind)
        .unwrap();
    plan.assemble_cell_residuals(&mut state).unwrap();
    state.download_residual(&backend).unwrap()
}

#[test]
fn cpu_reference_f64_is_conservative_and_deterministic() {
    let expected = vec![-4.0, -3.5];
    let first = run_reference(&[3.0_f64, 5.0], &[2.0, -1.0, 0.5], &[10.0, 20.0]);
    let second = run_reference(&[3.0_f64, 5.0], &[2.0, -1.0, 0.5], &[10.0, 20.0]);
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    // Internal flux cancels exactly; only boundary flux remains globally.
    assert_eq!(first.iter().sum::<f64>(), -7.5);
}

#[test]
fn cpu_reference_supports_f32_and_rejects_stale_geometry() {
    assert_eq!(
        run_reference(&[3.0_f32, 5.0], &[2.0, -1.0, 0.5], &[10.0, 20.0]),
        vec![-4.0, -3.5]
    );
    let epochs = PlanEpochs {
        topology: 1,
        atlas: 1,
        geometry: 1,
    };
    let plan = DeviceFvmPlan::compile(&CpuBackend, &sample_inputs(), None, epochs).unwrap();
    assert_eq!(
        plan.validate(PlanEpochs {
            geometry: 2,
            ..epochs
        }),
        Err(AcceleratorError::StaleGeometryPlan {
            expected: 1,
            found: 2,
        })
    );
}

#[test]
fn gradients_diffusion_and_wet_dry_masks_use_the_same_plan() {
    let backend = CpuBackend;
    let plan =
        DeviceFvmPlan::compile(&backend, &sample_inputs(), None, PlanEpochs::default()).unwrap();
    let mut state = DeviceFvmState::upload(
        &backend,
        &plan,
        &[3.0_f64, 5.0],
        &[0.0, 0.0, 0.0],
        &[10.0, 20.0],
    )
    .unwrap();
    plan.compute_gradients(&mut state).unwrap();
    assert_eq!(state.gradient_x.as_slice(), &[-6.0, 16.0]);
    plan.compute_diffusive_face_fluxes(&mut state, 1.0).unwrap();
    plan.assemble_cell_residuals(&mut state).unwrap();
    assert_eq!(
        state.download_residual(&backend).unwrap(),
        vec![-16.0, -28.0]
    );

    let mut labels = LabelSet::new();
    labels.set_label(point(2), WET_DRY_MASK_LABEL, WetDryMask::Dry.code());
    let dry_plan = DeviceFvmPlan::compile(
        &backend,
        &sample_inputs(),
        Some(&labels),
        PlanEpochs::default(),
    )
    .unwrap();
    assert_eq!(dry_plan.cell_active.as_slice(), &[1, 0]);
    assert_eq!(dry_plan.face_active.as_slice(), &[0, 1, 0]);
}
