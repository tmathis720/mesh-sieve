#![cfg(feature = "cuda")]

use mesh_sieve::accelerator::cuda::{CudaBackend, CudaOptions};
use mesh_sieve::accelerator::{
    AcceleratorBackend, DeviceFvmPlan, DeviceFvmState, PlanEpochs, ScalarFluxScheme,
};
use mesh_sieve::discretization::runtime::{CellGeometry, FaceGeometry, FluxStencil};
use mesh_sieve::physics::fvm::FvmInputs;
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
    let Ok(backend) = CudaBackend::new(CudaOptions::default()) else {
        eprintln!("CUDA runtime unavailable; execution smoke test skipped");
        return;
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
