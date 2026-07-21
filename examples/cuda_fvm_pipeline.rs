//! End-to-end CUDA finite-volume pipeline.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --features cuda --example cuda_fvm_pipeline
//! ```
//!
//! The CUDA feature uses dynamic loading, so compiling this example does not
//! require a CUDA toolkit. Running it requires an NVIDIA driver and the CUDA 12
//! NVRTC shared library.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!(
        "This example requires CUDA support. Re-run with:\n\
         cargo run --release --features cuda --example cuda_fvm_pipeline"
    );
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuda_example::run()
}

#[cfg(feature = "cuda")]
mod cuda_example {
    use mesh_sieve::accelerator::cuda::{CudaBackend, CudaBuffer, CudaOptions};
    use mesh_sieve::accelerator::{
        AcceleratorBackend, AcceleratorError, CpuBackend, DeviceBuffer, DeviceFvmPlan,
        DeviceFvmState, PlanEpochs, ScalarFluxScheme,
    };
    use mesh_sieve::discretization::runtime::{CellGeometry, FaceGeometry, FluxStencil};
    use mesh_sieve::physics::fvm::FvmInputs;
    use mesh_sieve::topology::coastal::{WET_DRY_MASK_LABEL, WetDryMask};
    use mesh_sieve::topology::labels::LabelSet;
    use mesh_sieve::topology::point::PointId;

    const EPSILON: f64 = 1.0e-12;

    #[derive(Debug)]
    struct ReferenceOutput {
        gradient_x: Vec<f64>,
        convective_residual: Vec<f64>,
        diffusive_residual: Vec<f64>,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let inputs = three_cell_problem();
        let epochs = PlanEpochs {
            topology: 7,
            atlas: 3,
            geometry: 11,
        };
        let cell_values = [1.0_f64, 2.0, 4.0];
        // DeviceFvmPlan orders internal faces first, then boundary faces.
        let face_mass_flux = [0.5_f64, -0.25, -1.0, 0.75];
        let boundary_values = [10.0_f64, 20.0];
        let diffusivity = 0.125;

        let reference = cpu_reference(
            &inputs,
            &cell_values,
            &face_mass_flux,
            &boundary_values,
            ScalarFluxScheme::Upwind,
            diffusivity,
            epochs,
        )?;

        println!("Initializing CUDA device 0 and probing NVRTC ...");
        let backend = CudaBackend::new(CudaOptions {
            enable_profiling: true,
            ..CudaOptions::default()
        })?;
        println!(
            "CUDA ready: device={}, streams={}, profiling={}",
            backend.options().device_ordinal,
            backend.options().stream_count,
            backend.options().enable_profiling
        );

        // Compile geometry, connectivity, masks, and deterministic cell-face
        // CSR once. Only scalar state changes in the iterations below.
        let plan = DeviceFvmPlan::compile(&backend, &inputs, None, epochs)?;
        println!(
            "Plan uploaded: {} cells, {} internal faces, {} boundary faces, {}D",
            plan.cell_ids.len(),
            plan.internal_face_count,
            plan.boundary_face_count,
            plan.dimension
        );
        let mut state = DeviceFvmState::upload(
            &backend,
            &plan,
            &cell_values,
            &face_mass_flux,
            &boundary_values,
        )?;

        backend.compute_gradients(&plan, &mut state, epochs)?;
        backend.synchronize()?;
        let gradient_x = download_f64(&backend, &state.gradient_x)?;
        assert_close(
            "Green--Gauss gradient",
            &gradient_x,
            &reference.gradient_x,
            EPSILON,
        );
        println!("Green--Gauss gradient x: {gradient_x:?}");

        backend.compute_face_fluxes(&plan, &mut state, ScalarFluxScheme::Upwind, epochs)?;
        backend.assemble_cell_residuals(&plan, &mut state, epochs)?;
        backend.synchronize()?;
        let convective_residual = state.download_residual(&backend)?;
        assert_close(
            "upwind residual",
            &convective_residual,
            &reference.convective_residual,
            EPSILON,
        );
        println!("Upwind convective residual: {convective_residual:?}");

        backend.compute_diffusive_face_fluxes(&plan, &mut state, diffusivity, epochs)?;
        backend.assemble_cell_residuals(&plan, &mut state, epochs)?;
        backend.synchronize()?;
        let diffusive_residual = state.download_residual(&backend)?;
        assert_close(
            "diffusive residual",
            &diffusive_residual,
            &reference.diffusive_residual,
            EPSILON,
        );
        println!("Orthogonal diffusive residual: {diffusive_residual:?}");

        // A second iteration refreshes existing allocations instead of
        // recompiling topology/geometry or reallocating the state.
        let next_cell_values = [1.5_f64, 2.5, 3.5];
        let next_mass_flux = [0.25_f64, 0.5, -0.5, 1.0];
        backend.upload_into(&next_cell_values, &mut state.cell_values)?;
        backend.upload_into(&next_mass_flux, &mut state.face_mass_flux)?;
        backend.compute_face_fluxes(&plan, &mut state, ScalarFluxScheme::Central, epochs)?;
        backend.assemble_cell_residuals(&plan, &mut state, epochs)?;
        backend.synchronize()?;
        let second_residual = state.download_residual(&backend)?;
        let expected_second = cpu_convection(
            &inputs,
            None,
            &next_cell_values,
            &next_mass_flux,
            &boundary_values,
            ScalarFluxScheme::Central,
            epochs,
        )?;
        assert_close(
            "second-iteration central residual",
            &second_residual,
            &expected_second,
            EPSILON,
        );
        println!("Reused plan/state, central residual: {second_residual:?}");

        demonstrate_f32(&backend, &plan, epochs, &inputs)?;
        demonstrate_wet_dry_mask(&backend, &inputs, epochs)?;
        demonstrate_resident_vector_ops(&backend)?;

        // Epochs make an invalidated geometry/topology plan fail before a
        // kernel is launched rather than silently using stale data.
        let stale_epochs = PlanEpochs {
            geometry: epochs.geometry + 1,
            ..epochs
        };
        let stale_error = backend
            .compute_face_fluxes(&plan, &mut state, ScalarFluxScheme::Upwind, stale_epochs)
            .expect_err("changed geometry epoch must invalidate the plan");
        assert!(matches!(
            stale_error,
            AcceleratorError::StaleGeometryPlan { .. }
        ));
        println!("Stale-plan guard: {stale_error}");

        println!("All CUDA results match the CPU packed-plan reference.");
        Ok(())
    }

    fn demonstrate_f32(
        backend: &CudaBackend,
        plan: &DeviceFvmPlan<CudaBackend>,
        epochs: PlanEpochs,
        inputs: &FvmInputs,
    ) -> Result<(), AcceleratorError> {
        let cell_values = [1.0_f32, 2.0, 4.0];
        let face_mass_flux = [0.5_f32, -0.25, -1.0, 0.75];
        let boundary_values = [10.0_f32, 20.0];
        let mut state = DeviceFvmState::upload(
            backend,
            plan,
            &cell_values,
            &face_mass_flux,
            &boundary_values,
        )?;
        backend.compute_face_fluxes(plan, &mut state, ScalarFluxScheme::Upwind, epochs)?;
        backend.assemble_cell_residuals(plan, &mut state, epochs)?;
        backend.synchronize()?;
        let actual = state.download_residual(backend)?;
        let expected = cpu_convection(
            inputs,
            None,
            &[1.0_f64, 2.0, 4.0],
            &[0.5_f64, -0.25, -1.0, 0.75],
            &[10.0_f64, 20.0],
            ScalarFluxScheme::Upwind,
            epochs,
        )?;
        let expected: Vec<f32> = expected.into_iter().map(|value| value as f32).collect();
        assert_eq!(actual, expected);
        println!("The same plan also executes f32 state: {actual:?}");
        Ok(())
    }

    fn demonstrate_wet_dry_mask(
        backend: &CudaBackend,
        inputs: &FvmInputs,
        epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        let mut labels = LabelSet::new();
        labels.set_label(point(2), WET_DRY_MASK_LABEL, WetDryMask::Dry.code());
        let plan = DeviceFvmPlan::compile(backend, inputs, Some(&labels), epochs)?;
        let cell_values = [1.0_f64, 2.0, 4.0];
        let face_mass_flux = [0.5_f64, -0.25, -1.0, 0.75];
        let boundary_values = [10.0_f64, 20.0];
        let mut state = DeviceFvmState::upload(
            backend,
            &plan,
            &cell_values,
            &face_mass_flux,
            &boundary_values,
        )?;
        backend.compute_face_fluxes(&plan, &mut state, ScalarFluxScheme::Upwind, epochs)?;
        backend.assemble_cell_residuals(&plan, &mut state, epochs)?;
        backend.synchronize()?;
        let actual = state.download_residual(backend)?;
        let expected = cpu_convection(
            inputs,
            Some(&labels),
            &cell_values,
            &face_mass_flux,
            &boundary_values,
            ScalarFluxScheme::Upwind,
            epochs,
        )?;
        assert_close("wet/dry residual", &actual, &expected, EPSILON);
        println!("Dry middle cell masks adjacent faces: {actual:?}");
        Ok(())
    }

    fn demonstrate_resident_vector_ops(backend: &CudaBackend) -> Result<(), AcceleratorError> {
        let x: CudaBuffer<f64> = backend.upload(&[1.0, 2.0, 3.0])?;
        let mask: CudaBuffer<u8> = backend.upload(&[1, 0, 1])?;
        let mut y: CudaBuffer<f64> = backend.allocate(3)?;
        let mut result: CudaBuffer<f64> = backend.allocate(3)?;
        backend.fill(&mut y, 2.0)?;
        backend.axpy(0.5, &x, &mut y)?;
        backend.apply_mask(&mask, &mut y)?;
        backend.copy(&y, &mut result)?;
        backend.synchronize()?;
        let actual = download_f64(backend, &result)?;
        assert_eq!(actual, vec![2.5, 0.0, 3.5]);
        println!("Resident fill/AXPY/mask/copy result: {actual:?}");
        Ok(())
    }

    fn cpu_reference(
        inputs: &FvmInputs,
        cell_values: &[f64],
        face_mass_flux: &[f64],
        boundary_values: &[f64],
        scheme: ScalarFluxScheme,
        diffusivity: f64,
        epochs: PlanEpochs,
    ) -> Result<ReferenceOutput, AcceleratorError> {
        let backend = CpuBackend;
        let plan = DeviceFvmPlan::compile(&backend, inputs, None, epochs)?;
        let mut state = DeviceFvmState::upload(
            &backend,
            &plan,
            cell_values,
            face_mass_flux,
            boundary_values,
        )?;
        plan.compute_gradients(&mut state)?;
        let gradient_x = state.gradient_x.as_slice().to_vec();
        plan.compute_face_fluxes(&mut state, scheme)?;
        plan.assemble_cell_residuals(&mut state)?;
        let convective_residual = state.download_residual(&backend)?;
        plan.compute_diffusive_face_fluxes(&mut state, diffusivity)?;
        plan.assemble_cell_residuals(&mut state)?;
        let diffusive_residual = state.download_residual(&backend)?;
        Ok(ReferenceOutput {
            gradient_x,
            convective_residual,
            diffusive_residual,
        })
    }

    fn cpu_convection(
        inputs: &FvmInputs,
        labels: Option<&LabelSet>,
        cell_values: &[f64],
        face_mass_flux: &[f64],
        boundary_values: &[f64],
        scheme: ScalarFluxScheme,
        epochs: PlanEpochs,
    ) -> Result<Vec<f64>, AcceleratorError> {
        let backend = CpuBackend;
        let plan = DeviceFvmPlan::compile(&backend, inputs, labels, epochs)?;
        let mut state = DeviceFvmState::upload(
            &backend,
            &plan,
            cell_values,
            face_mass_flux,
            boundary_values,
        )?;
        plan.compute_face_fluxes(&mut state, scheme)?;
        plan.assemble_cell_residuals(&mut state)?;
        state.download_residual(&backend)
    }

    fn download_f64(
        backend: &CudaBackend,
        buffer: &CudaBuffer<f64>,
    ) -> Result<Vec<f64>, AcceleratorError> {
        let mut host = vec![0.0; buffer.len()];
        backend.download(buffer, &mut host)?;
        Ok(host)
    }

    fn assert_close(name: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{name}: vector length differs"
        );
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "{name}[{index}]: expected {expected}, found {actual}"
            );
        }
    }

    fn point(raw: u64) -> PointId {
        PointId::new(raw).expect("example point IDs are non-zero")
    }

    fn three_cell_problem() -> FvmInputs {
        let c0 = point(1);
        let c1 = point(2);
        let c2 = point(3);
        let f01 = point(10);
        let f12 = point(11);
        let left = point(12);
        let right = point(13);

        FvmInputs::new(
            [
                FluxStencil {
                    face: f01,
                    left: c0,
                    right: Some(c1),
                },
                FluxStencil {
                    face: f12,
                    left: c1,
                    right: Some(c2),
                },
                FluxStencil {
                    face: left,
                    left: c0,
                    right: None,
                },
                FluxStencil {
                    face: right,
                    left: c2,
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
                (
                    c2,
                    CellGeometry {
                        centroid: vec![2.0],
                        volume: 1.0,
                    },
                ),
            ],
            vec![
                face_geometry(f01, 0.5, 1.0, vec![c0, c1]),
                face_geometry(f12, 1.5, 1.0, vec![c1, c2]),
                face_geometry(left, -0.5, -1.0, vec![c0]),
                face_geometry(right, 2.5, 1.0, vec![c2]),
            ],
        )
    }

    fn face_geometry(
        face: PointId,
        center: f64,
        normal: f64,
        neighbors: Vec<PointId>,
    ) -> (PointId, FaceGeometry) {
        (
            face,
            FaceGeometry {
                face,
                centroid: vec![center],
                normal: vec![normal],
                area: 1.0,
                neighbors,
            },
        )
    }
}
