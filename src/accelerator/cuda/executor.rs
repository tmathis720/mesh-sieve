//! CUDA FVM executor.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::accelerator::{
    AcceleratorError, DeviceBuffer, DeviceFvmPlan, DeviceFvmState, FvmScalar, PlanEpochs,
    ScalarFluxScheme,
};

use super::buffer::CudaBuffer;
use super::context::CudaBackend;

const FVM_KERNELS: &str = include_str!("kernels/fvm_scalar.cu");

/// Scalar type/function mapping for built-in CUDA FVM kernels.
pub trait CudaFvmScalar: FvmScalar {
    /// Internal flux entry point.
    const INTERNAL_FLUX: &'static str;
    /// Boundary flux entry point.
    const BOUNDARY_FLUX: &'static str;
    /// Cell gather entry point.
    const CELL_GATHER: &'static str;
    /// Internal face interpolation entry point.
    const INTERNAL_VALUE: &'static str;
    /// Boundary face interpolation entry point.
    const BOUNDARY_VALUE: &'static str;
    /// Green--Gauss gather entry point.
    const GRADIENT_GATHER: &'static str;
    /// Internal diffusive flux entry point.
    const INTERNAL_DIFFUSION: &'static str;
    /// Boundary diffusive flux entry point.
    const BOUNDARY_DIFFUSION: &'static str;
    /// Vector fill entry point.
    const FILL: &'static str;
    /// Vector copy entry point.
    const COPY: &'static str;
    /// Vector AXPY entry point.
    const AXPY: &'static str;
    /// Mask application entry point.
    const APPLY_MASK: &'static str;
}

impl CudaFvmScalar for f32 {
    const INTERNAL_FLUX: &'static str = "internal_flux_f32";
    const BOUNDARY_FLUX: &'static str = "boundary_flux_f32";
    const CELL_GATHER: &'static str = "cell_gather_f32";
    const INTERNAL_VALUE: &'static str = "internal_value_f32";
    const BOUNDARY_VALUE: &'static str = "boundary_value_f32";
    const GRADIENT_GATHER: &'static str = "gradient_gather_f32";
    const INTERNAL_DIFFUSION: &'static str = "internal_diffusion_f32";
    const BOUNDARY_DIFFUSION: &'static str = "boundary_diffusion_f32";
    const FILL: &'static str = "fill_f32";
    const COPY: &'static str = "copy_f32";
    const AXPY: &'static str = "axpy_f32";
    const APPLY_MASK: &'static str = "apply_mask_f32";
}

impl CudaFvmScalar for f64 {
    const INTERNAL_FLUX: &'static str = "internal_flux_f64";
    const BOUNDARY_FLUX: &'static str = "boundary_flux_f64";
    const CELL_GATHER: &'static str = "cell_gather_f64";
    const INTERNAL_VALUE: &'static str = "internal_value_f64";
    const BOUNDARY_VALUE: &'static str = "boundary_value_f64";
    const GRADIENT_GATHER: &'static str = "gradient_gather_f64";
    const INTERNAL_DIFFUSION: &'static str = "internal_diffusion_f64";
    const BOUNDARY_DIFFUSION: &'static str = "boundary_diffusion_f64";
    const FILL: &'static str = "fill_f64";
    const COPY: &'static str = "copy_f64";
    const AXPY: &'static str = "axpy_f64";
    const APPLY_MASK: &'static str = "apply_mask_f64";
}

impl CudaBackend {
    /// Fill a resident vector without allocating or staging through the host.
    pub fn fill<T: CudaFvmScalar>(
        &self,
        output: &mut CudaBuffer<T>,
        value: T,
    ) -> Result<(), AcceleratorError> {
        if output.is_empty() {
            return Ok(());
        }
        let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::FILL)?;
        let count = output.len() as u32;
        let mut launch = self.stream().launch_builder(&function);
        launch.arg(&mut output.inner).arg(&value).arg(&count);
        // SAFETY: the kernel bounds-checks against the allocation length.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
            .map(|_| ())
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))
    }

    /// Copy equally sized resident vectors.
    pub fn copy<T: CudaFvmScalar>(
        &self,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        if input.len() != output.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: input.len(),
                found: output.len(),
            });
        }
        if input.is_empty() {
            return Ok(());
        }
        let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::COPY)?;
        let count = input.len() as u32;
        let mut launch = self.stream().launch_builder(&function);
        launch.arg(&input.inner).arg(&mut output.inner).arg(&count);
        // SAFETY: both buffers have exactly `count` elements.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
            .map(|_| ())
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))
    }

    /// Compute `y = alpha * x + y` in resident memory.
    pub fn axpy<T: CudaFvmScalar>(
        &self,
        alpha: T,
        x: &CudaBuffer<T>,
        y: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        if x.len() != y.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: x.len(),
                found: y.len(),
            });
        }
        if x.is_empty() {
            return Ok(());
        }
        let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::AXPY)?;
        let count = x.len() as u32;
        let mut launch = self.stream().launch_builder(&function);
        launch
            .arg(&x.inner)
            .arg(&mut y.inner)
            .arg(&alpha)
            .arg(&count);
        // SAFETY: both buffers have exactly `count` elements.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
            .map(|_| ())
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))
    }

    /// Zero resident values wherever the corresponding byte mask is zero.
    pub fn apply_mask<T: CudaFvmScalar>(
        &self,
        mask: &CudaBuffer<u8>,
        values: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        if mask.len() != values.len() {
            return Err(AcceleratorError::LengthMismatch {
                expected: mask.len(),
                found: values.len(),
            });
        }
        if mask.is_empty() {
            return Ok(());
        }
        let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::APPLY_MASK)?;
        let count = mask.len() as u32;
        let mut launch = self.stream().launch_builder(&function);
        launch.arg(&mask.inner).arg(&mut values.inner).arg(&count);
        // SAFETY: mask and values have exactly `count` elements.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
            .map(|_| ())
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))
    }

    /// Compute Green--Gauss cell gradients using face interpolation followed
    /// by deterministic CSR gathering.
    pub fn compute_gradients<T: CudaFvmScalar>(
        &self,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        if plan.internal_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::INTERNAL_VALUE)?;
            let count = plan.internal_face_count as u32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(&plan.internal_owner.inner)
                .arg(&plan.internal_neighbor.inner)
                .arg(&state.cell_values.inner)
                .arg(&mut state.face_values.inner)
                .arg(&count);
            // SAFETY: plan compilation validates every dense cell index.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        if plan.boundary_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::BOUNDARY_VALUE)?;
            let count = plan.boundary_face_count as u32;
            let offset = plan.internal_face_count as u32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(&state.boundary_values.inner)
                .arg(&mut state.face_values.inner)
                .arg(&offset)
                .arg(&count);
            // SAFETY: the boundary vector and output tail have `count` values.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        if !plan.cell_ids.is_empty() {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::GRADIENT_GATHER)?;
            let count = plan.cell_ids.len() as u32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(&plan.cell_face_offsets.inner)
                .arg(&plan.cell_face_indices.inner)
                .arg(&plan.cell_face_signs.inner)
                .arg(&plan.face_geometry_indices.inner)
                .arg(&plan.face_normal_x.inner)
                .arg(&plan.face_normal_y.inner)
                .arg(&plan.face_normal_z.inner)
                .arg(&plan.cell_volume.inner)
                .arg(&plan.face_active.inner)
                .arg(&plan.cell_active.inner)
                .arg(&state.face_values.inner)
                .arg(&mut state.gradient_x.inner)
                .arg(&mut state.gradient_y.inner)
                .arg(&mut state.gradient_z.inner)
                .arg(&count);
            // SAFETY: all CSR and geometry indices were checked during compile.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Compute orthogonal internal and Dirichlet boundary diffusive fluxes.
    pub fn compute_diffusive_face_fluxes<T: CudaFvmScalar>(
        &self,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        diffusivity: f64,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        let diffusivity = T::from_f64(diffusivity);
        if plan.internal_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::INTERNAL_DIFFUSION)?;
            let count = plan.internal_face_count as u32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(&plan.internal_owner.inner)
                .arg(&plan.internal_neighbor.inner)
                .arg(&plan.internal_geometry.inner)
                .arg(&plan.face_active.inner)
                .arg(&plan.face_normal_x.inner)
                .arg(&plan.face_normal_y.inner)
                .arg(&plan.face_normal_z.inner)
                .arg(&plan.cell_center_x.inner)
                .arg(&plan.cell_center_y.inner)
                .arg(&plan.cell_center_z.inner)
                .arg(&state.cell_values.inner)
                .arg(&mut state.face_flux.inner)
                .arg(&diffusivity)
                .arg(&count);
            // SAFETY: plan compilation validates cell and geometry indices.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        if plan.boundary_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::BOUNDARY_DIFFUSION)?;
            let count = plan.boundary_face_count as u32;
            let offset = plan.internal_face_count as u32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(&plan.boundary_owner.inner)
                .arg(&plan.boundary_geometry.inner)
                .arg(&plan.face_active.inner)
                .arg(&plan.face_normal_x.inner)
                .arg(&plan.face_normal_y.inner)
                .arg(&plan.face_normal_z.inner)
                .arg(&plan.face_center_x.inner)
                .arg(&plan.face_center_y.inner)
                .arg(&plan.face_center_z.inner)
                .arg(&plan.cell_center_x.inner)
                .arg(&plan.cell_center_y.inner)
                .arg(&plan.cell_center_z.inner)
                .arg(&state.cell_values.inner)
                .arg(&state.boundary_values.inner)
                .arg(&mut state.face_flux.inner)
                .arg(&diffusivity)
                .arg(&offset)
                .arg(&count);
            // SAFETY: boundary state and geometry arrays match the plan counts.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Launch distinct internal and boundary scalar face-flux kernels.
    pub fn compute_face_fluxes<T: CudaFvmScalar>(
        &self,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        scheme: ScalarFluxScheme,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        let scheme = scheme as u32;
        if plan.internal_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::INTERNAL_FLUX)?;
            let count = plan.internal_face_count as u32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(&plan.internal_owner.inner)
                .arg(&plan.internal_neighbor.inner)
                .arg(&plan.face_active.inner)
                .arg(&state.cell_values.inner)
                .arg(&state.face_mass_flux.inner)
                .arg(&mut state.face_flux.inner)
                .arg(&scheme)
                .arg(&count);
            // SAFETY: every buffer length and dense index is validated while
            // compiling the plan/state; the kernel checks its thread index.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        if plan.boundary_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::BOUNDARY_FLUX)?;
            let count = plan.boundary_face_count as u32;
            let offset = plan.internal_face_count as u32;
            let mut launch = self.stream().launch_builder(&function);
            launch
                .arg(&plan.boundary_owner.inner)
                .arg(&plan.face_active.inner)
                .arg(&state.cell_values.inner)
                .arg(&state.face_mass_flux.inner)
                .arg(&state.boundary_values.inner)
                .arg(&mut state.face_flux.inner)
                .arg(&scheme)
                .arg(&offset)
                .arg(&count);
            // SAFETY: see the internal launch; boundary values and output
            // offsets are validated by DeviceFvmState::upload.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        Ok(())
    }

    /// Gather signed incident face fluxes without floating-point atomics.
    pub fn assemble_cell_residuals<T: CudaFvmScalar>(
        &self,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        if plan.cell_ids.is_empty() {
            return Ok(());
        }
        let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::CELL_GATHER)?;
        let count = plan.cell_ids.len() as u32;
        let mut launch = self.stream().launch_builder(&function);
        launch
            .arg(&plan.cell_face_offsets.inner)
            .arg(&plan.cell_face_indices.inner)
            .arg(&plan.cell_face_signs.inner)
            .arg(&plan.cell_active.inner)
            .arg(&state.face_flux.inner)
            .arg(&mut state.residual.inner)
            .arg(&count);
        // SAFETY: CSR offsets are built monotonically from in-range face
        // indices, and all buffers retain their plan-defined sizes.
        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
            .map(|_| ())
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))
    }
}
