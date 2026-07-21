//! CUDA FVM executor.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::accelerator::fvm::validate_scalar_state;
use crate::accelerator::{
    AcceleratorError, DeviceBuffer, DeviceFvmOperator, DeviceFvmPlan, DeviceFvmState,
    DeviceReduction, FvmScalar, PlanEpochs, ScalarFluxScheme,
};
use crate::physics::fvm::{
    ConvectiveScheme, LimiterOption, NonOrthogonalCorrectionMode, ReconstructionGradient,
    ReconstructionMode, SlopeLimiterFamily,
};

use super::buffer::CudaBuffer;
use super::context::CudaBackend;

const FVM_KERNELS: &str = include_str!("kernels/fvm_scalar.cu");
const REDUCTION_KERNELS: &str = include_str!("kernels/reduction.cu");
const OPERATOR_KERNELS: &str = include_str!("kernels/fvm_operator.cu");

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
    /// Sum reduction entry point.
    const REDUCE_SUM: &'static str;
    /// Dot-product reduction entry point.
    const REDUCE_DOT: &'static str;
    /// L2-norm reduction entry point.
    const REDUCE_L2: &'static str;
    /// Maximum-absolute-value reduction entry point.
    const REDUCE_MAX_ABS: &'static str;
    /// In-place square-root entry point used to finalize L2 reductions.
    const REDUCE_SQRT: &'static str;
    /// Multi-component internal interpolation.
    const OP_INTERNAL_VALUE: &'static str;
    /// Multi-component boundary interpolation.
    const OP_BOUNDARY_VALUE: &'static str;
    /// Multi-component Green--Gauss gather.
    const OP_GREEN_GAUSS: &'static str;
    /// Multi-component least-squares gather.
    const OP_LEAST_SQUARES: &'static str;
    /// Multi-component internal total flux.
    const OP_INTERNAL_FLUX: &'static str;
    /// Multi-component boundary total flux.
    const OP_BOUNDARY_FLUX: &'static str;
    /// Multi-component deterministic residual gather.
    const OP_CELL_GATHER: &'static str;
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
    const REDUCE_SUM: &'static str = "sum_f32";
    const REDUCE_DOT: &'static str = "dot_f32";
    const REDUCE_L2: &'static str = "l2_f32";
    const REDUCE_MAX_ABS: &'static str = "max_abs_f32";
    const REDUCE_SQRT: &'static str = "sqrt_f32";
    const OP_INTERNAL_VALUE: &'static str = "op_internal_value_f32";
    const OP_BOUNDARY_VALUE: &'static str = "op_boundary_value_f32";
    const OP_GREEN_GAUSS: &'static str = "op_green_gauss_f32";
    const OP_LEAST_SQUARES: &'static str = "op_least_squares_f32";
    const OP_INTERNAL_FLUX: &'static str = "op_internal_flux_f32";
    const OP_BOUNDARY_FLUX: &'static str = "op_boundary_flux_f32";
    const OP_CELL_GATHER: &'static str = "op_cell_gather_f32";
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
    const REDUCE_SUM: &'static str = "sum_f64";
    const REDUCE_DOT: &'static str = "dot_f64";
    const REDUCE_L2: &'static str = "l2_f64";
    const REDUCE_MAX_ABS: &'static str = "max_abs_f64";
    const REDUCE_SQRT: &'static str = "sqrt_f64";
    const OP_INTERNAL_VALUE: &'static str = "op_internal_value_f64";
    const OP_BOUNDARY_VALUE: &'static str = "op_boundary_value_f64";
    const OP_GREEN_GAUSS: &'static str = "op_green_gauss_f64";
    const OP_LEAST_SQUARES: &'static str = "op_least_squares_f64";
    const OP_INTERNAL_FLUX: &'static str = "op_internal_flux_f64";
    const OP_BOUNDARY_FLUX: &'static str = "op_boundary_flux_f64";
    const OP_CELL_GATHER: &'static str = "op_cell_gather_f64";
}

impl<T: CudaFvmScalar> DeviceReduction<T, CudaBackend> {
    fn launch_unary(
        &mut self,
        backend: &CudaBackend,
        input: &CudaBuffer<T>,
        function_name: &str,
    ) -> Result<T, AcceleratorError> {
        self.launch_unary_on(backend, 0, input, function_name)
    }

    fn launch_unary_on(
        &mut self,
        backend: &CudaBackend,
        stream_index: usize,
        input: &CudaBuffer<T>,
        function_name: &str,
    ) -> Result<T, AcceleratorError> {
        self.validate(input.len())?;
        backend.validate_buffer(input)?;
        backend.validate_buffer(&self.result)?;
        backend.validate_buffer(&self.workspace)?;
        backend.validate_backend_id(self.backend_id)?;
        let function = backend.function(REDUCTION_KERNELS, "reduction.cu", function_name)?;
        let count = u32::try_from(input.len()).map_err(|_| AcceleratorError::IndexOverflow {
            what: "reduction input length",
            value: input.len(),
        })?;
        let stream = backend.stream_at(stream_index)?;
        let blocks =
            u32::try_from(self.workspace.len()).map_err(|_| AcceleratorError::IndexOverflow {
                what: "reduction workspace length",
                value: self.workspace.len(),
            })?;
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&input.inner)
            .arg(&mut self.workspace.inner)
            .arg(&count);
        let first_config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the kernel reads `count` values and writes one partial per block.
        unsafe { launch.launch(first_config) }
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        let combine_name = if function_name == T::REDUCE_MAX_ABS {
            T::REDUCE_MAX_ABS
        } else {
            T::REDUCE_SUM
        };
        let combine = backend.function(REDUCTION_KERNELS, "reduction.cu", combine_name)?;
        let mut combine_launch = stream.launch_builder(&combine);
        combine_launch
            .arg(&self.workspace.inner)
            .arg(&mut self.result.inner)
            .arg(&blocks);
        let final_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the second stage consumes exactly the initialized partials.
        unsafe { combine_launch.launch(final_config) }
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        if function_name == T::REDUCE_L2 {
            let sqrt = backend.function(REDUCTION_KERNELS, "reduction.cu", T::REDUCE_SQRT)?;
            let mut sqrt_launch = stream.launch_builder(&sqrt);
            sqrt_launch.arg(&mut self.result.inner);
            // SAFETY: the result buffer contains one initialized nonnegative sum.
            unsafe { sqrt_launch.launch(LaunchConfig::for_num_elems(1)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        let mut value = [T::zeroed()];
        backend.download_on(stream_index, &self.result, &mut value)?;
        Ok(value[0])
    }

    /// Deterministic resident sum.
    pub fn sum(
        &mut self,
        backend: &CudaBackend,
        input: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.launch_unary(backend, input, T::REDUCE_SUM)
    }

    /// Deterministic resident sum on an indexed stream.
    pub fn sum_on(
        &mut self,
        backend: &CudaBackend,
        stream_index: usize,
        input: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.launch_unary_on(backend, stream_index, input, T::REDUCE_SUM)
    }

    /// Deterministic resident dot product.
    pub fn dot(
        &mut self,
        backend: &CudaBackend,
        lhs: &CudaBuffer<T>,
        rhs: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.dot_on(backend, 0, lhs, rhs)
    }

    /// Deterministic resident dot product on an indexed stream.
    pub fn dot_on(
        &mut self,
        backend: &CudaBackend,
        stream_index: usize,
        lhs: &CudaBuffer<T>,
        rhs: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.validate(lhs.len())?;
        self.validate(rhs.len())?;
        backend.validate_buffer(lhs)?;
        backend.validate_buffer(rhs)?;
        backend.validate_buffer(&self.result)?;
        backend.validate_buffer(&self.workspace)?;
        backend.validate_backend_id(self.backend_id)?;
        let function = backend.function(REDUCTION_KERNELS, "reduction.cu", T::REDUCE_DOT)?;
        let count = u32::try_from(lhs.len()).map_err(|_| AcceleratorError::IndexOverflow {
            what: "reduction input length",
            value: lhs.len(),
        })?;
        let stream = backend.stream_at(stream_index)?;
        let blocks =
            u32::try_from(self.workspace.len()).map_err(|_| AcceleratorError::IndexOverflow {
                what: "reduction workspace length",
                value: self.workspace.len(),
            })?;
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&lhs.inner)
            .arg(&rhs.inner)
            .arg(&mut self.workspace.inner)
            .arg(&count);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: both inputs contain `count` values and the workspace has one slot per block.
        unsafe { launch.launch(config) }
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        let combine = backend.function(REDUCTION_KERNELS, "reduction.cu", T::REDUCE_SUM)?;
        let mut combine_launch = stream.launch_builder(&combine);
        combine_launch
            .arg(&self.workspace.inner)
            .arg(&mut self.result.inner)
            .arg(&blocks);
        let final_config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: the second stage consumes exactly the initialized dot-product partials.
        unsafe { combine_launch.launch(final_config) }
            .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        let mut value = [T::zeroed()];
        backend.download_on(stream_index, &self.result, &mut value)?;
        Ok(value[0])
    }

    /// Deterministic resident Euclidean norm.
    pub fn l2_norm(
        &mut self,
        backend: &CudaBackend,
        input: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.launch_unary(backend, input, T::REDUCE_L2)
    }

    /// Deterministic resident Euclidean norm on an indexed stream.
    pub fn l2_norm_on(
        &mut self,
        backend: &CudaBackend,
        stream_index: usize,
        input: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.launch_unary_on(backend, stream_index, input, T::REDUCE_L2)
    }

    /// Deterministic resident maximum absolute value.
    pub fn max_abs(
        &mut self,
        backend: &CudaBackend,
        input: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.launch_unary(backend, input, T::REDUCE_MAX_ABS)
    }

    /// Deterministic resident maximum absolute value on an indexed stream.
    pub fn max_abs_on(
        &mut self,
        backend: &CudaBackend,
        stream_index: usize,
        input: &CudaBuffer<T>,
    ) -> Result<T, AcceleratorError> {
        self.launch_unary_on(backend, stream_index, input, T::REDUCE_MAX_ABS)
    }
}

impl CudaBackend {
    /// Enqueue a complete multi-component FVM residual evaluation on stream 0.
    pub fn evaluate_residual<T: CudaFvmScalar>(
        &self,
        operator: &DeviceFvmOperator<Self>,
        state: &mut DeviceFvmState<T, Self>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        self.evaluate_residual_on(0, operator, state, current_epochs)
    }

    /// Enqueue a complete multi-component FVM residual evaluation on an indexed stream.
    pub fn evaluate_residual_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        operator: &DeviceFvmOperator<Self>,
        state: &mut DeviceFvmState<T, Self>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        operator.validate_state(state, current_epochs)?;
        self.validate_backend_id(operator.plan.backend_id)?;
        let stream = self.stream_at(stream_index)?;
        let cells = checked_count(operator.plan.cell_ids.len(), "cell count")?;
        let faces = checked_count(operator.plan.face_count(), "face count")?;
        let internal = checked_count(operator.plan.internal_face_count, "internal face count")?;
        let boundary = checked_count(operator.plan.boundary_face_count, "boundary face count")?;
        let components = checked_count(state.components(), "component count")?;
        let internal_work = checked_work(internal, components, "internal operator work")?;
        let boundary_work = checked_work(boundary, components, "boundary operator work")?;
        let cell_work = checked_work(cells, components, "cell operator work")?;
        let mode = operator.schemes.reconstruction.mode;
        let needs_green_gauss = matches!(
            mode,
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss)
                | ReconstructionMode::LeastSquaresWithGreenGaussFallback
        );
        if needs_green_gauss {
            if internal_work != 0 {
                let function =
                    self.function(OPERATOR_KERNELS, "fvm_operator.cu", T::OP_INTERNAL_VALUE)?;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&operator.plan.internal_owner.inner)
                    .arg(&operator.plan.internal_neighbor.inner)
                    .arg(&state.cell_values.inner)
                    .arg(&mut state.face_values.inner)
                    .arg(&cells)
                    .arg(&faces)
                    .arg(&internal)
                    .arg(&components);
                // SAFETY: plan and state lengths were validated above.
                unsafe { launch.launch(LaunchConfig::for_num_elems(internal_work)) }
                    .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
            }
            if boundary_work != 0 {
                let function =
                    self.function(OPERATOR_KERNELS, "fvm_operator.cu", T::OP_BOUNDARY_VALUE)?;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&operator.plan.boundary_owner.inner)
                    .arg(&operator.boundary.convective_kind.inner)
                    .arg(&operator.boundary.convective_alpha.inner)
                    .arg(&operator.boundary.convective_beta.inner)
                    .arg(&operator.boundary.convective_gamma.inner)
                    .arg(&state.cell_values.inner)
                    .arg(&state.boundary_values.inner)
                    .arg(&state.boundary_override_active.inner)
                    .arg(&mut state.face_values.inner)
                    .arg(&cells)
                    .arg(&faces)
                    .arg(&internal)
                    .arg(&boundary)
                    .arg(&components);
                // SAFETY: boundary coefficient and state lengths match the plan.
                unsafe { launch.launch(LaunchConfig::for_num_elems(boundary_work)) }
                    .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
            }
            if cell_work != 0 {
                let function =
                    self.function(OPERATOR_KERNELS, "fvm_operator.cu", T::OP_GREEN_GAUSS)?;
                let mut launch = stream.launch_builder(&function);
                launch
                    .arg(&operator.plan.cell_face_offsets.inner)
                    .arg(&operator.plan.cell_face_indices.inner)
                    .arg(&operator.plan.cell_face_signs.inner)
                    .arg(&operator.plan.face_geometry_indices.inner)
                    .arg(&operator.plan.face_normal_x.inner)
                    .arg(&operator.plan.face_normal_y.inner)
                    .arg(&operator.plan.face_normal_z.inner)
                    .arg(&operator.plan.cell_volume.inner)
                    .arg(&operator.plan.face_active.inner)
                    .arg(&operator.plan.cell_active.inner)
                    .arg(&state.face_values.inner)
                    .arg(&mut state.gradient_x.inner)
                    .arg(&mut state.gradient_y.inner)
                    .arg(&mut state.gradient_z.inner)
                    .arg(&cells)
                    .arg(&faces)
                    .arg(&components);
                // SAFETY: cell-face CSR and all referenced arrays were validated at compile time.
                unsafe { launch.launch(LaunchConfig::for_num_elems(cell_work)) }
                    .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
            }
        }

        if !matches!(
            mode,
            ReconstructionMode::GradientOnly(ReconstructionGradient::GreenGauss)
        ) && cell_work != 0
        {
            let function =
                self.function(OPERATOR_KERNELS, "fvm_operator.cu", T::OP_LEAST_SQUARES)?;
            let preserve_fallback = u32::from(matches!(
                mode,
                ReconstructionMode::LeastSquaresWithGreenGaussFallback
            ));
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&operator.plan.cell_face_offsets.inner)
                .arg(&operator.plan.cell_face_indices.inner)
                .arg(&operator.least_squares.neighbor.inner)
                .arg(&operator.least_squares.weight_x.inner)
                .arg(&operator.least_squares.weight_y.inner)
                .arg(&operator.least_squares.weight_z.inner)
                .arg(&operator.least_squares.fallback.inner)
                .arg(&operator.plan.face_active.inner)
                .arg(&operator.plan.cell_active.inner)
                .arg(&operator.plan.boundary_owner.inner)
                .arg(&operator.boundary.convective_kind.inner)
                .arg(&operator.boundary.convective_alpha.inner)
                .arg(&operator.boundary.convective_beta.inner)
                .arg(&operator.boundary.convective_gamma.inner)
                .arg(&state.cell_values.inner)
                .arg(&state.boundary_values.inner)
                .arg(&state.boundary_override_active.inner)
                .arg(&mut state.gradient_x.inner)
                .arg(&mut state.gradient_y.inner)
                .arg(&mut state.gradient_z.inner)
                .arg(&cells)
                .arg(&internal)
                .arg(&boundary)
                .arg(&components)
                .arg(&preserve_fallback);
            // SAFETY: least-squares arrays align one-for-one with cell-face CSR incidences.
            unsafe { launch.launch(LaunchConfig::for_num_elems(cell_work)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }

        let (scheme, blend, scheme_limiter) = encode_convective(operator.schemes.convective);
        let reconstruction_limiter = encode_limiter_option(operator.schemes.reconstruction.limiter);
        let reconstruct = u32::from(operator.metadata.reconstruction_order > 1);
        let diffusivity = operator.schemes.diffusion.diffusivity;
        let diffusion_mode = encode_diffusion(operator.schemes.diffusion.non_orthogonal_mode);
        if internal_work != 0 {
            let function =
                self.function(OPERATOR_KERNELS, "fvm_operator.cu", T::OP_INTERNAL_FLUX)?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&operator.plan.internal_owner.inner)
                .arg(&operator.plan.internal_neighbor.inner)
                .arg(&operator.plan.internal_geometry.inner)
                .arg(&operator.plan.face_active.inner)
                .arg(&operator.plan.face_normal_x.inner)
                .arg(&operator.plan.face_normal_y.inner)
                .arg(&operator.plan.face_normal_z.inner)
                .arg(&operator.plan.face_center_x.inner)
                .arg(&operator.plan.face_center_y.inner)
                .arg(&operator.plan.face_center_z.inner)
                .arg(&operator.plan.cell_center_x.inner)
                .arg(&operator.plan.cell_center_y.inner)
                .arg(&operator.plan.cell_center_z.inner)
                .arg(&state.cell_values.inner)
                .arg(&state.face_mass_flux.inner)
                .arg(&state.gradient_x.inner)
                .arg(&state.gradient_y.inner)
                .arg(&state.gradient_z.inner)
                .arg(&state.face_source.inner)
                .arg(&mut state.face_flux.inner)
                .arg(&mut state.face_deferred_source.inner)
                .arg(&scheme)
                .arg(&blend)
                .arg(&scheme_limiter)
                .arg(&reconstruction_limiter)
                .arg(&reconstruct)
                .arg(&diffusivity)
                .arg(&diffusion_mode)
                .arg(&cells)
                .arg(&faces)
                .arg(&internal)
                .arg(&components);
            // SAFETY: all dense cell/geometry indices and workspace sizes are plan-validated.
            unsafe { launch.launch(LaunchConfig::for_num_elems(internal_work)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        if boundary_work != 0 {
            let function =
                self.function(OPERATOR_KERNELS, "fvm_operator.cu", T::OP_BOUNDARY_FLUX)?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&operator.plan.boundary_owner.inner)
                .arg(&operator.plan.boundary_geometry.inner)
                .arg(&operator.plan.face_active.inner)
                .arg(&operator.plan.face_area.inner)
                .arg(&operator.plan.face_center_x.inner)
                .arg(&operator.plan.face_center_y.inner)
                .arg(&operator.plan.face_center_z.inner)
                .arg(&operator.plan.cell_center_x.inner)
                .arg(&operator.plan.cell_center_y.inner)
                .arg(&operator.plan.cell_center_z.inner)
                .arg(&operator.boundary.convective_kind.inner)
                .arg(&operator.boundary.convective_alpha.inner)
                .arg(&operator.boundary.convective_beta.inner)
                .arg(&operator.boundary.convective_gamma.inner)
                .arg(&operator.boundary.diffusive_kind.inner)
                .arg(&operator.boundary.diffusive_alpha.inner)
                .arg(&operator.boundary.diffusive_beta.inner)
                .arg(&operator.boundary.diffusive_gamma.inner)
                .arg(&state.cell_values.inner)
                .arg(&state.boundary_values.inner)
                .arg(&state.boundary_override_active.inner)
                .arg(&state.face_mass_flux.inner)
                .arg(&state.gradient_x.inner)
                .arg(&state.gradient_y.inner)
                .arg(&state.gradient_z.inner)
                .arg(&state.face_source.inner)
                .arg(&mut state.face_flux.inner)
                .arg(&mut state.face_deferred_source.inner)
                .arg(&scheme)
                .arg(&blend)
                .arg(&scheme_limiter)
                .arg(&reconstruction_limiter)
                .arg(&reconstruct)
                .arg(&diffusivity)
                .arg(&cells)
                .arg(&faces)
                .arg(&internal)
                .arg(&boundary)
                .arg(&components);
            // SAFETY: packed boundary metadata has exactly one entry per boundary face.
            unsafe { launch.launch(LaunchConfig::for_num_elems(boundary_work)) }
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        if cell_work != 0 {
            let function = self.function(OPERATOR_KERNELS, "fvm_operator.cu", T::OP_CELL_GATHER)?;
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&operator.plan.cell_face_offsets.inner)
                .arg(&operator.plan.cell_face_indices.inner)
                .arg(&operator.plan.cell_face_signs.inner)
                .arg(&operator.plan.cell_active.inner)
                .arg(&state.face_flux.inner)
                .arg(&state.face_deferred_source.inner)
                .arg(&state.cell_source.inner)
                .arg(&mut state.residual.inner)
                .arg(&cells)
                .arg(&faces)
                .arg(&components);
            // SAFETY: deterministic CSR incidences reference only initialized face workspaces.
            unsafe { launch.launch(LaunchConfig::for_num_elems(cell_work)) }
                .map(|_| ())
                .map_err(|e| AcceleratorError::KernelLaunchFailed(e.to_string()))?;
        }
        Ok(())
    }
}

fn checked_count(value: usize, what: &'static str) -> Result<u32, AcceleratorError> {
    u32::try_from(value).map_err(|_| AcceleratorError::IndexOverflow { what, value })
}

fn checked_work(count: u32, components: u32, what: &'static str) -> Result<u32, AcceleratorError> {
    count
        .checked_mul(components)
        .ok_or(AcceleratorError::IndexOverflow {
            what,
            value: usize::MAX,
        })
}

fn encode_limiter(limiter: SlopeLimiterFamily) -> u32 {
    match limiter {
        SlopeLimiterFamily::None => 0,
        SlopeLimiterFamily::Minmod => 1,
        SlopeLimiterFamily::VanLeer => 2,
        SlopeLimiterFamily::Superbee => 3,
    }
}

fn encode_limiter_option(limiter: LimiterOption) -> u32 {
    match limiter {
        LimiterOption::None => 0,
        LimiterOption::Family(limiter) => encode_limiter(limiter),
    }
}

fn encode_convective(scheme: ConvectiveScheme) -> (u32, f64, u32) {
    match scheme {
        ConvectiveScheme::Upwind => (0, 0.0, 0),
        ConvectiveScheme::Central => (1, 0.0, 0),
        ConvectiveScheme::BoundedLinear { blend } => (2, blend, 0),
        ConvectiveScheme::BlendUpwindCentral { blend } => (3, blend, 0),
        ConvectiveScheme::HighResolution { blend, limiter } => (4, blend, encode_limiter(limiter)),
    }
}

fn encode_diffusion(mode: NonOrthogonalCorrectionMode) -> u32 {
    match mode {
        NonOrthogonalCorrectionMode::OrthogonalOnly => 0,
        NonOrthogonalCorrectionMode::Deferred => 1,
        NonOrthogonalCorrectionMode::FullyCorrected => 2,
    }
}

impl CudaBackend {
    /// Fill a resident vector without allocating or staging through the host.
    pub fn fill<T: CudaFvmScalar>(
        &self,
        output: &mut CudaBuffer<T>,
        value: T,
    ) -> Result<(), AcceleratorError> {
        self.fill_on(0, output, value)
    }

    /// Fill a resident vector on an indexed stream.
    pub fn fill_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        output: &mut CudaBuffer<T>,
        value: T,
    ) -> Result<(), AcceleratorError> {
        self.validate_buffer(output)?;
        if output.is_empty() {
            return Ok(());
        }
        let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::FILL)?;
        let count = checked_count(output.len(), "fill length")?;
        let mut launch = self.stream_at(stream_index)?.launch_builder(&function);
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
        self.validate_buffer(input)?;
        self.validate_buffer(output)?;
        self.copy_on(0, input, output)
    }

    /// Copy equally sized resident vectors on an indexed stream.
    pub fn copy_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        input: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        self.validate_buffer(input)?;
        self.validate_buffer(output)?;
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
        let count = checked_count(input.len(), "copy length")?;
        let mut launch = self.stream_at(stream_index)?.launch_builder(&function);
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
        self.validate_buffer(x)?;
        self.validate_buffer(y)?;
        self.axpy_on(0, alpha, x, y)
    }

    /// Compute `y = alpha * x + y` on an indexed stream.
    pub fn axpy_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        alpha: T,
        x: &CudaBuffer<T>,
        y: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        self.validate_buffer(x)?;
        self.validate_buffer(y)?;
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
        let count = checked_count(x.len(), "AXPY length")?;
        let mut launch = self.stream_at(stream_index)?.launch_builder(&function);
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
        self.validate_buffer(mask)?;
        self.validate_buffer(values)?;
        self.apply_mask_on(0, mask, values)
    }

    /// Apply a byte mask on an indexed stream.
    pub fn apply_mask_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        mask: &CudaBuffer<u8>,
        values: &mut CudaBuffer<T>,
    ) -> Result<(), AcceleratorError> {
        self.validate_buffer(mask)?;
        self.validate_buffer(values)?;
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
        let count = checked_count(mask.len(), "mask length")?;
        let mut launch = self.stream_at(stream_index)?.launch_builder(&function);
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
        self.compute_gradients_on(0, plan, state, current_epochs)
    }

    /// Compute Green--Gauss gradients on an indexed stream.
    pub fn compute_gradients_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        validate_scalar_state(plan, state)?;
        self.validate_backend_id(plan.backend_id)?;
        let stream = self.stream_at(stream_index)?;
        if plan.internal_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::INTERNAL_VALUE)?;
            let count = checked_count(plan.internal_face_count, "internal face count")?;
            let mut launch = stream.launch_builder(&function);
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
            let count = checked_count(plan.boundary_face_count, "boundary face count")?;
            let offset = checked_count(plan.internal_face_count, "internal face offset")?;
            let mut launch = stream.launch_builder(&function);
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
            let count = checked_count(plan.cell_ids.len(), "cell count")?;
            let mut launch = stream.launch_builder(&function);
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
        self.compute_diffusive_face_fluxes_on(0, plan, state, diffusivity, current_epochs)
    }

    /// Compute orthogonal diffusive fluxes on an indexed stream.
    pub fn compute_diffusive_face_fluxes_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        diffusivity: f64,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        validate_scalar_state(plan, state)?;
        self.validate_backend_id(plan.backend_id)?;
        let stream = self.stream_at(stream_index)?;
        let diffusivity = T::from_f64(diffusivity);
        if plan.internal_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::INTERNAL_DIFFUSION)?;
            let count = checked_count(plan.internal_face_count, "internal face count")?;
            let mut launch = stream.launch_builder(&function);
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
            let count = checked_count(plan.boundary_face_count, "boundary face count")?;
            let offset = checked_count(plan.internal_face_count, "internal face offset")?;
            let mut launch = stream.launch_builder(&function);
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
        self.compute_face_fluxes_on(0, plan, state, scheme, current_epochs)
    }

    /// Compute scalar convective face fluxes on an indexed stream.
    pub fn compute_face_fluxes_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        scheme: ScalarFluxScheme,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        validate_scalar_state(plan, state)?;
        self.validate_backend_id(plan.backend_id)?;
        let stream = self.stream_at(stream_index)?;
        let scheme = scheme as u32;
        if plan.internal_face_count != 0 {
            let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::INTERNAL_FLUX)?;
            let count = checked_count(plan.internal_face_count, "internal face count")?;
            let mut launch = stream.launch_builder(&function);
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
            let count = checked_count(plan.boundary_face_count, "boundary face count")?;
            let offset = checked_count(plan.internal_face_count, "internal face offset")?;
            let mut launch = stream.launch_builder(&function);
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
        self.assemble_cell_residuals_on(0, plan, state, current_epochs)
    }

    /// Assemble deterministic scalar cell residuals on an indexed stream.
    pub fn assemble_cell_residuals_on<T: CudaFvmScalar>(
        &self,
        stream_index: usize,
        plan: &DeviceFvmPlan<Self>,
        state: &mut DeviceFvmState<T, Self>,
        current_epochs: PlanEpochs,
    ) -> Result<(), AcceleratorError> {
        plan.validate(current_epochs)?;
        validate_scalar_state(plan, state)?;
        self.validate_backend_id(plan.backend_id)?;
        if plan.cell_ids.is_empty() {
            return Ok(());
        }
        let function = self.function(FVM_KERNELS, "fvm_scalar.cu", T::CELL_GATHER)?;
        let count = checked_count(plan.cell_ids.len(), "cell count")?;
        let mut launch = self.stream_at(stream_index)?.launch_builder(&function);
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
