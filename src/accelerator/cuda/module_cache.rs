//! Process-local NVRTC/module cache owned by a backend.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{Ptx, compile_ptx};

use crate::accelerator::AcceleratorError;

pub(super) struct CudaModuleCache {
    context: Arc<CudaContext>,
    modules: Mutex<HashMap<u64, Arc<CudaModule>>>,
}

impl CudaModuleCache {
    pub(super) fn new(context: Arc<CudaContext>) -> Self {
        Self {
            context,
            modules: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn function(
        &self,
        source: &str,
        source_name: &str,
        function_name: &str,
    ) -> Result<CudaFunction, AcceleratorError> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        source_name.hash(&mut hasher);
        source.hash(&mut hasher);
        let key = hasher.finish();

        let module = {
            let mut modules = self.modules.lock().map_err(|_| {
                AcceleratorError::ModuleLoadFailed("CUDA module cache lock was poisoned".into())
            })?;
            if let Some(module) = modules.get(&key) {
                module.clone()
            } else {
                let ptx = std::panic::catch_unwind(|| compile_ptx(source))
                    .map_err(|payload| {
                        let message = if let Some(message) = payload.downcast_ref::<String>() {
                            message.clone()
                        } else if let Some(message) = payload.downcast_ref::<&str>() {
                            (*message).to_string()
                        } else {
                            "NVRTC dynamic library initialization panicked".to_string()
                        };
                        AcceleratorError::KernelCompilationFailed(format!(
                            "{source_name}: {message}"
                        ))
                    })?
                    .map_err(|e| {
                        AcceleratorError::KernelCompilationFailed(format!("{source_name}: {e}"))
                    })?;
                let module = self.context.load_module(ptx).map_err(|e| {
                    AcceleratorError::ModuleLoadFailed(format!("{source_name}: {e}"))
                })?;
                modules.insert(key, module.clone());
                module
            }
        };
        module
            .load_function(function_name)
            .map_err(|_| AcceleratorError::KernelNotFound(function_name.to_string()))
    }

    pub(super) fn ptx_function(
        &self,
        ptx: &str,
        module_name: &str,
        function_name: &str,
    ) -> Result<CudaFunction, AcceleratorError> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "ptx".hash(&mut hasher);
        module_name.hash(&mut hasher);
        ptx.hash(&mut hasher);
        let key = hasher.finish();
        let module = {
            let mut modules = self.modules.lock().map_err(|_| {
                AcceleratorError::ModuleLoadFailed("CUDA module cache lock was poisoned".into())
            })?;
            if let Some(module) = modules.get(&key) {
                module.clone()
            } else {
                let module = self.context.load_module(Ptx::from_src(ptx)).map_err(|e| {
                    AcceleratorError::ModuleLoadFailed(format!("{module_name}: {e}"))
                })?;
                modules.insert(key, module.clone());
                module
            }
        };
        module
            .load_function(function_name)
            .map_err(|_| AcceleratorError::KernelNotFound(function_name.to_string()))
    }
}
