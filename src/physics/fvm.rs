//! Finite-volume oriented wrappers over mesh-sieve topology/data/geometry primitives.
//!
//! This module is the canonical entrypoint for finite-volume (FV) callers.
//! It intentionally exposes FV-ready mesh loops and interpolation helpers without
//! requiring users to couple directly to finite-element runtime APIs.

use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::discretization::runtime::{CellGeometry, FaceGeometry, FluxStencil};
use crate::mesh_error::MeshSieveError;
use crate::topology::point::PointId;
use crate::topology::sieve::Sieve;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceKind {
    Internal,
    Boundary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvectiveScheme {
    Upwind,
    Central,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BoundaryCondition {
    Dirichlet { value: f64 },
    Neumann { gradient: f64 },
    Robin { alpha: f64, beta: f64, gamma: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiffusionSettings {
    pub diffusivity: f64,
    pub non_orthogonal_correction: bool,
}

impl Default for DiffusionSettings {
    fn default() -> Self {
        Self { diffusivity: 1.0, non_orthogonal_correction: true }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FluxAssembly {
    pub residual: HashMap<PointId, f64>,
    pub source: HashMap<PointId, f64>,
}

impl FluxAssembly {
    fn add_residual(&mut self, cell: PointId, value: f64) {
        *self.residual.entry(cell).or_insert(0.0) += value;
    }
    fn add_source(&mut self, cell: PointId, value: f64) {
        *self.source.entry(cell).or_insert(0.0) += value;
    }
}

#[derive(Clone, Debug, Default)]
pub struct FvmFaceLoops {
    pub internal: Vec<FluxStencil>,
    pub boundary: Vec<FluxStencil>,
}

#[derive(Clone, Debug, Default)]
pub struct FvmInputs {
    pub loops: FvmFaceLoops,
    pub cell_geometry: Vec<(PointId, CellGeometry)>,
    pub face_geometry: Vec<(PointId, FaceGeometry)>,
}

impl FvmInputs {
    pub fn new(
        stencils: impl IntoIterator<Item = FluxStencil>,
        cell_geometry: Vec<(PointId, CellGeometry)>,
        face_geometry: Vec<(PointId, FaceGeometry)>,
    ) -> Self {
        let mut loops = FvmFaceLoops::default();
        for stencil in stencils {
            if stencil.right.is_some() {
                loops.internal.push(stencil);
            } else {
                loops.boundary.push(stencil);
            }
        }
        Self { loops, cell_geometry, face_geometry }
    }

    pub fn internal_faces(&self) -> impl Iterator<Item = &FluxStencil> { self.loops.internal.iter() }
    pub fn boundary_faces(&self) -> impl Iterator<Item = &FluxStencil> { self.loops.boundary.iter() }
    pub fn cell_metrics(&self, cell: PointId) -> Option<&CellGeometry> { self.cell_geometry.iter().find_map(|(id,m)|(*id==cell).then_some(m)) }
    pub fn face_metrics(&self, face: PointId) -> Option<&FaceGeometry> { self.face_geometry.iter().find_map(|(id,m)|(*id==face).then_some(m)) }
}

pub fn classify_face_loops<T>(topology: &T, faces: impl IntoIterator<Item = PointId>) -> Result<FvmFaceLoops, MeshSieveError>
where T: Sieve<Point = PointId>,
{
    let mut loops = FvmFaceLoops::default();
    for face in faces {
        let mut cells: Vec<_> = topology.support_points(face).collect();
        cells.sort_unstable();
        match cells.as_slice() {
            [left] => loops.boundary.push(FluxStencil { face, left: *left, right: None }),
            [left, right] => loops.internal.push(FluxStencil { face, left: *left, right: Some(*right) }),
            [] => return Err(MeshSieveError::InvalidGeometry(format!("face {face:?} has no supporting cells"))),
            _ => return Err(MeshSieveError::InvalidGeometry(format!("non-manifold face {face:?} has {} support cells", cells.len()))),
        }
    }
    Ok(loops)
}

pub fn interpolate_face_centered_scalar<S: Storage<f64>>(section: &Section<f64, S>, stencil: &FluxStencil) -> Result<f64, MeshSieveError> {
    let left = section.try_restrict(stencil.left)?;
    let lval = *left.first().ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar value at cell {}", stencil.left)))?;
    if let Some(right_cell) = stencil.right {
        let right = section.try_restrict(right_cell)?;
        let rval = *right.first().ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar value at cell {}", right_cell)))?;
        Ok(0.5 * (lval + rval))
    } else {
        Ok(lval)
    }
}

pub fn interpolate_cell_centered_scalar<S: Storage<f64>>(section: &Section<f64, S>, incident_faces: &[PointId]) -> Result<f64, MeshSieveError> {
    if incident_faces.is_empty() {
        return Err(MeshSieveError::InvalidGeometry("cannot interpolate to cell center from zero incident faces".to_string()));
    }
    let mut accum = 0.0;
    for face in incident_faces {
        let value = section.try_restrict(*face)?;
        let scalar = *value.first().ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar value at face {}", face)))?;
        accum += scalar;
    }
    Ok(accum / incident_faces.len() as f64)
}

pub fn assemble_convective_fluxes(
    inputs: &FvmInputs,
    cell_scalar: &HashMap<PointId, f64>,
    face_mass_flux: &HashMap<PointId, f64>,
    boundary_conditions: &HashMap<PointId, BoundaryCondition>,
    scheme: ConvectiveScheme,
) -> Result<FluxAssembly, MeshSieveError> {
    let mut assembly = FluxAssembly::default();
    for stencil in inputs.internal_faces() {
        let mdot = *face_mass_flux.get(&stencil.face).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing mass flux for face {}", stencil.face)))?;
        let l = *cell_scalar.get(&stencil.left).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left)))?;
        let rcell = stencil.right.expect("internal face must have neighbor");
        let r = *cell_scalar.get(&rcell).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", rcell)))?;
        let phi_f = match scheme { ConvectiveScheme::Upwind => if mdot >= 0.0 { l } else { r }, ConvectiveScheme::Central => 0.5 * (l + r) };
        let flux = mdot * phi_f;
        assembly.add_residual(stencil.left, flux);
        assembly.add_residual(rcell, -flux);
    }
    for stencil in inputs.boundary_faces() {
        let mdot = *face_mass_flux.get(&stencil.face).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing mass flux for boundary face {}", stencil.face)))?;
        let phi_i = *cell_scalar.get(&stencil.left).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left)))?;
        let bc = boundary_conditions.get(&stencil.face).copied().ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing boundary condition for face {}", stencil.face)))?;
        let phi_b = match bc {
            BoundaryCondition::Dirichlet { value } => value,
            BoundaryCondition::Neumann { .. } => phi_i,
            BoundaryCondition::Robin { alpha, beta, gamma } => {
                if alpha.abs() < 1e-14 { phi_i } else { (gamma - beta * phi_i) / alpha }
            }
        };
        let phi_f = if mdot >= 0.0 { phi_i } else { phi_b };
        assembly.add_residual(stencil.left, mdot * phi_f);
    }
    Ok(assembly)
}

pub fn assemble_diffusive_fluxes(
    inputs: &FvmInputs,
    cell_scalar: &HashMap<PointId, f64>,
    cell_gradient: &HashMap<PointId, Vec<f64>>,
    boundary_conditions: &HashMap<PointId, BoundaryCondition>,
    settings: DiffusionSettings,
) -> Result<FluxAssembly, MeshSieveError> {
    let mut assembly = FluxAssembly::default();
    for stencil in inputs.internal_faces() {
        let fg = inputs.face_metrics(stencil.face).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing face geometry for face {}", stencil.face)))?;
        let lc = inputs.cell_metrics(stencil.left).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing cell geometry for cell {}", stencil.left)))?;
        let rcell = stencil.right.expect("internal face");
        let rc = inputs.cell_metrics(rcell).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing cell geometry for cell {}", rcell)))?;
        let phi_l = *cell_scalar.get(&stencil.left).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left)))?;
        let phi_r = *cell_scalar.get(&rcell).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", rcell)))?;
        let d = sub(&rc.centroid, &lc.centroid);
        let d2 = dot(&d, &d);
        let a = &fg.normal;
        let orth = if d2 > 0.0 { (phi_r - phi_l) * dot(a, &d) / d2 } else { 0.0 };
        let mut flux = -settings.diffusivity * orth;
        if settings.non_orthogonal_correction {
            let gf = avg(cell_gradient.get(&stencil.left), cell_gradient.get(&rcell), a.len())?;
            let a_orth = if d2 > 0.0 { scale(&d, dot(a, &d) / d2) } else { vec![0.0; a.len()] };
            let a_non = sub(a, &a_orth);
            flux += -settings.diffusivity * dot(&gf, &a_non);
        }
        assembly.add_residual(stencil.left, flux);
        assembly.add_residual(rcell, -flux);
    }
    for stencil in inputs.boundary_faces() {
        let fg = inputs.face_metrics(stencil.face).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing face geometry for boundary face {}", stencil.face)))?;
        let lc = inputs.cell_metrics(stencil.left).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing cell geometry for cell {}", stencil.left)))?;
        let phi_i = *cell_scalar.get(&stencil.left).ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing scalar for cell {}", stencil.left)))?;
        let bc = boundary_conditions.get(&stencil.face).copied().ok_or_else(|| MeshSieveError::InvalidGeometry(format!("missing boundary condition for face {}", stencil.face)))?;
        let n = normalize(&fg.normal);
        let d = norm(&sub(&fg.centroid, &lc.centroid)).max(1e-14);
        match bc {
            BoundaryCondition::Dirichlet { value } => {
                let flux = -settings.diffusivity * (value - phi_i) / d * fg.area;
                assembly.add_residual(stencil.left, flux);
            }
            BoundaryCondition::Neumann { gradient } => {
                let flux = -settings.diffusivity * gradient * fg.area;
                assembly.add_residual(stencil.left, flux);
            }
            BoundaryCondition::Robin { alpha, beta, gamma } => {
                let grad_n = (gamma - alpha * phi_i) / beta.max(1e-14);
                let flux = -settings.diffusivity * grad_n * fg.area;
                let _ = n;
                assembly.add_residual(stencil.left, flux);
                assembly.add_source(stencil.left, settings.diffusivity * gamma * fg.area / beta.max(1e-14));
            }
        }
    }
    Ok(assembly)
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b.iter()).map(|(x,y)| x*y).sum() }
fn sub(a: &[f64], b: &[f64]) -> Vec<f64> { a.iter().zip(b.iter()).map(|(x,y)| x-y).collect() }
fn scale(a: &[f64], s: f64) -> Vec<f64> { a.iter().map(|x| x*s).collect() }
fn norm(a: &[f64]) -> f64 { dot(a,a).sqrt() }
fn normalize(a:&[f64])->Vec<f64>{ let n=norm(a).max(1e-14); scale(a,1.0/n)}
fn avg(a: Option<&Vec<f64>>, b: Option<&Vec<f64>>, n: usize) -> Result<Vec<f64>, MeshSieveError> {
    let la = a.ok_or_else(|| MeshSieveError::InvalidGeometry("missing left gradient".into()))?;
    let lb = b.ok_or_else(|| MeshSieveError::InvalidGeometry("missing right gradient".into()))?;
    if la.len() != n || lb.len() != n { return Err(MeshSieveError::InvalidGeometry("gradient dimension mismatch".into())); }
    Ok(la.iter().zip(lb.iter()).map(|(x,y)|0.5*(x+y)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pid(id: u64) -> PointId { PointId::new(id).unwrap() }

    #[test]
    fn convective_two_cell_conservation() {
        let c0 = pid(1); let c1 = pid(2); let f = pid(10);
        let inputs = FvmInputs::new([FluxStencil{face:f,left:c0,right:Some(c1)}], vec![], vec![]);
        let phi = HashMap::from([(c0, 2.0), (c1, 6.0)]);
        let mdot = HashMap::from([(f, 3.0)]);
        let r = assemble_convective_fluxes(&inputs, &phi, &mdot, &HashMap::new(), ConvectiveScheme::Central).unwrap();
        assert!((r.residual[&c0] + r.residual[&c1]).abs() < 1e-12);
        assert_eq!(r.residual[&c0], 12.0);
    }

    #[test]
    fn diffusive_two_cell_symmetry() {
        let c0=pid(1); let c1=pid(2); let f=pid(9);
        let inputs = FvmInputs::new([FluxStencil{face:f,left:c0,right:Some(c1)}], vec![(c0,CellGeometry{centroid:vec![0.0,0.0],volume:1.0}),(c1,CellGeometry{centroid:vec![1.0,0.0],volume:1.0})], vec![(f,FaceGeometry{face:f,centroid:vec![0.5,0.0],normal:vec![1.0,0.0],area:1.0,neighbors:vec![c0,c1]})]);
        let phi = HashMap::from([(c0,1.0),(c1,3.0)]);
        let grad = HashMap::from([(c0,vec![2.0,0.0]),(c1,vec![2.0,0.0])]);
        let r = assemble_diffusive_fluxes(&inputs,&phi,&grad,&HashMap::new(),DiffusionSettings::default()).unwrap();
        assert!((r.residual[&c0] + r.residual[&c1]).abs() < 1e-12);
    }
}
