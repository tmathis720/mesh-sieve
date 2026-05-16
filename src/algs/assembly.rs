//! High-level assembly orchestration across overlaps.
//!
//! This module provides helpers to:
//! 1. reduce/accumulate shared DOFs using ownership metadata,
//! 2. apply constraints, and
//! 3. complete (copy) values back to ghosts.

use crate::algs::communicator::{CommTag, Communicator, SectionCommTags};
use crate::algs::completion::section_completion::complete_section_with_tags_and_ownership;
use crate::data::constrained_section::{ConstraintSet, apply_constraints_to_section};
use crate::data::section::Section;
use crate::data::storage::Storage;
use crate::mesh_error::MeshSieveError;
use crate::overlap::delta::{CopyDelta, ValueDelta};
use crate::overlap::overlap::Overlap;
use crate::topology::ownership::PointOwnership;

/// Communication tags for the two-phase assembly workflow.
#[derive(Copy, Clone, Debug)]
pub struct AssemblyCommTags {
    /// Tags for the reduction/accumulation phase.
    pub reduce: SectionCommTags,
    /// Tags for the owner-to-ghost completion phase.
    pub complete: SectionCommTags,
}

impl AssemblyCommTags {
    /// Construct tags from a base, assigning deterministic offsets per phase.
    #[inline]
    pub const fn from_base(base: CommTag) -> Self {
        Self {
            reduce: SectionCommTags::from_base(base),
            complete: SectionCommTags::from_base(base.offset(2)),
        }
    }
}

/// High-level assembly across overlap with ownership and constraints.
///
/// # Steps
/// 1. Reduce/accumulate shared DOFs using `ValueDelta` and ownership metadata.
/// 2. Apply constraints locally.
/// 3. Complete (copy) owned values back to ghosts.
pub fn assemble_section_with_tags_and_ownership<V, S, D, C, Con>(
    section: &mut Section<V, S>,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
    tags: AssemblyCommTags,
    constraints: &Con,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + PartialEq + bytemuck::Pod + Copy + 'static,
    S: Storage<V>,
    D: ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default + Copy,
    C: Communicator + Sync,
    Con: ConstraintSet<V>,
{
    complete_section_with_tags_and_ownership::<V, S, D, C>(
        section,
        overlap,
        ownership,
        comm,
        my_rank,
        tags.reduce,
    )?;

    apply_constraints_to_section(section, constraints)?;

    complete_section_with_tags_and_ownership::<V, S, CopyDelta, C>(
        section,
        overlap,
        ownership,
        comm,
        my_rank,
        tags.complete,
    )?;

    Ok(())
}

/// Convenience wrapper using a legacy default tag (0xBEEF).
pub fn assemble_section_with_ownership<V, S, D, C, Con>(
    section: &mut Section<V, S>,
    overlap: &Overlap,
    ownership: &PointOwnership,
    comm: &C,
    my_rank: usize,
    constraints: &Con,
) -> Result<(), MeshSieveError>
where
    V: Clone + Default + Send + PartialEq + bytemuck::Pod + Copy + 'static,
    S: Storage<V>,
    D: ValueDelta<V> + Send + Sync + 'static,
    D::Part: bytemuck::Pod + Default + Copy,
    C: Communicator + Sync,
    Con: ConstraintSet<V>,
{
    let tags = AssemblyCommTags::from_base(CommTag::new(0xBEEF));
    assemble_section_with_tags_and_ownership::<V, S, D, C, Con>(
        section,
        overlap,
        ownership,
        comm,
        my_rank,
        tags,
        constraints,
    )
}

use crate::data::closure::{
    ClosureOrder, IdentitySectionSym, SectionSym, build_closure_index,
    build_closure_index_unoriented,
};
use crate::data::global_map::LocalToGlobalMap;
use crate::discretization::runtime::{ClosureDof, CsrPattern, DofMap, dof_map_from_closure_index};
use crate::topology::point::PointId;
use crate::topology::sieve::{Orientation, OrientedSieve, Sieve};
use std::collections::{BTreeSet, HashMap};

/// Extract an orientation-correct closure DOF map for one cell.
pub fn cell_closure_dof_map<T, V, Sct>(
    topology: &T,
    section: &Section<V, Sct>,
    cell: PointId,
    topology_version: u64,
    order: &ClosureOrder,
) -> Result<DofMap, MeshSieveError>
where
    T: Sieve<Point = PointId>,
    Sct: Storage<V>,
{
    let index = build_closure_index_unoriented(
        topology,
        section,
        cell,
        topology_version,
        order,
        &IdentitySectionSym,
    )?;
    Ok(dof_map_from_closure_index(&index))
}

/// Extract an oriented closure DOF map using a caller-provided symmetry table.
pub fn oriented_cell_closure_dof_map<T, V, Sct, O, Sym>(
    topology: &T,
    section: &Section<V, Sct>,
    cell: PointId,
    topology_version: u64,
    order: &ClosureOrder,
    sym: &Sym,
) -> Result<DofMap, MeshSieveError>
where
    T: OrientedSieve<Point = PointId, Orient = O>,
    Sct: Storage<V>,
    O: Orientation + Eq + std::hash::Hash,
    Sym: SectionSym<O>,
{
    let index = build_closure_index(topology, section, cell, topology_version, order, sym)?;
    Ok(dof_map_from_closure_index(&index))
}

/// Sparse matrix preallocation pattern from cells and a local section.
pub fn preallocation_csr_from_closure<T, V, Sct>(
    topology: &T,
    section: &Section<V, Sct>,
    cells: impl IntoIterator<Item = PointId>,
    topology_version: u64,
    order: &ClosureOrder,
) -> Result<CsrPattern, MeshSieveError>
where
    T: Sieve<Point = PointId>,
    Sct: Storage<V>,
{
    let mut rows: HashMap<ClosureDof, BTreeSet<ClosureDof>> = HashMap::new();
    for cell in cells {
        let map = cell_closure_dof_map(topology, section, cell, topology_version, order)?;
        let dofs = map.closure_dofs().to_vec();
        for row in &dofs {
            rows.entry(*row).or_default().extend(dofs.iter().copied());
        }
    }
    Ok(csr_from_rows(rows))
}

/// Sparse matrix preallocation pattern in global-numbered CSR columns.
pub fn global_preallocation_csr_from_closure<T, V, Sct>(
    topology: &T,
    section: &Section<V, Sct>,
    global_map: &LocalToGlobalMap,
    cells: impl IntoIterator<Item = PointId>,
    topology_version: u64,
    order: &ClosureOrder,
) -> Result<GlobalCsrPattern, MeshSieveError>
where
    T: Sieve<Point = PointId>,
    Sct: Storage<V>,
{
    let local = preallocation_csr_from_closure(topology, section, cells, topology_version, order)?;
    let rows = local
        .rows
        .iter()
        .map(|dof| {
            global_map
                .global_index(dof.point, dof.local_dof)
                .map(|g| g as usize)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let adjncy = local
        .adjncy
        .iter()
        .map(|dof| {
            global_map
                .global_index(dof.point, dof.local_dof)
                .map(|g| g as usize)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GlobalCsrPattern {
        xadj: local.xadj,
        adjncy,
        rows,
    })
}

/// Global-numbered CSR sparsity pattern for solver matrix preallocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlobalCsrPattern {
    /// CSR row offsets.
    pub xadj: Vec<usize>,
    /// Global column indices.
    pub adjncy: Vec<usize>,
    /// Global row indices represented by each CSR row.
    pub rows: Vec<usize>,
}

fn csr_from_rows(mut rows: HashMap<ClosureDof, BTreeSet<ClosureDof>>) -> CsrPattern {
    let mut row_dofs: Vec<_> = rows.keys().copied().collect();
    row_dofs.sort_unstable();
    let mut xadj = Vec::with_capacity(row_dofs.len() + 1);
    let mut adjncy = Vec::new();
    xadj.push(0);
    for row in &row_dofs {
        if let Some(cols) = rows.remove(row) {
            adjncy.extend(cols);
        }
        xadj.push(adjncy.len());
    }
    CsrPattern {
        xadj,
        adjncy,
        rows: row_dofs,
    }
}
