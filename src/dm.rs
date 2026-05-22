//! DMPLEX-like high-level mesh data-management facade.
//!
//! [`MeshDM`] is the feature-parity entry point for users coming from PETSc
//! DMPLEX.  It does not replace the lower-level sieve, section, label,
//! coordinate, distribution, and discretization modules; instead it owns those
//! pieces together and provides a single orchestration surface for the common
//! "build topology → attach data → validate → distribute → number sections →
//! assemble vectors/matrices" workflow.

use std::collections::BTreeMap;

use crate::algs::communicator::Communicator;
use crate::algs::distribute::{
    CellPartitioner, DistributedMeshData, DistributionConfig, distribute_with_overlap,
};
use crate::algs::dual_graph::{DualGraph, build_dual};
use crate::algs::renumber::{StratifiedOrdering, stratified_permutation};
use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::discretization::Discretization;
use crate::data::global_map::{LocalToGlobalMap, global_vector_for_map};
use crate::data::multi_section::constrained_section_from_label_specs;
use crate::data::section::Section;
use crate::data::storage::{Storage, VecStorage};
use crate::data::{ConstrainedSection, LabelConstraintSpec};
use crate::diagnostics::{MeshCheckOptions, run_mesh_checks};
use crate::io::MeshData;
use crate::mesh_error::MeshSieveError;
use crate::mesh_graph::{AdjacencyWeighting, MeshGraph, cell_adjacency_graph_with_cells};
use crate::overlap::overlap::Overlap;
use crate::topology::cell_type::CellType;
use crate::topology::labels::LabelSet;
use crate::topology::ownership::PointOwnership;
use crate::topology::point::PointId;
use crate::topology::sieve::strata::compute_strata;
use crate::topology::sieve::{MeshSieve, Sieve};

/// Options for a DMPLEX-like setup pipeline.
///
/// The refinement counters are intentionally high-level policy markers.  They
/// are stored with the DM so application-specific refinement hooks can inspect
/// them; generic topology refinement remains available in lower-level modules
/// because mesh-sieve supports several refinement representations.
#[derive(Clone, Debug)]
pub struct MeshDMOptions {
    /// Number of pre-distribution refinement passes requested by the user.
    pub pre_refine: usize,
    /// Whether the setup pipeline is expected to distribute the DM.
    pub distribute: bool,
    /// Number of ghost layers to create during distribution.
    pub distribute_overlap: usize,
    /// Number of post-distribution refinement passes requested by the user.
    pub post_refine: usize,
    /// Ensure coordinate data is present on local/ghost points after distribution.
    pub localize_coordinates: bool,
    /// Run orientation/symmetry checks when cell-type metadata is available.
    pub check_symmetry: bool,
    /// Run stratum/skeleton construction checks.
    pub check_skeleton: bool,
    /// Run face adjacency construction checks.
    pub check_faces: bool,
    /// Validate coordinate geometry metadata when coordinates are present.
    pub check_geometry: bool,
    /// Enable all mesh checks in a DMPLEX-like single switch.
    pub check_all: bool,
    /// Reorder section atlas entries by topology strata during setup.
    pub reorder_section: Option<StratifiedOrdering>,
    /// Balance ownership of partition-boundary points during distribution.
    pub balance_boundary_ownership: bool,
    /// Synchronize section data onto ghost points during distribution.
    pub synchronize_sections: bool,
}

impl Default for MeshDMOptions {
    fn default() -> Self {
        Self {
            pre_refine: 0,
            distribute: false,
            distribute_overlap: 1,
            post_refine: 0,
            localize_coordinates: false,
            check_symmetry: false,
            check_skeleton: false,
            check_faces: false,
            check_geometry: false,
            check_all: false,
            reorder_section: None,
            balance_boundary_ownership: false,
            synchronize_sections: true,
        }
    }
}

impl MeshDMOptions {
    /// Convert the distribution-related subset to the lower-level config type.
    pub fn distribution_config(&self) -> DistributionConfig {
        DistributionConfig {
            overlap_depth: self.distribute_overlap,
            synchronize_sections: self.synchronize_sections || self.localize_coordinates,
            balance_boundary_ownership: self.balance_boundary_ownership,
        }
    }
}

/// Builder for [`MeshDM`].
#[derive(Debug)]
pub struct MeshDMBuilder<V, St = VecStorage<V>, CtSt = VecStorage<CellType>>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    mesh_data: MeshData<MeshSieve, V, St, CtSt>,
    options: MeshDMOptions,
}

impl<V, St, CtSt> MeshDMBuilder<V, St, CtSt>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    /// Start a builder from an existing oriented mesh topology.
    pub fn new(topology: MeshSieve) -> Self {
        Self {
            mesh_data: MeshData::new(topology),
            options: MeshDMOptions::default(),
        }
    }

    /// Start a builder from a full mesh-data container.
    pub fn from_mesh_data(mesh_data: MeshData<MeshSieve, V, St, CtSt>) -> Self {
        Self {
            mesh_data,
            options: MeshDMOptions::default(),
        }
    }

    /// Replace the full options object.
    pub fn options(mut self, options: MeshDMOptions) -> Self {
        self.options = options;
        self
    }

    /// Mutate options in-place.
    pub fn configure(mut self, f: impl FnOnce(&mut MeshDMOptions)) -> Self {
        f(&mut self.options);
        self
    }

    /// Attach coordinate metadata.
    pub fn coordinates(mut self, coordinates: Coordinates<V, St>) -> Self {
        self.mesh_data.coordinates = Some(coordinates);
        self
    }

    /// Attach labels.
    pub fn labels(mut self, labels: LabelSet) -> Self {
        self.mesh_data.labels = Some(labels);
        self
    }

    /// Attach cell types.
    pub fn cell_types(mut self, cell_types: Section<CellType, CtSt>) -> Self {
        self.mesh_data.cell_types = Some(cell_types);
        self
    }

    /// Attach discretization metadata.
    pub fn discretization(mut self, discretization: Discretization) -> Self {
        self.mesh_data.discretization = Some(discretization);
        self
    }

    /// Attach a named scalar section.
    pub fn section(mut self, name: impl Into<String>, section: Section<V, St>) -> Self {
        self.mesh_data.sections.insert(name.into(), section);
        self
    }

    /// Build the DM and run serial setup checks/reordering requested in options.
    pub fn build(self) -> Result<MeshDM<V, St, CtSt>, MeshSieveError> {
        let mut dm = MeshDM::from_mesh_data_with_options(self.mesh_data, self.options);
        dm.setup_serial()?;
        Ok(dm)
    }
}

/// Minimal vector object created by a [`MeshDM`] section.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshVector<V> {
    /// Optional section name for solver diagnostics.
    pub section: Option<String>,
    /// Contiguous values in local or global numbering order.
    pub values: Vec<V>,
}

/// Matrix/preallocation graph derived from a DM adjacency graph and section map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreallocationGraph {
    /// CSR offsets into [`Self::adjncy`].
    pub xadj: Vec<usize>,
    /// CSR adjacency by point index.
    pub adjncy: Vec<usize>,
    /// Point ordering represented by CSR rows.
    pub order: Vec<PointId>,
    /// Number of point-neighbor blocks for each row.
    pub row_nnz: Vec<usize>,
}

/// Distribution state owned by a [`MeshDM`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshDMDistribution {
    /// Point owners indexed by `PointId - 1`.
    pub point_owners: Vec<usize>,
    /// Cell partition assignment used to create this local DM.
    pub cell_parts: Vec<usize>,
    /// Rank that owns this local DM state.
    pub rank: usize,
    /// Communicator size at distribution time.
    pub size: usize,
}

/// DMPLEX-like facade owning topology, coordinates, labels, sections,
/// discretization metadata, distribution state, and solver numbering maps.
#[derive(Debug)]
pub struct MeshDM<V, St = VecStorage<V>, CtSt = VecStorage<CellType>>
where
    St: Storage<V>,
    CtSt: Storage<CellType>,
{
    mesh_data: MeshData<MeshSieve, V, St, CtSt>,
    options: MeshDMOptions,
    ownership: Option<PointOwnership>,
    overlap: Option<Overlap>,
    global_sections: BTreeMap<String, LocalToGlobalMap>,
    distribution: Option<MeshDMDistribution>,
}

impl<V, St, CtSt> MeshDM<V, St, CtSt>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
    CtSt: Storage<CellType> + Clone,
{
    /// Create a builder from topology.
    pub fn builder(topology: MeshSieve) -> MeshDMBuilder<V, St, CtSt> {
        MeshDMBuilder::new(topology)
    }

    /// Wrap existing mesh data with default DM options.
    pub fn from_mesh_data(mesh_data: MeshData<MeshSieve, V, St, CtSt>) -> Self {
        Self::from_mesh_data_with_options(mesh_data, MeshDMOptions::default())
    }

    /// Wrap existing mesh data with explicit DM options.
    pub fn from_mesh_data_with_options(
        mesh_data: MeshData<MeshSieve, V, St, CtSt>,
        options: MeshDMOptions,
    ) -> Self {
        Self {
            mesh_data,
            options,
            ownership: None,
            overlap: None,
            global_sections: BTreeMap::new(),
            distribution: None,
        }
    }

    /// Consume the facade, returning the owned lower-level mesh data.
    pub fn into_mesh_data(self) -> MeshData<MeshSieve, V, St, CtSt> {
        self.mesh_data
    }

    /// Borrow the full lower-level mesh data container.
    pub fn mesh_data(&self) -> &MeshData<MeshSieve, V, St, CtSt> {
        &self.mesh_data
    }

    /// Borrow the topology.
    pub fn topology(&self) -> &MeshSieve {
        &self.mesh_data.sieve
    }

    /// Mutably borrow the topology.
    pub fn topology_mut(&mut self) -> &mut MeshSieve {
        &mut self.mesh_data.sieve
    }

    /// Borrow labels, if present.
    pub fn labels(&self) -> Option<&LabelSet> {
        self.mesh_data.labels.as_ref()
    }

    /// Borrow coordinates, if present.
    pub fn coordinates(&self) -> Option<&Coordinates<V, St>> {
        self.mesh_data.coordinates.as_ref()
    }

    /// Borrow a named local section.
    pub fn section(&self, name: &str) -> Option<&Section<V, St>> {
        self.mesh_data.sections.get(name)
    }

    /// Mutably borrow or create labels.
    pub fn labels_mut_or_insert(&mut self) -> &mut LabelSet {
        self.mesh_data.labels.get_or_insert_with(LabelSet::new)
    }

    /// Insert or replace a named local section.
    pub fn insert_section(
        &mut self,
        name: impl Into<String>,
        section: Section<V, St>,
    ) -> Option<Section<V, St>> {
        self.mesh_data.sections.insert(name.into(), section)
    }

    /// Borrow cell type metadata, if present.
    pub fn cell_types(&self) -> Option<&Section<CellType, CtSt>> {
        self.mesh_data.cell_types.as_ref()
    }

    /// Borrow discretization metadata, if present.
    pub fn discretization(&self) -> Option<&Discretization> {
        self.mesh_data.discretization.as_ref()
    }

    /// Borrow ownership metadata, if this DM has been distributed or numbered.
    pub fn ownership(&self) -> Option<&PointOwnership> {
        self.ownership.as_ref()
    }

    /// Borrow overlap/SF-like state, if this DM has been distributed.
    pub fn overlap(&self) -> Option<&Overlap> {
        self.overlap.as_ref()
    }

    /// Borrow distribution metadata, if this DM has been distributed.
    pub fn distribution(&self) -> Option<&MeshDMDistribution> {
        self.distribution.as_ref()
    }

    /// Borrow DM setup options.
    pub fn options(&self) -> &MeshDMOptions {
        &self.options
    }

    /// Run local setup actions that do not require a partitioner/communicator.
    pub fn setup_serial(&mut self) -> Result<(), MeshSieveError> {
        if let Some(ordering) = self.options.reorder_section {
            self.reorder_sections(ordering)?;
        }
        self.run_requested_checks()?;
        Ok(())
    }

    /// Run topology/geometry checks requested by [`MeshDMOptions`].
    pub fn run_requested_checks(&mut self) -> Result<(), MeshSieveError> {
        let check_all = self.options.check_all;
        let check_options = MeshCheckOptions {
            check_symmetry: check_all || self.options.check_symmetry,
            check_skeleton: check_all || self.options.check_skeleton,
            check_faces: check_all || self.options.check_faces,
            check_geometry: check_all || self.options.check_geometry,
            check_overlap: check_all,
            check_ownership: check_all,
            check_sections: check_all,
        };

        if check_options.check_faces {
            let cells = self.height_stratum(0)?;
            let _ = self.cell_adjacency_graph(cells, Default::default(), AdjacencyWeighting::None);
        }

        run_mesh_checks(
            &mut self.mesh_data.sieve,
            self.mesh_data.cell_types.as_ref(),
            self.mesh_data.coordinates.as_ref(),
            self.ownership.as_ref(),
            self.overlap.as_ref(),
            self.mesh_data.sections.values(),
            check_options,
        )
    }

    /// Return points in a height stratum (height 0 are cells in DMPLEX terms).
    pub fn height_stratum(&self, height: u32) -> Result<Vec<PointId>, MeshSieveError> {
        let strata = compute_strata(&self.mesh_data.sieve)?;
        Ok(strata
            .strata
            .get(height as usize)
            .cloned()
            .unwrap_or_default())
    }

    /// Return points in a depth stratum (depth 0 are vertices in DMPLEX terms).
    pub fn depth_stratum(&self, depth: u32) -> Result<Vec<PointId>, MeshSieveError> {
        let strata = compute_strata(&self.mesh_data.sieve)?;
        let mut points: Vec<_> = strata
            .depth
            .iter()
            .filter_map(|(&point, &d)| (d == depth).then_some(point))
            .collect();
        points.sort_unstable();
        Ok(points)
    }

    /// Build a dual graph for the provided cells.
    pub fn dual_graph(&self, cells: impl IntoIterator<Item = PointId>) -> DualGraph {
        build_dual(&self.mesh_data.sieve, cells)
    }

    /// Build a cell adjacency/preallocation graph for the provided cells.
    pub fn cell_adjacency_graph(
        &self,
        cells: impl IntoIterator<Item = PointId>,
        opts: crate::algs::adjacency_graph::CellAdjacencyOpts,
        weighting: AdjacencyWeighting,
    ) -> MeshGraph {
        cell_adjacency_graph_with_cells(&self.mesh_data.sieve, cells, opts, weighting)
    }

    /// Build a matrix preallocation graph from cell adjacency.
    pub fn matrix_preallocation_graph(
        &self,
        cells: impl IntoIterator<Item = PointId>,
        opts: crate::algs::adjacency_graph::CellAdjacencyOpts,
    ) -> PreallocationGraph {
        let graph = self.cell_adjacency_graph(cells, opts, AdjacencyWeighting::None);
        let row_nnz = graph
            .xadj
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .collect();
        PreallocationGraph {
            xadj: graph.xadj,
            adjncy: graph.adjncy,
            order: graph.order,
            row_nnz,
        }
    }

    /// Create a zero-initialized local vector matching a named section.
    pub fn create_local_vector(&self, section_name: &str) -> Result<MeshVector<V>, MeshSieveError> {
        let section = self.mesh_data.sections.get(section_name).ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: section_name.to_string(),
            }
        })?;
        Ok(MeshVector {
            section: Some(section_name.to_string()),
            values: vec![V::default(); section.atlas().total_len()],
        })
    }

    /// Build and store global numbering maps for all named local sections.
    pub fn build_global_sections<C>(&mut self, comm: &C) -> Result<(), MeshSieveError>
    where
        C: Communicator + Sync,
    {
        self.ensure_serial_ownership(comm.rank())?;
        let ownership = self.ownership.as_ref().expect("ownership inserted");
        let empty_overlap = Overlap::default();
        let overlap = self.overlap.as_ref().unwrap_or(&empty_overlap);
        let mut maps = BTreeMap::new();
        for (name, section) in &self.mesh_data.sections {
            maps.insert(
                name.clone(),
                LocalToGlobalMap::from_section_with_ownership(
                    section,
                    overlap,
                    ownership,
                    comm,
                    comm.rank(),
                )?,
            );
        }
        self.global_sections = maps;
        Ok(())
    }

    /// Borrow a stored global section map.
    pub fn global_section(&self, name: &str) -> Option<&LocalToGlobalMap> {
        self.global_sections.get(name)
    }

    /// Create a zero-initialized global vector matching a named global section.
    pub fn create_global_vector(
        &self,
        section_name: &str,
    ) -> Result<MeshVector<V>, MeshSieveError> {
        let map = self.global_sections.get(section_name).ok_or_else(|| {
            MeshSieveError::MissingSectionName {
                name: section_name.to_string(),
            }
        })?;
        Ok(MeshVector {
            section: Some(section_name.to_string()),
            values: global_vector_for_map(map),
        })
    }

    /// Create a constrained section for a field using label constraints.
    pub fn create_constrained_section_from_labels(
        &self,
        field_name: &str,
        point_dofs: &[(PointId, usize)],
        constraints: &[LabelConstraintSpec],
    ) -> Result<ConstrainedSection<V, St>, MeshSieveError> {
        if let Some(discretization) = self.discretization() {
            if discretization.field(field_name).is_none() {
                return Err(MeshSieveError::MissingSectionName {
                    name: field_name.to_string(),
                });
            }
        }
        let labels = self
            .labels()
            .ok_or_else(|| MeshSieveError::MissingSectionName {
                name: "labels".to_string(),
            })?;
        constrained_section_from_label_specs(point_dofs, labels, constraints)
    }

    /// Distribute this DM through the lower-level distribution pipeline and
    /// return the local DM for the calling rank.
    pub fn distribute_with<P, C>(
        &self,
        cells: &[PointId],
        partitioner: &P,
        comm: &C,
    ) -> Result<Self, MeshSieveError>
    where
        P: CellPartitioner<MeshSieve>,
        C: Communicator + Sync,
        V: Send + PartialEq + bytemuck::Pod + 'static,
    {
        let distributed = distribute_with_overlap(
            &self.mesh_data,
            cells,
            partitioner,
            self.options.distribution_config(),
            comm,
        )?;
        Ok(Self::from_distributed(
            distributed,
            self.options.clone(),
            comm.rank(),
            comm.size(),
        ))
    }

    /// Complete/synchronize all registered fields through the owned overlap/SF state.
    pub fn distribute_fields<C>(&mut self, comm: &C) -> Result<(), MeshSieveError>
    where
        C: Communicator + Sync,
        V: Send + PartialEq + bytemuck::Pod + 'static,
    {
        let Some(ownership) = &self.ownership else {
            return Ok(());
        };
        let Some(overlap) = &self.overlap else {
            return Ok(());
        };
        let sf =
            crate::algs::point_sf::PointSF::with_ownership(overlap, ownership, comm, comm.rank());
        sf.validate()?;
        if let Some(coords) = &mut self.mesh_data.coordinates {
            sf.complete_section(coords.section_mut())?;
            if let Some(high_order) = coords.high_order_mut() {
                sf.complete_section(high_order.section_mut())?;
            }
        }
        for section in self.mesh_data.sections.values_mut() {
            sf.complete_section(section)?;
        }
        Ok(())
    }

    fn from_distributed(
        data: DistributedMeshData<V, St, CtSt>,
        options: MeshDMOptions,
        rank: usize,
        size: usize,
    ) -> Self {
        let mesh_data = MeshData {
            sieve: data.sieve,
            coordinates: data.coordinates,
            sections: data.sections,
            mixed_sections: data.mixed_sections,
            labels: data.labels,
            cell_types: data.cell_types,
            discretization: data.discretization,
        };
        Self {
            mesh_data,
            options,
            ownership: Some(data.ownership),
            overlap: Some(data.overlap),
            global_sections: BTreeMap::new(),
            distribution: Some(MeshDMDistribution {
                point_owners: data.point_owners,
                cell_parts: data.cell_parts,
                rank,
                size,
            }),
        }
    }

    fn ensure_serial_ownership(&mut self, rank: usize) -> Result<(), MeshSieveError> {
        if self.ownership.is_some() {
            return Ok(());
        }
        let mut ownership = PointOwnership::default();
        for point in self.mesh_data.sieve.points() {
            ownership.set(point, rank, false)?;
        }
        self.ownership = Some(ownership);
        Ok(())
    }

    fn reorder_sections(&mut self, ordering: StratifiedOrdering) -> Result<(), MeshSieveError> {
        let permutation = stratified_permutation(&self.mesh_data.sieve, ordering)?;
        for section in self.mesh_data.sections.values_mut() {
            *section = reorder_section_by_points(section, &permutation)?;
        }
        if let Some(coords) = &mut self.mesh_data.coordinates {
            let topological_dimension = coords.topological_dimension();
            let embedding_dimension = coords.embedding_dimension();
            let section = reorder_section_by_points(coords.section(), &permutation)?;
            *coords =
                Coordinates::from_section(topological_dimension, embedding_dimension, section)?;
        }
        if let Some(cell_types) = &mut self.mesh_data.cell_types {
            *cell_types = reorder_section_by_points(cell_types, &permutation)?;
        }
        Ok(())
    }
}

fn reorder_section_by_points<V, St>(
    section: &Section<V, St>,
    order: &[PointId],
) -> Result<Section<V, St>, MeshSieveError>
where
    V: Clone + Default,
    St: Storage<V> + Clone,
{
    let mut atlas = Atlas::default();
    for &point in order {
        if let Some((_offset, len)) = section.atlas().get(point) {
            atlas.try_insert(point, len)?;
        }
    }
    for point in section.atlas().points() {
        if !atlas.contains(point) {
            let (_offset, len) = section
                .atlas()
                .get(point)
                .ok_or(MeshSieveError::PointNotInAtlas(point))?;
            atlas.try_insert(point, len)?;
        }
    }
    let mut reordered = Section::new(atlas);
    for point in section.atlas().points() {
        let values = section.try_restrict(point)?.to_vec();
        reordered.try_set(point, &values)?;
    }
    Ok(reordered)
}
