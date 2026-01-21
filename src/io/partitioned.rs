//! Partitioned mesh I/O helpers with communicator-aware coordination.

use crate::algs::communicator::Communicator;
use crate::algs::communicator::Wait;
use crate::io::{MeshData, SieveSectionReader, SieveSectionWriter};
use crate::mesh_error::MeshSieveError;
use crate::overlap::overlap::Overlap;
use crate::topology::point::PointId;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

const PARTITIONED_METADATA_VERSION: u32 = 1;

/// Metadata describing a partitioned mesh piece.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitionedMeshMetadata {
    /// Metadata format version.
    pub version: u32,
    /// Rank that owns this piece.
    pub rank: usize,
    /// Total number of ranks in the communicator.
    pub size: usize,
    /// Optional overlap information for this rank.
    pub overlap: Option<OverlapMetadata>,
}

impl PartitionedMeshMetadata {
    /// Build metadata for a rank.
    pub fn new(rank: usize, size: usize, overlap: Option<OverlapMetadata>) -> Self {
        Self {
            version: PARTITIONED_METADATA_VERSION,
            rank,
            size,
            overlap,
        }
    }
}

/// Serializable overlap metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlapMetadata {
    /// Neighbor rank link lists.
    pub neighbors: Vec<NeighborLinks>,
}

/// Links to a neighbor rank.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeighborLinks {
    /// Neighbor rank id.
    pub rank: usize,
    /// Links (local, remote) pairs; `remote` may be unresolved.
    pub links: Vec<OverlapLink>,
}

/// One overlap link pair for serialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlapLink {
    /// Local point id (raw value).
    pub local: u64,
    /// Remote point id (raw value) if resolved.
    pub remote: Option<u64>,
}

/// Combined mesh data and overlap metadata for a partitioned piece.
pub struct PartitionedMeshData<S, V, St, CtSt>
where
    St: crate::data::storage::Storage<V>,
    CtSt: crate::data::storage::Storage<crate::topology::cell_type::CellType>,
{
    /// Local mesh piece.
    pub mesh: MeshData<S, V, St, CtSt>,
    /// Optional overlap graph reconstructed from metadata.
    pub overlap: Option<Overlap>,
    /// Metadata describing the piece.
    pub metadata: PartitionedMeshMetadata,
}

/// Partitioned mesh write policy.
#[derive(Clone, Copy, Debug)]
pub enum GatherPolicy {
    /// Always write per-rank mesh pieces with metadata.
    LocalPieces,
    /// Gather to root if total payload size is below the threshold (bytes).
    GatherToRoot { max_total_bytes: usize },
}

/// Result of a partitioned mesh write.
#[derive(Clone, Debug)]
pub enum PartitionedMeshWriteOutcome {
    /// Wrote per-rank mesh files and metadata.
    LocalPieces,
    /// Wrote a gathered bundle file on root.
    Bundle { path: PathBuf },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PartitionedMeshBundle {
    version: u32,
    size: usize,
    pieces: Vec<PartitionedMeshBundlePiece>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PartitionedMeshBundlePiece {
    rank: usize,
    metadata: PartitionedMeshMetadata,
    mesh: Vec<u8>,
}

/// Convert an [`Overlap`] into serializable metadata.
pub fn overlap_to_metadata(overlap: &Overlap) -> OverlapMetadata {
    let mut neighbors: Vec<NeighborLinks> = overlap
        .neighbor_ranks()
        .map(|rank| NeighborLinks {
            rank,
            links: overlap
                .links_to_sorted(rank)
                .into_iter()
                .map(|(local, remote)| OverlapLink {
                    local: local.get(),
                    remote: remote.map(|r| r.get()),
                })
                .collect(),
        })
        .collect();
    neighbors.sort_by_key(|entry| entry.rank);
    OverlapMetadata { neighbors }
}

/// Reconstruct an [`Overlap`] from metadata.
pub fn overlap_from_metadata(meta: &OverlapMetadata) -> Result<Overlap, MeshSieveError> {
    let mut overlap = Overlap::new();
    for neighbor in &meta.neighbors {
        for link in &neighbor.links {
            let local = PointId::new(link.local).map_err(|_| {
                MeshSieveError::MeshIoParse(format!("invalid local point id {}", link.local))
            })?;
            match link.remote {
                Some(remote_id) => {
                    let remote = PointId::new(remote_id).map_err(|_| {
                        MeshSieveError::MeshIoParse(format!(
                            "invalid remote point id {}",
                            remote_id
                        ))
                    })?;
                    overlap.try_add_link(local, neighbor.rank, remote)?;
                }
                None => {
                    overlap.try_add_link_structural_one(local, neighbor.rank)?;
                }
            }
        }
    }
    Ok(overlap)
}

/// Write a per-rank mesh piece and metadata in the partitioned format.
pub fn write_local_piece_with_metadata<W, C>(
    writer: &W,
    dir: impl AsRef<Path>,
    prefix: &str,
    mesh: &MeshData<W::Sieve, W::Value, W::Storage, W::CellStorage>,
    overlap: Option<&Overlap>,
    comm: &C,
) -> Result<PartitionedMeshMetadata, MeshSieveError>
where
    W: SieveSectionWriter,
    C: Communicator,
{
    let rank = comm.rank();
    let size = comm.size();
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;

    let mesh_path = local_mesh_path(dir, prefix, rank);
    let meta_path = local_meta_path(dir, prefix, rank);

    let mut mesh_bytes = Vec::new();
    writer.write(&mut mesh_bytes, mesh)?;
    fs::write(&mesh_path, mesh_bytes)?;

    let overlap_meta = overlap.map(overlap_to_metadata);
    let metadata = PartitionedMeshMetadata::new(rank, size, overlap_meta);
    let meta_bytes = serde_json::to_vec_pretty(&metadata)
        .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;
    fs::write(&meta_path, meta_bytes)?;

    Ok(metadata)
}

/// Write a partitioned mesh, optionally gathering to root for small meshes.
pub fn write_partitioned_mesh<W, C>(
    writer: &W,
    dir: impl AsRef<Path>,
    prefix: &str,
    mesh: &MeshData<W::Sieve, W::Value, W::Storage, W::CellStorage>,
    overlap: Option<&Overlap>,
    comm: &C,
    policy: GatherPolicy,
) -> Result<PartitionedMeshWriteOutcome, MeshSieveError>
where
    W: SieveSectionWriter,
    C: Communicator,
{
    let dir = dir.as_ref();
    let rank = comm.rank();
    let size = comm.size();

    let mut mesh_bytes = Vec::new();
    writer.write(&mut mesh_bytes, mesh)?;
    let overlap_meta = overlap.map(overlap_to_metadata);
    let metadata = PartitionedMeshMetadata::new(rank, size, overlap_meta);
    let metadata_bytes = serde_json::to_vec(&metadata)
        .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;

    let local_total = (mesh_bytes.len() + metadata_bytes.len()) as u64;
    let mut total_bytes = [local_total];
    comm.allreduce_sum(&mut total_bytes);
    let total_bytes = total_bytes[0] as usize;

    let gather = match policy {
        GatherPolicy::LocalPieces => false,
        GatherPolicy::GatherToRoot { max_total_bytes } => total_bytes <= max_total_bytes,
    };

    if gather {
        let tags = comm.reserve_tag_range(2)?;
        let sizes_tag = tags.as_u16();
        let data_tag = tags.offset(1).as_u16();
        let root = 0;

        let mut pieces = Vec::with_capacity(size);
        if rank == root {
            pieces.push(PartitionedMeshBundlePiece {
                rank,
                metadata: metadata.clone(),
                mesh: mesh_bytes.clone(),
            });
            for peer in 0..size {
                if peer == root {
                    continue;
                }
                let mut size_buf = vec![0u8; 16];
                let recv = comm.irecv(peer, sizes_tag, &mut size_buf);
                let data_len = recv
                    .wait()
                    .and_then(|buf| {
                        if buf.len() == 16 {
                            let meta_len = u64::from_le_bytes(buf[0..8].try_into().ok()?);
                            let mesh_len = u64::from_le_bytes(buf[8..16].try_into().ok()?);
                            Some((meta_len as usize, mesh_len as usize))
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| {
                        MeshSieveError::MeshIoParse("failed to receive size header".into())
                    })?;

                let mut data_buf = vec![0u8; data_len.0 + data_len.1];
                let recv = comm.irecv(peer, data_tag, &mut data_buf);
                let payload = recv.wait().ok_or_else(|| {
                    MeshSieveError::MeshIoParse("failed to receive mesh payload".into())
                })?;
                if payload.len() != data_buf.len() {
                    return Err(MeshSieveError::MeshIoParse(
                        "mesh payload length mismatch".into(),
                    ));
                }
                let meta_bytes = &payload[..data_len.0];
                let mesh_bytes = payload[data_len.0..].to_vec();
                let meta: PartitionedMeshMetadata = serde_json::from_slice(meta_bytes)
                    .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;
                pieces.push(PartitionedMeshBundlePiece {
                    rank: peer,
                    metadata: meta,
                    mesh: mesh_bytes,
                });
            }

            pieces.sort_by_key(|piece| piece.rank);
            fs::create_dir_all(dir)?;
            let bundle = PartitionedMeshBundle {
                version: PARTITIONED_METADATA_VERSION,
                size,
                pieces,
            };
            let bundle_path = bundle_path(dir, prefix);
            let bundle_bytes = serde_json::to_vec_pretty(&bundle)
                .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;
            fs::write(&bundle_path, bundle_bytes)?;
            Ok(PartitionedMeshWriteOutcome::Bundle { path: bundle_path })
        } else {
            let mut size_buf = Vec::with_capacity(16);
            size_buf.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
            size_buf.extend_from_slice(&(mesh_bytes.len() as u64).to_le_bytes());
            let _ = comm.isend(root, sizes_tag, &size_buf).wait();

            let mut payload = Vec::with_capacity(metadata_bytes.len() + mesh_bytes.len());
            payload.extend_from_slice(&metadata_bytes);
            payload.extend_from_slice(&mesh_bytes);
            let _ = comm.isend(root, data_tag, &payload).wait();

            Ok(PartitionedMeshWriteOutcome::Bundle {
                path: bundle_path(dir, prefix),
            })
        }
    } else {
        write_local_piece_with_metadata(writer, dir, prefix, mesh, overlap, comm)?;
        Ok(PartitionedMeshWriteOutcome::LocalPieces)
    }
}

/// Read a partitioned mesh, using the bundle format if available.
pub fn read_partitioned_mesh<R, C>(
    reader: &R,
    dir: impl AsRef<Path>,
    prefix: &str,
    comm: &C,
) -> Result<PartitionedMeshData<R::Sieve, R::Value, R::Storage, R::CellStorage>, MeshSieveError>
where
    R: SieveSectionReader,
    C: Communicator,
{
    let dir = dir.as_ref();
    let rank = comm.rank();
    let size = comm.size();
    let bundle_path = bundle_path(dir, prefix);

    let mut bundle_flag = [0u8; 1];
    if rank == 0 {
        bundle_flag[0] = bundle_path.exists() as u8;
    }
    comm.broadcast(0, &mut bundle_flag);
    if bundle_flag[0] == 1 {
        read_partitioned_bundle(reader, &bundle_path, comm)
    } else {
        read_partitioned_local_piece(reader, dir, prefix, rank, size)
    }
}

fn read_partitioned_bundle<R, C>(
    reader: &R,
    bundle_path: &Path,
    comm: &C,
) -> Result<PartitionedMeshData<R::Sieve, R::Value, R::Storage, R::CellStorage>, MeshSieveError>
where
    R: SieveSectionReader,
    C: Communicator,
{
    let rank = comm.rank();
    let size = comm.size();
    let tags = comm.reserve_tag_range(2)?;
    let sizes_tag = tags.as_u16();
    let data_tag = tags.offset(1).as_u16();
    let root = 0;

    let (metadata, mesh_bytes) = if rank == root {
        let bundle_bytes = fs::read(bundle_path)?;
        let bundle: PartitionedMeshBundle = serde_json::from_slice(&bundle_bytes)
            .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;
        if bundle.size != size {
            return Err(MeshSieveError::MeshIoParse(format!(
                "bundle size mismatch: expected {size}, found {}",
                bundle.size
            )));
        }
        let mut pieces = bundle.pieces;
        pieces.sort_by_key(|piece| piece.rank);
        for piece in &pieces {
            if piece.rank == root {
                continue;
            }
            let meta_bytes = serde_json::to_vec(&piece.metadata)
                .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;
            let mut size_buf = Vec::with_capacity(16);
            size_buf.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
            size_buf.extend_from_slice(&(piece.mesh.len() as u64).to_le_bytes());
            let _ = comm.isend(piece.rank, sizes_tag, &size_buf).wait();

            let mut payload = Vec::with_capacity(meta_bytes.len() + piece.mesh.len());
            payload.extend_from_slice(&meta_bytes);
            payload.extend_from_slice(&piece.mesh);
            let _ = comm.isend(piece.rank, data_tag, &payload).wait();
        }

        let my_piece = pieces
            .into_iter()
            .find(|piece| piece.rank == root)
            .ok_or_else(|| MeshSieveError::MeshIoParse("missing root piece".into()))?;
        (my_piece.metadata, my_piece.mesh)
    } else {
        let mut size_buf = vec![0u8; 16];
        let recv = comm.irecv(root, sizes_tag, &mut size_buf);
        let data_len = recv
            .wait()
            .and_then(|buf| {
                if buf.len() == 16 {
                    let meta_len = u64::from_le_bytes(buf[0..8].try_into().ok()?);
                    let mesh_len = u64::from_le_bytes(buf[8..16].try_into().ok()?);
                    Some((meta_len as usize, mesh_len as usize))
                } else {
                    None
                }
            })
            .ok_or_else(|| MeshSieveError::MeshIoParse("failed to receive size header".into()))?;

        let mut data_buf = vec![0u8; data_len.0 + data_len.1];
        let recv = comm.irecv(root, data_tag, &mut data_buf);
        let payload = recv.wait().ok_or_else(|| {
            MeshSieveError::MeshIoParse("failed to receive mesh payload".into())
        })?;
        if payload.len() != data_buf.len() {
            return Err(MeshSieveError::MeshIoParse(
                "mesh payload length mismatch".into(),
            ));
        }
        let meta_bytes = &payload[..data_len.0];
        let mesh_bytes = payload[data_len.0..].to_vec();
        let meta: PartitionedMeshMetadata = serde_json::from_slice(meta_bytes)
            .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;
        (meta, mesh_bytes)
    };

    let mesh = reader.read(Cursor::new(mesh_bytes))?;
    let overlap = metadata
        .overlap
        .as_ref()
        .map(overlap_from_metadata)
        .transpose()?;
    Ok(PartitionedMeshData {
        mesh,
        overlap,
        metadata,
    })
}

fn read_partitioned_local_piece<R>(
    reader: &R,
    dir: &Path,
    prefix: &str,
    rank: usize,
    size: usize,
) -> Result<PartitionedMeshData<R::Sieve, R::Value, R::Storage, R::CellStorage>, MeshSieveError>
where
    R: SieveSectionReader,
{
    let mesh_path = local_mesh_path(dir, prefix, rank);
    let meta_path = local_meta_path(dir, prefix, rank);

    let metadata_bytes = fs::read(meta_path)?;
    let metadata: PartitionedMeshMetadata = serde_json::from_slice(&metadata_bytes)
        .map_err(|e| MeshSieveError::MeshIoParse(e.to_string()))?;
    if metadata.rank != rank {
        return Err(MeshSieveError::MeshIoParse(format!(
            "metadata rank mismatch: expected {rank}, found {}",
            metadata.rank
        )));
    }
    if metadata.size != size {
        return Err(MeshSieveError::MeshIoParse(format!(
            "metadata size mismatch: expected {size}, found {}",
            metadata.size
        )));
    }

    let mesh_bytes = fs::read(mesh_path)?;
    let mesh = reader.read(Cursor::new(mesh_bytes))?;
    let overlap = metadata
        .overlap
        .as_ref()
        .map(overlap_from_metadata)
        .transpose()?;

    Ok(PartitionedMeshData {
        mesh,
        overlap,
        metadata,
    })
}

fn local_mesh_path(dir: &Path, prefix: &str, rank: usize) -> PathBuf {
    dir.join(format!("{prefix}.rank{rank}.mesh"))
}

fn local_meta_path(dir: &Path, prefix: &str, rank: usize) -> PathBuf {
    dir.join(format!("{prefix}.rank{rank}.meta.json"))
}

fn bundle_path(dir: &Path, prefix: &str) -> PathBuf {
    dir.join(format!("{prefix}.bundle.json"))
}
