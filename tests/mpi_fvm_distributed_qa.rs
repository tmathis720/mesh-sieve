#![cfg(feature = "mpi-support")]

mod util;

use mesh_sieve::algs::communicator::{Communicator, MpiComm};
use mesh_sieve::algs::completion::section_completion::complete_section;
use mesh_sieve::data::{atlas::Atlas, section::Section, storage::VecStorage};
use mesh_sieve::overlap::{delta::CopyDelta, overlap::Overlap};
use mesh_sieve::topology::point::PointId;

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

fn run_layout(comm: &MpiComm, swap_partitioning: bool) -> (PointId, f64, f64) {
    let rank = comm.rank();
    let mut ov = Overlap::default();
    let mut atlas = Atlas::default();

    let rank0_owns_left = !swap_partitioning;
    let rank_owns_left = (rank == 0 && rank0_owns_left) || (rank == 1 && !rank0_owns_left);

    let (cell_local, face_if_local, face_bdry_local) = if rank_owns_left {
        ov.try_add_link(p(1), 1 - rank, p(101)).unwrap();
        ov.try_add_link(p(21), 1 - rank, p(201)).unwrap();
        for id in [1, 10, 21, 101, 201] {
            atlas.try_insert(p(id), 1).unwrap();
        }
        (p(1), p(21), p(10))
    } else {
        ov.try_add_link(p(101), 1 - rank, p(1)).unwrap();
        ov.try_add_link(p(201), 1 - rank, p(21)).unwrap();
        for id in [101, 201, 20, 1, 21] {
            atlas.try_insert(p(id), 1).unwrap();
        }
        (p(101), p(201), p(20))
    };

    let mut cell = Section::<f64, VecStorage<f64>>::new(atlas.clone());
    let mut face = Section::<f64, VecStorage<f64>>::new(atlas);
    if rank_owns_left {
        cell.try_set(cell_local, &[1.0]).unwrap();
        face.try_set(face_if_local, &[5.0]).unwrap();
        face.try_set(face_bdry_local, &[3.0]).unwrap();
    } else {
        cell.try_set(cell_local, &[2.0]).unwrap();
        face.try_set(face_if_local, &[5.0]).unwrap();
        face.try_set(face_bdry_local, &[7.0]).unwrap();
    }

    complete_section::<f64, _, CopyDelta, _>(&mut cell, &ov, comm, rank).unwrap();
    complete_section::<f64, _, CopyDelta, _>(&mut face, &ov, comm, rank).unwrap();

    let interface = face.try_restrict(face_if_local).unwrap()[0];
    let residual = cell.try_restrict(cell_local).unwrap()[0]
        + face.try_restrict(face_bdry_local).unwrap()[0]
        + if rank_owns_left {
            interface
        } else {
            -interface
        };
    let signed_interface = if rank_owns_left {
        interface
    } else {
        -interface
    };

    (cell_local, residual, signed_interface)
}

#[test]
fn mpi_fvm_modes_parity_and_partitioning_invariance() {
    if util::mpi_launcher_world_size() != Some(2) {
        return;
    }
    let comm = MpiComm::new().expect("MPI init");
    assert_eq!(comm.size(), 2);

    let (cell_a, residual_a, iface_a) = run_layout(&comm, false);
    let (cell_b, residual_b, iface_b) = run_layout(&comm, true);

    let mut residuals = vec![0u8; 8 * comm.size()];
    comm.allgather(&residual_a.to_le_bytes(), &mut residuals);
    let global_residual: f64 = residuals
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .sum();
    assert!((global_residual - 13.0).abs() < 1e-12);

    let mut interfaces = vec![0u8; 8 * comm.size()];
    comm.allgather(&iface_a.to_le_bytes(), &mut interfaces);
    let interface_balance: f64 = interfaces
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .sum();
    assert!(interface_balance.abs() < 1e-12);

    let packed = [
        cell_a.get().to_le_bytes(),
        residual_a.to_le_bytes(),
        cell_b.get().to_le_bytes(),
        residual_b.to_le_bytes(),
    ]
    .concat();
    let mut gathered = vec![0u8; 32 * comm.size()];
    comm.allgather(&packed, &mut gathered);

    for chunk in gathered.chunks_exact(32) {
        let a_id = u64::from_le_bytes(chunk[0..8].try_into().unwrap());
        let a_r = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
        let b_id = u64::from_le_bytes(chunk[16..24].try_into().unwrap());
        let b_r = f64::from_le_bytes(chunk[24..32].try_into().unwrap());
        assert_eq!(a_id, b_id, "cell assignment changed unexpectedly");
        assert!(
            (a_r - b_r).abs() < 1e-12,
            "cell residual drift on cell {a_id}"
        );
    }

    // also ensure cancellation still holds in swapped layout
    let mut interfaces_swapped = vec![0u8; 8 * comm.size()];
    comm.allgather(&iface_b.to_le_bytes(), &mut interfaces_swapped);
    let interface_balance_swapped: f64 = interfaces_swapped
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .sum();
    assert!(interface_balance_swapped.abs() < 1e-12);
}
