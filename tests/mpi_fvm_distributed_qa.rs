#![cfg(feature = "mpi-support")]

use mesh_sieve::algs::communicator::{Communicator, MpiComm};
use mesh_sieve::algs::completion::section_completion::complete_section;
use mesh_sieve::data::{atlas::Atlas, section::Section, storage::VecStorage};
use mesh_sieve::overlap::{delta::CopyDelta, overlap::Overlap};
use mesh_sieve::topology::point::PointId;

fn p(id: u64) -> PointId { PointId::new(id).unwrap() }

#[test]
fn mpi_fvm_conservation_flux_and_ghost_updates() {
    let comm = MpiComm::new().expect("MPI init");
    if comm.size() != 2 {
        return;
    }
    let rank = comm.rank();

    let mut ov = Overlap::default();
    let mut atlas = Atlas::default();
    let (cell_local, face_if_local, face_bdry_local, cell_ghost, face_if_ghost) = if rank == 0 {
        ov.try_add_link(p(1), 1, p(101)).unwrap();
        ov.try_add_link(p(21), 1, p(201)).unwrap();
        for id in [1, 10, 21, 101, 201] { atlas.try_insert(p(id), 1).unwrap(); }
        (p(1), p(21), p(10), p(101), p(201))
    } else {
        ov.try_add_link(p(101), 0, p(1)).unwrap();
        ov.try_add_link(p(201), 0, p(21)).unwrap();
        for id in [101, 201, 20, 1, 21] { atlas.try_insert(p(id), 1).unwrap(); }
        (p(101), p(201), p(20), p(1), p(21))
    };

    let mut cell = Section::<f64, VecStorage<f64>>::new(atlas.clone());
    let mut face = Section::<f64, VecStorage<f64>>::new(atlas);
    if rank == 0 {
        cell.try_set(cell_local, &[1.0]).unwrap();
        face.try_set(face_if_local, &[5.0]).unwrap();
        face.try_set(face_bdry_local, &[3.0]).unwrap();
    } else {
        cell.try_set(cell_local, &[2.0]).unwrap();
        face.try_set(face_if_local, &[5.0]).unwrap();
        face.try_set(face_bdry_local, &[7.0]).unwrap();
    }

    complete_section::<f64, _, CopyDelta, _>(&mut cell, &ov, &comm, rank).unwrap();
    complete_section::<f64, _, CopyDelta, _>(&mut face, &ov, &comm, rank).unwrap();

    let local_residual = if rank == 0 {
        cell.try_restrict(cell_local).unwrap()[0]
            + face.try_restrict(face_bdry_local).unwrap()[0]
            + face.try_restrict(face_if_local).unwrap()[0]
    } else {
        cell.try_restrict(cell_local).unwrap()[0]
            + face.try_restrict(face_bdry_local).unwrap()[0]
            - face.try_restrict(face_if_local).unwrap()[0]
    };
    let interface_signed = if rank == 0 {
        face.try_restrict(face_if_local).unwrap()[0]
    } else {
        -face.try_restrict(face_if_local).unwrap()[0]
    };

    let ghost_pair = [
        cell.try_restrict(cell_ghost).unwrap()[0].to_le_bytes(),
        face.try_restrict(face_if_ghost).unwrap()[0].to_le_bytes(),
    ];

    let mut residuals = vec![0u8; 8 * comm.size()];
    comm.allgather(&local_residual.to_le_bytes(), &mut residuals);
    let global_residual: f64 = residuals
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .sum();
    assert!((global_residual - 13.0).abs() < 1e-12);

    let mut interfaces = vec![0u8; 8 * comm.size()];
    comm.allgather(&interface_signed.to_le_bytes(), &mut interfaces);
    let interface_balance: f64 = interfaces
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .sum();
    assert!(interface_balance.abs() < 1e-12);

    // Check ghost update parity between both ranks.
    let packed = [ghost_pair[0], ghost_pair[1]].concat();
    let mut gathered = vec![0u8; 16 * comm.size()];
    comm.allgather(&packed, &mut gathered);
    let vals: Vec<(f64, f64)> = gathered
        .chunks_exact(16)
        .map(|chunk| {
            let c = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
            let f = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
            (c, f)
        })
        .collect();
    assert!(vals.iter().any(|(c, _)| (*c - 1.0).abs() < 1e-12));
    assert!(vals.iter().any(|(c, _)| (*c - 2.0).abs() < 1e-12));
    assert!(vals.iter().all(|(_, f)| (*f - 5.0).abs() < 1e-12));
}
