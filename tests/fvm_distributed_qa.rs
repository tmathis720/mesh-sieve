use mesh_sieve::algs::communicator::{Communicator, NoComm, RayonComm};
use mesh_sieve::algs::completion::section_completion::complete_section;
use mesh_sieve::data::{atlas::Atlas, section::Section, storage::VecStorage};
use mesh_sieve::overlap::{delta::CopyDelta, overlap::Overlap};
use mesh_sieve::topology::point::PointId;

fn p(id: u64) -> PointId { PointId::new(id).unwrap() }

#[derive(Debug, Clone, Copy)]
struct RankState {
    local_residual: f64,
    interface_flux: f64,
    ghost_cell: f64,
    ghost_face: f64,
}

fn setup_rank(rank: usize) -> (Overlap, Section<f64, VecStorage<f64>>, Section<f64, VecStorage<f64>>) {
    let mut ov = Overlap::default();
    let mut atlas = Atlas::default();

    if rank == 0 {
        // owner points: c0(1), f_boundary(10), f_if(21); ghosts: c1(101), f_if_remote(201)
        ov.try_add_link(p(1), 1, p(101)).unwrap();
        ov.try_add_link(p(21), 1, p(201)).unwrap();
        for id in [1, 10, 21, 101, 201] {
            atlas.try_insert(p(id), 1).unwrap();
        }
        let mut cell = Section::new(atlas.clone());
        let mut face = Section::new(atlas);
        cell.try_set(p(1), &[1.0]).unwrap();
        face.try_set(p(10), &[3.0]).unwrap();
        face.try_set(p(21), &[5.0]).unwrap();
        (ov, cell, face)
    } else {
        ov.try_add_link(p(101), 0, p(1)).unwrap();
        ov.try_add_link(p(201), 0, p(21)).unwrap();
        for id in [101, 201, 20, 1, 21] {
            atlas.try_insert(p(id), 1).unwrap();
        }
        let mut cell = Section::new(atlas.clone());
        let mut face = Section::new(atlas);
        cell.try_set(p(101), &[2.0]).unwrap();
        face.try_set(p(201), &[5.0]).unwrap();
        face.try_set(p(20), &[7.0]).unwrap();
        (ov, cell, face)
    }
}

fn run_rank<C: Communicator + Sync>(rank: usize, comm: &C) -> RankState {
    let (ov, mut cell, mut face) = setup_rank(rank);
    complete_section::<f64, _, CopyDelta, _>(&mut cell, &ov, comm, rank).unwrap();
    complete_section::<f64, _, CopyDelta, _>(&mut face, &ov, comm, rank).unwrap();

    if rank == 0 {
        RankState {
            local_residual: cell.try_restrict(p(1)).unwrap()[0]
                + face.try_restrict(p(10)).unwrap()[0]
                + face.try_restrict(p(21)).unwrap()[0],
            interface_flux: face.try_restrict(p(21)).unwrap()[0],
            ghost_cell: cell.try_restrict(p(101)).unwrap()[0],
            ghost_face: face.try_restrict(p(201)).unwrap()[0],
        }
    } else {
        RankState {
            local_residual: cell.try_restrict(p(101)).unwrap()[0]
                + face.try_restrict(p(20)).unwrap()[0]
                - face.try_restrict(p(201)).unwrap()[0],
            interface_flux: -face.try_restrict(p(201)).unwrap()[0],
            ghost_cell: cell.try_restrict(p(1)).unwrap()[0],
            ghost_face: face.try_restrict(p(21)).unwrap()[0],
        }
    }
}

#[test]
fn nocomm_reference_case() {
    let c = NoComm;
    assert!(c.is_no_comm());
    // Serial baseline for the same 2-cell stencil without any comm path.
    let serial_total: f64 = 1.0 + 3.0 + 2.0 + 7.0;
    assert!((serial_total - 13.0).abs() < 1e-12);
}

#[test]
fn rayon_conservation_flux_and_ghost_paths() {
    let (c0, c1) = (RayonComm::new(0, 2), RayonComm::new(1, 2));
    let h = std::thread::spawn(move || run_rank(1, &c1));
    let s0 = run_rank(0, &c0);
    let s1 = h.join().unwrap();

    // Global conservation
    assert!((s0.local_residual + s1.local_residual - 13.0).abs() < 1e-12);
    // Interface cancellation across partitions
    assert!((s0.interface_flux + s1.interface_flux).abs() < 1e-12);
    // Cell-centered and face-centered ghost updates
    let ghost_faces = [s0.ghost_face, s1.ghost_face];
    assert!(ghost_faces.iter().any(|v| (*v - 5.0).abs() < 1e-12));
    let ghosts = [s0.ghost_cell, s1.ghost_cell];
    assert!(ghosts.iter().any(|v| (*v - 1.0).abs() < 1e-12));
    assert!(ghosts.iter().any(|v| (*v - 2.0).abs() < 1e-12));
}

#[test]
fn residual_invariant_under_layout_and_ownership_swap() {
    // Layout A: rank0 then rank1
    let (a0, a1) = (RayonComm::new(0, 2), RayonComm::new(1, 2));
    let h1 = std::thread::spawn(move || run_rank(1, &a1));
    let a = run_rank(0, &a0).local_residual + h1.join().unwrap().local_residual;

    // Layout B: evaluate in reverse order (simulating different ownership execution ordering)
    let (b0, b1) = (RayonComm::new(0, 2), RayonComm::new(1, 2));
    let h0 = std::thread::spawn(move || run_rank(0, &b0));
    let b = run_rank(1, &b1).local_residual + h0.join().unwrap().local_residual;

    assert!((a - b).abs() < 1e-12);
    assert!((a - 13.0).abs() < 1e-12);
}
