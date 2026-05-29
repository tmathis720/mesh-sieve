use mesh_sieve::algs::communicator::{Communicator, NoComm, RayonComm};
use mesh_sieve::algs::completion::section_completion::complete_section;
use mesh_sieve::data::{atlas::Atlas, section::Section, storage::VecStorage};
use mesh_sieve::overlap::{delta::CopyDelta, overlap::Overlap};
use mesh_sieve::topology::point::PointId;

fn p(id: u64) -> PointId {
    PointId::new(id).unwrap()
}

#[derive(Debug, Clone, Copy)]
struct RankState {
    local_cell: PointId,
    local_residual: f64,
    interface_flux: f64,
}

fn setup_rank(
    rank: usize,
    swap_partitioning: bool,
) -> (
    Overlap,
    Section<f64, VecStorage<f64>>,
    Section<f64, VecStorage<f64>>,
    PointId,
    PointId,
    PointId,
) {
    let mut ov = Overlap::default();
    let mut atlas = Atlas::default();

    let rank0_owns_left = !swap_partitioning;
    let rank_owns_left = (rank == 0 && rank0_owns_left) || (rank == 1 && !rank0_owns_left);

    if rank_owns_left {
        ov.try_add_link(p(1), 1 - rank, p(101)).unwrap();
        ov.try_add_link(p(21), 1 - rank, p(201)).unwrap();
        for id in [1, 10, 21, 101, 201] {
            atlas.try_insert(p(id), 1).unwrap();
        }
        let mut cell = Section::new(atlas.clone());
        let mut face = Section::new(atlas);
        cell.try_set(p(1), &[1.0]).unwrap();
        face.try_set(p(10), &[3.0]).unwrap();
        face.try_set(p(21), &[5.0]).unwrap();
        (ov, cell, face, p(1), p(10), p(21))
    } else {
        ov.try_add_link(p(101), 1 - rank, p(1)).unwrap();
        ov.try_add_link(p(201), 1 - rank, p(21)).unwrap();
        for id in [101, 201, 20, 1, 21] {
            atlas.try_insert(p(id), 1).unwrap();
        }
        let mut cell = Section::new(atlas.clone());
        let mut face = Section::new(atlas);
        cell.try_set(p(101), &[2.0]).unwrap();
        face.try_set(p(201), &[5.0]).unwrap();
        face.try_set(p(20), &[7.0]).unwrap();
        (ov, cell, face, p(101), p(20), p(201))
    }
}

fn run_rank<C: Communicator + Sync>(rank: usize, comm: &C, swap_partitioning: bool) -> RankState {
    let (ov, mut cell, mut face, local_cell, boundary_face, interface_face) =
        setup_rank(rank, swap_partitioning);
    complete_section::<f64, _, CopyDelta, _>(&mut cell, &ov, comm, rank).unwrap();
    complete_section::<f64, _, CopyDelta, _>(&mut face, &ov, comm, rank).unwrap();

    let interface_value = face.try_restrict(interface_face).unwrap()[0];
    let owns_left_cell = local_cell == p(1);

    RankState {
        local_cell,
        local_residual: cell.try_restrict(local_cell).unwrap()[0]
            + face.try_restrict(boundary_face).unwrap()[0]
            + if owns_left_cell {
                interface_value
            } else {
                -interface_value
            },
        interface_flux: if owns_left_cell {
            interface_value
        } else {
            -interface_value
        },
    }
}

fn run_rayon_pair(swap_partitioning: bool) -> (RankState, RankState) {
    let (c0, c1) = (RayonComm::new(0, 2), RayonComm::new(1, 2));
    let h = std::thread::spawn(move || run_rank(1, &c1, swap_partitioning));
    let s0 = run_rank(0, &c0, swap_partitioning);
    let s1 = h.join().unwrap();
    (s0, s1)
}

#[test]
fn nocomm_small_case_baseline() {
    let c = NoComm;
    assert!(c.is_no_comm());

    let left_cell = 1.0;
    let right_cell = 2.0;
    let left_boundary = 3.0;
    let right_boundary = 7.0;
    let left_iface = 5.0;
    let right_iface = -5.0;

    let global: f64 = left_cell + right_cell + left_boundary + right_boundary;
    let interface_cancel: f64 = left_iface + right_iface;

    assert!((global - 13.0).abs() < 1e-12);
    assert!(interface_cancel.abs() < 1e-12);
}

#[test]
fn rayon_case_validates_conservation_and_interface_cancellation() {
    let (s0, s1) = run_rayon_pair(false);
    assert!((s0.local_residual + s1.local_residual - 13.0).abs() < 1e-12);
    assert!((s0.interface_flux + s1.interface_flux).abs() < 1e-12);
}

#[test]
fn rayon_per_cell_residuals_are_partitioning_invariant() {
    let (a0, a1) = run_rayon_pair(false);
    let (b0, b1) = run_rayon_pair(true);

    let mut layout_a = [
        (a0.local_cell, a0.local_residual),
        (a1.local_cell, a1.local_residual),
    ];
    let mut layout_b = [
        (b0.local_cell, b0.local_residual),
        (b1.local_cell, b1.local_residual),
    ];
    layout_a.sort_by_key(|(pt, _)| pt.get());
    layout_b.sort_by_key(|(pt, _)| pt.get());

    for ((ca, ra), (cb, rb)) in layout_a.into_iter().zip(layout_b.into_iter()) {
        assert_eq!(ca, cb, "cell identity mismatch across layouts");
        assert!(
            (ra - rb).abs() < 1e-12,
            "residual mismatch on cell {:?}",
            ca
        );
    }
}
