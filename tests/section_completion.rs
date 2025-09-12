mod util;
use mesh_sieve::{
    algs::completion::section_completion::complete_section,
    data::{atlas::Atlas, section::Section, storage::VecStorage},
    overlap::{delta::ValueDelta, overlap::Overlap},
};
use util::*;

#[derive(Clone, Default, PartialEq, Debug)]
struct Vec3([f64; 3]);

#[derive(Copy, Clone)]
struct AvgDelta;
impl ValueDelta<Vec3> for AvgDelta {
    type Part = [f64; 3];
    fn restrict(v: &Vec3) -> Self::Part {
        v.0
    }
    fn fuse(local: &mut Vec3, incoming: Self::Part) {
        for i in 0..3 {
            local.0[i] = 0.5 * (local.0[i] + incoming[i]);
        }
    }
}

#[test]
fn complete_section_vec3_happy_path() {
    let (c0, c1) = rayons();

    let mut ov0 = Overlap::default();
    ov0.add_link(pid(1), 1, pid(101));

    let mut ov1 = Overlap::default();
    ov1.add_link(pid(101), 0, pid(1));

    let mut sec0 = Section::<Vec3, VecStorage<Vec3>>::new(Atlas::default());
    sec0.try_add_point(pid(1), 1).unwrap();
    sec0.try_add_point(pid(101), 1).unwrap();
    sec0.try_restrict_mut(pid(1)).unwrap()[0] = Vec3([1.0, 2.0, 3.0]);

    let mut sec1 = Section::<Vec3, VecStorage<Vec3>>::new(Atlas::default());
    sec1.try_add_point(pid(101), 1).unwrap();
    sec1.try_add_point(pid(1), 1).unwrap();
    sec1.try_restrict_mut(pid(101)).unwrap()[0] = Vec3([1.0, 2.0, 3.0]);

    let handle = std::thread::spawn(move || {
        let mut s1 = sec1;
        complete_section::<Vec3, VecStorage<Vec3>, AvgDelta, _>(&mut s1, &ov1, &c1, 1).unwrap();
    });
    complete_section::<Vec3, VecStorage<Vec3>, AvgDelta, _>(&mut sec0, &ov0, &c0, 0).unwrap();
    handle.join().unwrap();

    let got = &sec0.try_restrict(pid(1)).unwrap()[0];
    assert_eq!(got, &Vec3([1.0, 2.0, 3.0]));
}

#[test]
fn complete_section_missing_overlap_errors() {
    let (c0, _c1) = rayons();

    let mut ov = Overlap::default();
    ov.add_link_structural_one(pid(2), 1);

    let mut sec = Section::<Vec3, VecStorage<Vec3>>::new(Atlas::default());
    sec.try_add_point(pid(2), 1).unwrap();
    sec.try_restrict_mut(pid(2)).unwrap()[0] = Vec3([9.0, 9.0, 9.0]);

    let err = complete_section::<Vec3, VecStorage<Vec3>, AvgDelta, _>(&mut sec, &ov, &c0, 0)
        .err()
        .expect("should fail");
    let msg = format!("{:?}", err);
    assert!(msg.contains("Overlap link not found"), "{msg}");
}
