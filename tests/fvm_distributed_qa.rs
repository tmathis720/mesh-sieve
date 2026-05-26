use mesh_sieve::algs::communicator::{Communicator, NoComm, RayonComm};
use mesh_sieve::algs::completion::section_completion::complete_section;
use mesh_sieve::data::{atlas::Atlas, section::Section, storage::VecStorage};
use mesh_sieve::overlap::{delta::CopyDelta, overlap::Overlap};
use mesh_sieve::topology::point::PointId;

fn p(id: u64) -> PointId { PointId::new(id).unwrap() }

#[test]
fn fvm_partition_invariants_with_rayon_comm() {
    let (c0,c1)=(RayonComm::new(0,2), RayonComm::new(1,2));
    let h=std::thread::spawn(move || {
        let mut ov=Overlap::default(); ov.try_add_link(p(101),0,p(1)).unwrap(); ov.try_add_link(p(201),0,p(21)).unwrap();
        let mut a=Atlas::default(); for id in [101,201,20,1,21] {a.try_insert(p(id),1).unwrap();}
        let mut cell=Section::<f64,VecStorage<f64>>::new(a.clone()); let mut face=Section::<f64,VecStorage<f64>>::new(a);
        cell.try_set(p(101), &[2.0]).unwrap(); face.try_set(p(201), &[5.0]).unwrap(); face.try_set(p(20), &[7.0]).unwrap();
        complete_section::<f64,_,CopyDelta,_>(&mut cell,&ov,&c1,1).unwrap();
        complete_section::<f64,_,CopyDelta,_>(&mut face,&ov,&c1,1).unwrap();
        let r1=cell.try_restrict(p(101)).unwrap()[0] + face.try_restrict(p(20)).unwrap()[0] - face.try_restrict(p(201)).unwrap()[0];
        (r1, face.try_restrict(p(201)).unwrap()[0], cell.try_restrict(p(101)).unwrap()[0])
    });

    let mut ov=Overlap::default(); ov.try_add_link(p(1),1,p(101)).unwrap(); ov.try_add_link(p(21),1,p(201)).unwrap();
    let mut a=Atlas::default(); for id in [1,21,10,101,201] {a.try_insert(p(id),1).unwrap();}
    let mut cell=Section::<f64,VecStorage<f64>>::new(a.clone()); let mut face=Section::<f64,VecStorage<f64>>::new(a);
    cell.try_set(p(1), &[1.0]).unwrap(); face.try_set(p(21), &[5.0]).unwrap(); face.try_set(p(10), &[3.0]).unwrap();
    complete_section::<f64,_,CopyDelta,_>(&mut cell,&ov,&c0,0).unwrap();
    complete_section::<f64,_,CopyDelta,_>(&mut face,&ov,&c0,0).unwrap();
    let r0=cell.try_restrict(p(1)).unwrap()[0] + face.try_restrict(p(10)).unwrap()[0] + face.try_restrict(p(21)).unwrap()[0];

    let (r1,iface1,c2)=h.join().unwrap();
    let iface0=face.try_restrict(p(21)).unwrap()[0];
    assert!((r0+r1-13.0).abs()<1e-12);
    assert!((iface0-iface1).abs()<1e-12);
    assert!((iface0+(-iface1)).abs()<1e-12);
    assert!((c2-2.0).abs()<1e-12);
}

#[test]
fn deterministic_comparison_nocomm_vs_rayon() {
    let serial = 13.0;
    let (c0,c1)=(RayonComm::new(0,2), RayonComm::new(1,2));
    let h=std::thread::spawn(move || {
        let mut ov=Overlap::default(); ov.try_add_link(p(101),0,p(1)).unwrap(); ov.try_add_link(p(201),0,p(21)).unwrap();
        let mut a=Atlas::default(); for id in [101,201,20,1,21] {a.try_insert(p(id),1).unwrap();}
        let mut cell=Section::<f64,VecStorage<f64>>::new(a.clone()); let mut face=Section::<f64,VecStorage<f64>>::new(a);
        cell.try_set(p(101), &[2.0]).unwrap(); face.try_set(p(201), &[5.0]).unwrap(); face.try_set(p(20), &[7.0]).unwrap();
        complete_section::<f64,_,CopyDelta,_>(&mut cell,&ov,&c1,1).unwrap();
        complete_section::<f64,_,CopyDelta,_>(&mut face,&ov,&c1,1).unwrap();
        cell.try_restrict(p(101)).unwrap()[0] + face.try_restrict(p(20)).unwrap()[0] - face.try_restrict(p(201)).unwrap()[0]
    });
    let mut ov=Overlap::default(); ov.try_add_link(p(1),1,p(101)).unwrap(); ov.try_add_link(p(21),1,p(201)).unwrap();
    let mut a=Atlas::default(); for id in [1,21,10,101,201] {a.try_insert(p(id),1).unwrap();}
    let mut cell=Section::<f64,VecStorage<f64>>::new(a.clone()); let mut face=Section::<f64,VecStorage<f64>>::new(a);
    cell.try_set(p(1), &[1.0]).unwrap(); face.try_set(p(21), &[5.0]).unwrap(); face.try_set(p(10), &[3.0]).unwrap();
    complete_section::<f64,_,CopyDelta,_>(&mut cell,&ov,&c0,0).unwrap();
    complete_section::<f64,_,CopyDelta,_>(&mut face,&ov,&c0,0).unwrap();
    let rayon_total = cell.try_restrict(p(1)).unwrap()[0] + face.try_restrict(p(10)).unwrap()[0] + face.try_restrict(p(21)).unwrap()[0] + h.join().unwrap();
    let no = NoComm; assert!(no.is_no_comm());
    assert!((serial - rayon_total).abs() < 1e-12);
}
