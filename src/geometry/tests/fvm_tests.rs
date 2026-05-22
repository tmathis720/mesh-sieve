use crate::data::atlas::Atlas;
use crate::data::coordinates::Coordinates;
use crate::data::section::Section;
use crate::data::storage::VecStorage;
use crate::geometry::fvm::build_fvm_face_metrics;
use crate::mesh_error::MeshSieveError;
use crate::topology::cell_type::CellType;
use crate::topology::point::PointId;
use crate::topology::sieve::{InMemorySieve, MutableSieve, Sieve};

fn p(id: u64) -> PointId { PointId::new(id).unwrap() }

#[test]
fn fvm_internal_and_boundary_faces_triangle_pair() -> Result<(), MeshSieveError> {
    let mut s = InMemorySieve::<PointId, ()>::default();
    for id in 1..=11 { MutableSieve::add_point(&mut s, p(id)); }
    let (v1,v2,v3,v4,e12,e23,e31,e24,e43,c1,c2)=(p(1),p(2),p(3),p(4),p(5),p(6),p(7),p(8),p(9),p(10),p(11));
    for (e,a,b) in [(e12,v1,v2),(e23,v2,v3),(e31,v3,v1),(e24,v2,v4),(e43,v4,v3)] { s.add_arrow(e,a,()); s.add_arrow(e,b,()); }
    for e in [e12,e23,e31] { s.add_arrow(c1,e,()); }
    for e in [e23,e24,e43] { s.add_arrow(c2,e,()); }

    let mut a=Atlas::default(); for v in [v1,v2,v3,v4]{a.try_insert(v,2)?;} let mut coords=Coordinates::<f64,VecStorage<f64>>::try_new(2,2,a)?;
    coords.section_mut().try_set(v1,&[0.0,0.0])?; coords.section_mut().try_set(v2,&[1.0,0.0])?; coords.section_mut().try_set(v3,&[0.0,1.0])?; coords.section_mut().try_set(v4,&[1.0,1.0])?;

    let mut ta=Atlas::default(); for q in [c1,c2,e12,e23,e31,e24,e43]{ta.try_insert(q,1)?;} let mut types=Section::<CellType,VecStorage<CellType>>::new(ta);
    types.try_set(c1,&[CellType::Triangle])?; types.try_set(c2,&[CellType::Triangle])?;
    for e in [e12,e23,e31,e24,e43]{types.try_set(e,&[CellType::Segment])?;}

    let metrics = build_fvm_face_metrics(&s,&types,&coords)?;
    assert_eq!(metrics.len(), 5);
    let internal = metrics.iter().find(|m| m.face==e23).unwrap();
    assert_eq!(internal.owner, c1); assert_eq!(internal.neighbor, Some(c2));
    assert!(internal.non_orthogonality_deg.unwrap() >= 0.0);
    let boundary = metrics.iter().find(|m| m.face==e12).unwrap();
    assert_eq!(boundary.neighbor, None);
    assert!(boundary.orthogonal_distance > 0.0);
    Ok(())
}
