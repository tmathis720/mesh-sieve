use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::bundle::{AverageReducer, Bundle};
use mesh_sieve::data::section::Section;
use mesh_sieve::mesh_error::MeshSieveError;
use mesh_sieve::overlap::delta::CopyDelta;
use mesh_sieve::topology::arrow::Polarity;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::stack::InMemoryStack;

#[test]
fn assemble_reports_cap_point_on_length_mismatch() {
    let b = PointId::new(10).unwrap();
    let c_ok = PointId::new(20).unwrap();
    let c_bad = PointId::new(30).unwrap();

    let mut atlas = Atlas::default();
    atlas.try_insert(b, 3).unwrap();
    atlas.try_insert(c_ok, 3).unwrap();
    atlas.try_insert(c_bad, 2).unwrap();

    let section: Section<f64> = Section::new(atlas);

    let mut stack = InMemoryStack::<PointId, PointId, Polarity>::default();
    stack.add_arrow(b, c_ok, Polarity::Forward).unwrap();
    stack.add_arrow(b, c_bad, Polarity::Forward).unwrap();

    let mut bundle = Bundle { stack, section, delta: CopyDelta };

    let err = bundle.assemble_with([b], &AverageReducer).unwrap_err();
    match err {
        MeshSieveError::SliceLengthMismatch { point, expected, found } => {
            assert_eq!(point, c_bad);
            assert_eq!(expected, 3);
            assert_eq!(found, 2);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}
