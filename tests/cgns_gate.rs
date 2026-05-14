use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::cgns::CgnsReader;

#[cfg(not(feature = "cgns"))]
#[test]
fn cgns_reader_reports_feature_gate() {
    let err = CgnsReader::default()
        .read([].as_slice())
        .expect_err("CGNS should be feature gated by default");
    assert!(err.to_string().contains("--features cgns"));
}
