use mesh_sieve::algs::{classify_boundary_points, label_boundary_points};
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use mesh_sieve::topology::sieve::in_memory::InMemorySieve;

fn p(i: u64) -> PointId {
    PointId::new(i).unwrap()
}

fn build_two_triangle_sieve() -> InMemorySieve<PointId, ()> {
    let mut sieve = InMemorySieve::new();
    // Two triangles sharing edge5.
    let (c1, c2) = (p(1), p(2));
    let (e3, e4, e5, e6, e7) = (p(3), p(4), p(5), p(6), p(7));
    let (v8, v9, v10, v11, v12) = (p(8), p(9), p(10), p(11), p(12));

    for edge in [e3, e4, e5] {
        sieve.add_arrow(c1, edge, ());
    }
    for edge in [e5, e6, e7] {
        sieve.add_arrow(c2, edge, ());
    }

    sieve.add_arrow(e3, v8, ());
    sieve.add_arrow(e3, v9, ());
    sieve.add_arrow(e4, v9, ());
    sieve.add_arrow(e4, v10, ());
    sieve.add_arrow(e5, v10, ());
    sieve.add_arrow(e5, v11, ());
    sieve.add_arrow(e6, v11, ());
    sieve.add_arrow(e6, v12, ());
    sieve.add_arrow(e7, v12, ());
    sieve.add_arrow(e7, v8, ());

    sieve
}

#[test]
fn classify_boundary_edges_by_incident_cells() {
    let sieve = build_two_triangle_sieve();
    let edges = vec![p(3), p(4), p(5), p(6), p(7)];

    let classification = classify_boundary_points(&sieve, edges).unwrap();

    assert_eq!(classification.boundary, vec![p(3), p(4), p(6), p(7)]);
    assert_eq!(classification.interior, vec![p(5)]);
}

#[test]
fn label_boundary_edges_with_default_labels() {
    let sieve = build_two_triangle_sieve();
    let edges = vec![p(3), p(4), p(5), p(6), p(7)];
    let mut labels = LabelSet::new();

    let classification = label_boundary_points(&sieve, edges, &mut labels).unwrap();

    assert_eq!(classification.boundary, vec![p(3), p(4), p(6), p(7)]);
    assert_eq!(classification.interior, vec![p(5)]);
    assert_eq!(labels.get_label(p(3), "boundary"), Some(1));
    assert_eq!(labels.get_label(p(4), "boundary"), Some(1));
    assert_eq!(labels.get_label(p(6), "boundary"), Some(1));
    assert_eq!(labels.get_label(p(7), "boundary"), Some(1));
    assert_eq!(labels.get_label(p(5), "boundary"), None);
    assert_eq!(labels.get_label(p(5), "interior"), Some(0));
}
