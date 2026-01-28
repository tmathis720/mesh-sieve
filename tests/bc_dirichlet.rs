use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::bc::{
    FieldDofIndices, LabelQuery, apply_dirichlet_to_constrained_section,
    apply_dirichlet_to_section_fields,
};
use mesh_sieve::data::constrained_section::ConstrainedSection;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::topology::labels::LabelSet;
use mesh_sieve::topology::point::PointId;

fn build_section() -> Section<f64, VecStorage<f64>> {
    let mut atlas = Atlas::default();
    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    atlas.try_insert(p1, 3).unwrap();
    atlas.try_insert(p2, 3).unwrap();
    let mut section = Section::new(atlas);
    section.try_set(p1, &[1.0, 2.0, 3.0]).unwrap();
    section.try_set(p2, &[4.0, 5.0, 6.0]).unwrap();
    section
}

#[test]
fn dirichlet_on_label_applies_field_indices() {
    let mut section = build_section();
    let p1 = PointId::new(1).unwrap();
    let p2 = PointId::new(2).unwrap();
    let mut labels = LabelSet::new();
    labels.set_label(p1, "boundary", 1);

    let query = LabelQuery::new("boundary", 1);
    let field_dofs = [2usize, 1usize];
    let field_indices = vec![
        FieldDofIndices::new(0, vec![1]),
        FieldDofIndices::new(1, vec![0]),
    ];

    apply_dirichlet_to_section_fields(
        &mut section,
        &labels,
        &query,
        &field_dofs,
        &field_indices,
        |point, field, dof| 100.0 + point.get() as f64 + (field as f64) * 10.0 + dof as f64,
    )
    .unwrap();

    assert_eq!(section.try_restrict(p1).unwrap(), &[1.0, 102.0, 111.0]);
    assert_eq!(section.try_restrict(p2).unwrap(), &[4.0, 5.0, 6.0]);
}

#[test]
fn dirichlet_on_label_applies_to_constrained_section() {
    let section = build_section();
    let p2 = PointId::new(2).unwrap();
    let mut constrained = ConstrainedSection::new(section);
    let mut labels = LabelSet::new();
    labels.set_label(p2, "boundary", 2);

    let query = LabelQuery::new("boundary", 2);
    apply_dirichlet_to_constrained_section(
        &mut constrained,
        &labels,
        &query,
        &[0, 2],
        |point, dof| -(point.get() as f64 + dof as f64),
    )
    .unwrap();

    assert_eq!(
        constrained.section().try_restrict(p2).unwrap(),
        &[-2.0, 5.0, -4.0]
    );
}
