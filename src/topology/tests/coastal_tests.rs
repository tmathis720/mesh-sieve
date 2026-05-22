use crate::topology::point::PointId;
use crate::topology::{
    BOUNDARY_CLASS_LABEL, BOUNDARY_ROLE_LABEL, BoundaryClass, CoastalLabelQueries,
    CoastalMetadataError, CoastalValidationOptions, LabelSet, OpenBoundaryRole,
    VERTICAL_LAYER_LABEL, validate_coastal_metadata,
};

#[test]
fn canonical_coastal_queries_return_expected_points() {
    let p = |v| PointId::new(v).unwrap();
    let mut labels = LabelSet::new();
    labels.set_label(p(1), BOUNDARY_CLASS_LABEL, BoundaryClass::FreeSurface.code());
    labels.set_label(p(2), BOUNDARY_CLASS_LABEL, BoundaryClass::Bed.code());
    labels.set_label(p(3), BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());
    labels.set_label(p(3), BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Inflow.code());
    labels.set_label(p(10), VERTICAL_LAYER_LABEL, 0);
    labels.set_label(p(11), VERTICAL_LAYER_LABEL, 1);

    assert_eq!(labels.free_surface_points(), vec![p(1)]);
    assert_eq!(labels.bed_points(), vec![p(2)]);
    assert_eq!(labels.open_boundary_points(), vec![p(3)]);
    assert_eq!(labels.inflow_points(), vec![p(3)]);
    assert_eq!(labels.vertical_layer_points(0), vec![p(10)]);
}

#[test]
fn validates_open_role_membership() {
    let p = |v| PointId::new(v).unwrap();
    let mut labels = LabelSet::new();
    labels.set_label(p(9), BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Outflow.code());

    let err = validate_coastal_metadata(&labels, None, None, CoastalValidationOptions::default())
        .expect_err("open role without open class should fail");
    assert_eq!(
        err,
        CoastalMetadataError::OpenRoleWithoutOpenClass { points: vec![p(9)] }
    );
}

#[test]
fn validates_open_role_and_coverage() {
    let p = |v| PointId::new(v).unwrap();
    let mut labels = LabelSet::new();
    labels.set_label(p(2), BOUNDARY_CLASS_LABEL, BoundaryClass::Open.code());

    let err = validate_coastal_metadata(&labels, None, None, CoastalValidationOptions::default())
        .expect_err("open boundary role is required by default");
    assert_eq!(
        err,
        CoastalMetadataError::MissingOpenBoundaryRole { points: vec![p(2)] }
    );

    labels.set_label(p(2), BOUNDARY_ROLE_LABEL, OpenBoundaryRole::Tidal.code());

    let opts = CoastalValidationOptions {
        require_complete_boundary_partition: true,
        require_open_role_on_open_boundary: true,
        require_complete_vertical_coverage: true,
    };

    let err = validate_coastal_metadata(&labels, Some(&[p(1), p(2)]), Some(&[p(10)]), opts)
        .expect_err("missing coverage should fail");
    assert_eq!(
        err,
        CoastalMetadataError::MissingBoundaryClass { points: vec![p(1)] }
    );
}
