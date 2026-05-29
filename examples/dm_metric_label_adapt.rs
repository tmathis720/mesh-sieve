// cargo run --example dm_metric_label_adapt
use mesh_sieve::adapt::{MetricTensor, MetricThresholds};
use mesh_sieve::algs::meshgen::{MeshGenOptions, StructuredCellType, structured_box_2d};
use mesh_sieve::data::atlas::Atlas;
use mesh_sieve::data::section::Section;
use mesh_sieve::data::storage::VecStorage;
use mesh_sieve::dm::{MeshDM, MeshDMMetricAdaptOptions};
use mesh_sieve::io::MeshData;
use mesh_sieve::topology::sieve::{MeshSieve, OrientedSieve, Sieve};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = structured_box_2d(
        4,
        3,
        [0.0, 0.0],
        [1.0, 1.0],
        StructuredCellType::Triangle,
        MeshGenOptions::default(),
    )?;
    let mut oriented = MeshSieve::default();
    for source in mesh.sieve.base_points() {
        for target in mesh.sieve.cone_points(source) {
            oriented.add_arrow_o(source, target, (), 0);
        }
    }
    let mesh = MeshData {
        sieve: oriented,
        coordinates: mesh.coordinates,
        sections: mesh.sections,
        mixed_sections: mesh.mixed_sections,
        labels: mesh.labels,
        cell_types: mesh.cell_types,
        discretization: mesh.discretization,
    };
    let mut dm = MeshDM::<f64>::from_mesh_data(mesh);

    let cells = dm.height_stratum(0)?;
    let mut atlas = Atlas::default();
    for c in &cells {
        atlas.try_insert(*c, 1)?;
    }
    let mut metric = Section::<MetricTensor, VecStorage<MetricTensor>>::new(atlas);
    for cell in &cells {
        metric.try_set(*cell, &[MetricTensor::new_2d(2.0, 0.7, 0.0)])?;
    }

    let mut options = MeshDMMetricAdaptOptions::default();
    options.thresholds = MetricThresholds {
        refine_max_edge_length: 1.0,
        ..MetricThresholds::default()
    };
    options
        .labels
        .fixed_boundary_labels
        .push(("boundary".to_string(), 1));

    let out = dm.adapt_with_attached_metric("cell_metric", &metric, |_| Vec::new(), options)?;
    println!(
        "action={:?} changed_cells={} preserved_labels={}",
        out.action, out.diagnostics.changed_cells, out.diagnostics.preserved_labels
    );
    Ok(())
}
