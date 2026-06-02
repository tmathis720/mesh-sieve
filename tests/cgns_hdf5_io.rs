#![cfg(feature = "cgns")]

use hdf5::File;
use mesh_sieve::io::SieveSectionReader;
use mesh_sieve::io::cgns::CgnsReader;
use mesh_sieve::topology::cell_type::CellType;
use mesh_sieve::topology::point::PointId;
use mesh_sieve::topology::sieve::Sieve;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn reads_cgns_hdf5_mixed_elements_labels_and_solution() {
    let path = temp_path();
    {
        let file = File::create(&path).unwrap();
        let base = file.create_group("Base").unwrap();
        let zone = base.create_group("Zone1").unwrap();
        let grid = zone.create_group("GridCoordinates").unwrap();
        grid.new_dataset::<f64>()
            .shape(4)
            .create("CoordinateX")
            .unwrap()
            .write(&[0.0, 1.0, 1.0, 0.0])
            .unwrap();
        grid.new_dataset::<f64>()
            .shape(4)
            .create("CoordinateY")
            .unwrap()
            .write(&[0.0, 0.0, 1.0, 1.0])
            .unwrap();
        grid.new_dataset::<f64>()
            .shape(4)
            .create("CoordinateZ")
            .unwrap()
            .write(&[0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let elems = zone.create_group("Elements").unwrap();
        elems
            .new_dataset::<i32>()
            .shape(1)
            .create("ElementType")
            .unwrap()
            .write(&[20])
            .unwrap();
        elems
            .new_dataset::<i64>()
            .shape(8)
            .create("ElementConnectivity")
            .unwrap()
            .write(&[5, 1, 2, 3, 5, 1, 3, 4])
            .unwrap();
        let zone_bc = zone.create_group("ZoneBC").unwrap();
        let wall = zone_bc.create_group("Wall").unwrap();
        wall.new_dataset::<i64>()
            .shape(2)
            .create("PointList")
            .unwrap()
            .write(&[1, 4])
            .unwrap();
        let sol = zone.create_group("FlowSolution").unwrap();
        sol.new_dataset::<f64>()
            .shape(4)
            .create("Pressure")
            .unwrap()
            .write(&[1.0, 2.0, 3.0, 4.0])
            .unwrap();
    }
    let bytes = fs::read(&path).unwrap();
    let _ = fs::remove_file(&path);

    let mesh = CgnsReader::default().read(&bytes[..]).unwrap();
    let cell_types = mesh.cell_types.as_ref().unwrap();
    assert_eq!(
        cell_types.try_restrict(PointId::new(5).unwrap()).unwrap(),
        &[CellType::Triangle]
    );
    assert_eq!(
        cell_types.try_restrict(PointId::new(6).unwrap()).unwrap(),
        &[CellType::Triangle]
    );
    assert_eq!(mesh.sieve.cone_points(PointId::new(5).unwrap()).count(), 3);
    let labels = mesh.labels.as_ref().unwrap();
    assert_eq!(
        labels.get_label(PointId::new(1).unwrap(), "cgns:bc:Wall"),
        Some(1)
    );
    let pressure = mesh.sections.get("FlowSolution/Pressure").unwrap();
    assert_eq!(
        pressure.try_restrict(PointId::new(4).unwrap()).unwrap(),
        &[4.0]
    );
}

#[test]
fn reads_cgns_hdf5_multi_zone_offsets_labels_and_sections() {
    let path = temp_path();
    {
        let file = File::create(&path).unwrap();
        let base = file.create_group("Base").unwrap();

        let zone1 = base.create_group("Zone1").unwrap();
        let grid1 = zone1.create_group("GridCoordinates").unwrap();
        grid1
            .new_dataset::<f64>()
            .shape(3)
            .create("CoordinateX")
            .unwrap()
            .write(&[0.0, 1.0, 0.0])
            .unwrap();
        grid1
            .new_dataset::<f64>()
            .shape(3)
            .create("CoordinateY")
            .unwrap()
            .write(&[0.0, 0.0, 1.0])
            .unwrap();
        let elems1 = zone1.create_group("Elements").unwrap();
        elems1
            .new_dataset::<i32>()
            .shape(1)
            .create("ElementType")
            .unwrap()
            .write(&[5])
            .unwrap();
        elems1
            .new_dataset::<i64>()
            .shape(3)
            .create("ElementConnectivity")
            .unwrap()
            .write(&[1, 2, 3])
            .unwrap();
        let sol1 = zone1.create_group("FlowSolution").unwrap();
        sol1.new_dataset::<f64>()
            .shape(3)
            .create("Pressure")
            .unwrap()
            .write(&[10.0, 11.0, 12.0])
            .unwrap();

        let zone2 = base.create_group("Zone2").unwrap();
        let grid2 = zone2.create_group("GridCoordinates").unwrap();
        grid2
            .new_dataset::<f64>()
            .shape(2)
            .create("CoordinateX")
            .unwrap()
            .write(&[2.0, 3.0])
            .unwrap();
        grid2
            .new_dataset::<f64>()
            .shape(2)
            .create("CoordinateY")
            .unwrap()
            .write(&[0.0, 0.0])
            .unwrap();
        let elems2 = zone2.create_group("Elements").unwrap();
        elems2
            .new_dataset::<i32>()
            .shape(1)
            .create("ElementType")
            .unwrap()
            .write(&[3])
            .unwrap();
        elems2
            .new_dataset::<i64>()
            .shape(2)
            .create("ElementConnectivity")
            .unwrap()
            .write(&[1, 2])
            .unwrap();
        let zone_bc2 = zone2.create_group("ZoneBC").unwrap();
        let inlet = zone_bc2.create_group("Inlet").unwrap();
        inlet
            .new_dataset::<i64>()
            .shape(2)
            .create("PointRange")
            .unwrap()
            .write(&[1, 2])
            .unwrap();
        let sol2 = zone2.create_group("FlowSolution").unwrap();
        sol2.new_dataset::<f64>()
            .shape(2)
            .create("Pressure")
            .unwrap()
            .write(&[20.0, 21.0])
            .unwrap();
    }
    let bytes = fs::read(&path).unwrap();
    let _ = fs::remove_file(&path);

    let mesh = CgnsReader::default().read(&bytes[..]).unwrap();
    let labels = mesh.labels.as_ref().unwrap();
    assert_eq!(
        labels.get_label(PointId::new(4).unwrap(), "cgns:zone:Zone2"),
        Some(1)
    );
    assert_eq!(
        labels.get_label(PointId::new(4).unwrap(), "cgns:bc:Inlet"),
        Some(1)
    );
    assert_eq!(mesh.sieve.cone_points(PointId::new(6).unwrap()).count(), 2);
    assert_eq!(
        mesh.sections
            .get("Zone2/FlowSolution/Pressure")
            .unwrap()
            .try_restrict(PointId::new(5).unwrap())
            .unwrap(),
        &[21.0]
    );
}

fn temp_path() -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!("mesh_sieve_test_{nanos}.cgns"));
    path
}
