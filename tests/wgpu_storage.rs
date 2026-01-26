#![cfg(feature = "wgpu")]

use std::sync::Arc;

use mesh_sieve::data::slice_storage::SliceStorage;
use mesh_sieve::data::wgpu::WgpuStorage;
use mesh_sieve::topology::arrow::Polarity;
use pollster::block_on;

#[test]
fn wgpu_storage_basic_ops() {
    if std::env::var("MESH_SIEVE_RUN_WGPU_TESTS")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("skipping wgpu test; set MESH_SIEVE_RUN_WGPU_TESTS=1 to enable");
        return;
    }
    let instance = wgpu::Instance::default();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
    let Some(adapter) = adapter else {
        return;
    };
    let (device, queue) =
        block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let mut storage = WgpuStorage::<f32>::new(device.clone(), queue.clone(), 8);
    storage.write_slice(0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let v = storage.read_slice(0, 4).unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    storage.apply_delta(0, 4, 4, &Polarity::Forward).unwrap();
    let v = storage.read_slice(4, 4).unwrap();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    storage.apply_delta(0, 4, 4, &Polarity::Reverse).unwrap();
    let v = storage.read_slice(4, 4).unwrap();
    assert_eq!(v, vec![4.0, 3.0, 2.0, 1.0]);
}
