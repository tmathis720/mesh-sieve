#![allow(clippy::too_many_arguments)]

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::data::refine::delta::SliceDelta;
use crate::data::slice_storage::SliceStorage;
use crate::mesh_error::MeshSieveError;
use crate::topology::arrow::Polarity;

/// GPU-backed storage for slice data using wgpu.
#[derive(Debug)]
pub struct WgpuStorage<V: Pod> {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffer: wgpu::Buffer,
    len: usize,
    _pd: PhantomData<V>,
}

impl<V> WgpuStorage<V>
where
    V: Pod + Zeroable + 'static + Send + Sync,
{
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, len: usize) -> Self {
        let byte_len = (len * std::mem::size_of::<V>()) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Section/WgpuStorage"),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if byte_len > 0 {
            let zeros = vec![V::zeroed(); len];
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&zeros));
        }
        Self {
            device,
            queue,
            buffer,
            len,
            _pd: PhantomData,
        }
    }

    #[inline]
    fn elem_size() -> usize {
        std::mem::size_of::<V>()
    }

    #[inline]
    fn to_bytes(off: usize, len: usize) -> (u64, u64) {
        let b = Self::elem_size();
        ((off * b) as u64, (len * b) as u64)
    }

    fn copy_requires_alignment() -> bool {
        Self::elem_size() % 4 == 0
    }

    fn ranges_overlap(a_off: usize, a_len: usize, b_off: usize, b_len: usize) -> bool {
        let a_end = a_off + a_len;
        let b_end = b_off + b_len;
        !(a_end <= b_off || b_end <= a_off)
    }

    fn copy_via_staging(
        &mut self,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), MeshSieveError> {
        let host = self.read_slice(src_off, len)?;
        self.write_slice(dst_off, &host)
    }

    fn copy_forward_gpu(
        &mut self,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), MeshSieveError> {
        if !Self::copy_requires_alignment() {
            return self.copy_via_staging(src_off, dst_off, len);
        }
        let (src_b, size_b) = Self::to_bytes(src_off, len);
        let (dst_b, _) = Self::to_bytes(dst_off, len);
        if Self::ranges_overlap(src_off, len, dst_off, len) && src_off < dst_off {
            return self.copy_via_temp_gpu(src_off, dst_off, len);
        }
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WgpuStorage::copy_forward_gpu"),
            });
        enc.copy_buffer_to_buffer(&self.buffer, src_b, &self.buffer, dst_b, size_b);
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    fn copy_via_temp_gpu(
        &mut self,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), MeshSieveError> {
        if !Self::copy_requires_alignment() {
            return self.copy_via_staging(src_off, dst_off, len);
        }
        let (src_b, size_b) = Self::to_bytes(src_off, len);
        let (dst_b, _) = Self::to_bytes(dst_off, len);
        let temp = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuStorage::temp"),
            size: size_b,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WgpuStorage::copy_via_temp_gpu"),
            });
        enc.copy_buffer_to_buffer(&self.buffer, src_b, &temp, 0, size_b);
        enc.copy_buffer_to_buffer(&temp, 0, &self.buffer, dst_b, size_b);
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }

    fn reverse_via_compute(
        &mut self,
        src_off: usize,
        dst_off: usize,
        len: usize,
    ) -> Result<(), MeshSieveError> {
        if !Self::copy_requires_alignment() {
            return self.copy_via_staging(src_off, dst_off, len);
        }
        let (src_b, size_b) = Self::to_bytes(src_off, len);
        if size_b == 0 {
            return Ok(());
        }
        let temp = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuStorage::reverse_temp"),
            size: size_b,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reverse_copy.wgsl"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "reverse_copy.wgsl"
                ))),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reverse_copy_pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
            });
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        struct Params {
            elem_words: u32,
            elem_count: u32,
            src_off_words: u32,
            dst_off_words: u32,
        }
        let elem_words = (Self::elem_size() / 4) as u32;
        if elem_words == 0 {
            return self.copy_via_staging(src_off, dst_off, len);
        }
        let params = Params {
            elem_words,
            elem_count: len as u32,
            src_off_words: 0,
            dst_off_words: (dst_off * Self::elem_size() / 4) as u32,
        };
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reverse_copy_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let layout = pipeline.get_bind_group_layout(0);
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reverse_copy_bind"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: temp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ubuf.as_entire_binding(),
                },
            ],
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WgpuStorage::reverse_via_compute"),
            });
        enc.copy_buffer_to_buffer(&self.buffer, src_b, &temp, 0, size_b);
        {
            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reverse_copy_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind, &[]);
            let workgroups = ((len as u32) + 127) / 128;
            if workgroups > 0 {
                cpass.dispatch_workgroups(workgroups, 1, 1);
            }
        }
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }
}

impl<V> SliceStorage<V> for WgpuStorage<V>
where
    V: Pod + Zeroable + 'static + Send + Sync,
{
    fn total_len(&self) -> usize {
        self.len
    }

    fn resize(&mut self, new_len: usize) -> Result<(), MeshSieveError> {
        if new_len == self.len {
            return Ok(());
        }
        let new_bytes = (new_len * Self::elem_size()) as u64;
        let new_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Section/WgpuStorage[resize]"),
            size: new_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let copy_elems = self.len.min(new_len);
        if copy_elems > 0 {
            if Self::copy_requires_alignment() {
                let (src_b, size_b) = Self::to_bytes(0, copy_elems);
                let mut enc = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("WgpuStorage::resize"),
                    });
                enc.copy_buffer_to_buffer(&self.buffer, src_b, &new_buf, 0, size_b);
                self.queue.submit(Some(enc.finish()));
            } else {
                let host = self.read_slice(0, copy_elems)?;
                self.queue
                    .write_buffer(&new_buf, 0, bytemuck::cast_slice(&host));
            }
        }
        self.buffer = new_buf;
        self.len = new_len;
        Ok(())
    }

    fn read_slice(&self, offset: usize, len: usize) -> Result<Vec<V>, MeshSieveError> {
        let end = offset
            .checked_add(len)
            .ok_or(MeshSieveError::ScatterChunkMismatch { offset, len })?;
        if end > self.len {
            return Err(MeshSieveError::ScatterChunkMismatch { offset, len });
        }
        let (src_b, size_b) = Self::to_bytes(offset, len);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WgpuStorage[read] staging"),
            size: size_b,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WgpuStorage::read_slice"),
            });
        enc.copy_buffer_to_buffer(&self.buffer, src_b, &staging, 0, size_b);
        self.queue.submit(Some(enc.finish()));
        let buffer_slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);
        let res = pollster::block_on(receiver.receive());
        res.ok_or(MeshSieveError::GpuMappingFailed)?
            .map_err(|_| MeshSieveError::GpuMappingFailed)?;
        let data = buffer_slice.get_mapped_range();
        let mut out = vec![V::zeroed(); len];
        out.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
        Ok(out)
    }

    fn write_slice(&mut self, offset: usize, src: &[V]) -> Result<(), MeshSieveError> {
        let end = offset
            .checked_add(src.len())
            .ok_or(MeshSieveError::ScatterChunkMismatch {
                offset,
                len: src.len(),
            })?;
        if end > self.len {
            return Err(MeshSieveError::ScatterChunkMismatch {
                offset,
                len: src.len(),
            });
        }
        let (dst_b, _) = Self::to_bytes(offset, src.len());
        self.queue
            .write_buffer(&self.buffer, dst_b, bytemuck::cast_slice(src));
        Ok(())
    }

    fn apply_delta<D: SliceDelta<V> + 'static>(
        &mut self,
        src_off: usize,
        dst_off: usize,
        len: usize,
        delta: &D,
    ) -> Result<(), MeshSieveError> {
        if len == 0 {
            return Ok(());
        }
        if let Some(pol) = (delta as &dyn Any).downcast_ref::<Polarity>() {
            match pol {
                Polarity::Forward => return self.copy_forward_gpu(src_off, dst_off, len),
                Polarity::Reverse => {
                    return self
                        .reverse_via_compute(src_off, dst_off, len)
                        .or_else(|_| self.copy_via_staging(src_off, dst_off, len));
                }
            }
        }
        // Fallback: CPU staging
        let src_host = self.read_slice(src_off, len)?;
        let mut dst_host = vec![V::zeroed(); len];
        delta.apply(&src_host, &mut dst_host)?;
        self.write_slice(dst_off, &dst_host)
    }
}
