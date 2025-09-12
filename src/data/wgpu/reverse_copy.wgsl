struct Params {
  elem_words: u32,
  elem_count: u32,
  src_off_words: u32,
  dst_off_words: u32,
}
@group(0) @binding(0) var<storage, read>  src_buf : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_buf : array<u32>;
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.elem_count) { return; }
  let ew = params.elem_words;
  let src_base = params.src_off_words + i * ew;
  let dst_elem = params.elem_count - 1u - i;
  let dst_base = params.dst_off_words + dst_elem * ew;
  for (var k: u32 = 0u; k < ew; k = k + 1u) {
    dst_buf[dst_base + k] = src_buf[src_base + k];
  }
}
