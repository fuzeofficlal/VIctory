//! The argument pass on a device of its own (hardware or WARP): synthetic
//! records in, the commands it writes read back.

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::*;

fn device() -> Option<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;
    // SAFETY: a plain device creation with a valid out pointer.
    unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }.ok()?;
    device
}

fn buffer(
    device: &ID3D12Device,
    heap: D3D12_HEAP_TYPE,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: a plain buffer creation.
    unsafe {
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_UNKNOWN,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut r,
            )
            .unwrap();
    }
    r.unwrap()
}

fn transition(
    r: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Runs the pass over `args` (dwords) with `count` and returns the
/// command dwords it wrote.
fn run(
    device: &ID3D12Device,
    args: &[u32],
    _records: (u32, u32),
    count: Option<u32>,
    plan: impl Fn(u64) -> [u32; 16],
    out_dwords: usize,
) -> Vec<u32> {
    let c = make(device).expect("the pass compiles and the device takes it");
    let mut input = args.to_vec();
    let count_at = input.len() * 4;
    input.push(count.unwrap_or(0));
    let in_bytes = (input.len() * 4) as u64;
    let out_bytes = (out_dwords * 4) as u64;
    let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    let upload = buffer(
        device,
        D3D12_HEAP_TYPE_UPLOAD,
        in_bytes,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let data = buffer(
        device,
        D3D12_HEAP_TYPE_DEFAULT,
        in_bytes,
        uav,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let out = buffer(
        device,
        D3D12_HEAP_TYPE_DEFAULT,
        out_bytes,
        uav,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let readback = buffer(
        device,
        D3D12_HEAP_TYPE_READBACK,
        out_bytes,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: D3D12 calls on live objects; the fence wait ends the GPU's
    // use of them before they drop.
    unsafe {
        let mut p = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(input.as_ptr(), p.cast::<u32>(), input.len());
        upload.Unmap(0, None);
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let alloc: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
            .unwrap();
        list.CopyBufferRegion(&data, 0, &upload, 0, in_bytes);
        list.ResourceBarrier(&[transition(
            &data,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        let base = data.GetGPUVirtualAddress();
        let commands = out.GetGPUVirtualAddress();
        let constants = plan(commands);
        record(
            &list,
            &c,
            &constants,
            base,
            count.map(|_| base + count_at as u64),
            commands,
        );
        list.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyResource(&readback, &out);
        list.Close().unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        while fence.GetCompletedValue() < 1 {
            device.GetDeviceRemovedReason().unwrap();
            std::thread::yield_now();
        }
        let mut p = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut p)).unwrap();
        let v = std::slice::from_raw_parts(p.cast::<u32>(), out_dwords).to_vec();
        readback.Unmap(0, None);
        v
    }
}

/// A mesh program of 64 lanes, 64 vertices of 8 dwords, 32 triangles.
fn layout(primitive_dwords: u32) -> victory_shader::emit::MeshLayout {
    victory_shader::emit::MeshLayout {
        threads: 64,
        max_vertices: 64,
        max_primitives: 32,
        vertex_dwords: 8,
        primitive_dwords,
    }
}

/// Three records of 2, 0 and 5 groups (12-byte records, 16 bytes apart);
/// the count says the first two only, or all of them without a count.
fn check(expanded: bool, count: Option<u32>) {
    let Some(device) = device() else {
        eprintln!("no D3D12 device; skipped");
        return;
    };
    let layout = layout(if expanded { 4 } else { 0 });
    let regions = regions(&layout);
    let data = 0x1_0000_0000u64;
    let records = 3;
    let draws = records * COMPUTE_STRIDE;
    let args = [2, 1, 1, 0, 0, 1, 1, 0, 5, 1, 1, 0];
    let out_dwords = ((draws + records * draw_stride(expanded)) / 4) as usize;
    let words = run(
        &device,
        &args,
        (records, 16),
        count,
        |_| {
            plan(
                (records, 16, count.is_some()),
                &layout,
                &regions,
                expanded,
                data,
                draws,
            )
        },
        out_dwords,
    );
    let n = count.unwrap_or(records);
    let groups = [2, 0, 5].map(|g| g as u64);
    let mut used = 0u64;
    let [vb, ib, pb] = layout.bytes(1);
    for i in 0..records as usize {
        let g = if (i as u32) < n { groups[i] } else { 0 };
        let c = &words[i * 10..i * 10 + 10];
        let address = |w: &[u32]| u64::from(w[0]) | u64::from(w[1]) << 32;
        assert_eq!(address(&c[0..2]), data + used * vb, "record {i}: u0");
        assert_eq!(
            address(&c[2..4]),
            data + regions.indices + used * ib,
            "record {i}: u1"
        );
        assert_eq!(
            address(&c[4..6]),
            data + regions.prims + used * pb,
            "record {i}: u2"
        );
        assert_eq!(c[6..9], [g as u32, 1, 1], "record {i}: dispatch");
        let at = (draws / 4) as usize + i * (draw_stride(expanded) / 4) as usize;
        let d = &words[at..at + (draw_stride(expanded) / 4) as usize];
        assert_eq!(address(&d[0..2]), data + used * vb, "record {i}: t0");
        let first = (used * 32 * 3) as u32;
        assert_eq!(
            d[2..6],
            [32 * 3 * g as u32, 1, first, 0],
            "record {i}: draw"
        );
        used += g;
    }
}

#[test]
fn indexed_records_get_their_own_output() {
    check(false, None);
}

#[test]
fn expanded_records_get_their_own_output() {
    check(true, None);
}

#[test]
fn records_past_the_count_run_empty() {
    check(false, Some(2));
}

#[test]
fn the_buffer_fits_its_regions() {
    for prims in [0, 4] {
        let l = layout(prims);
        let r = regions(&l);
        let [v, i, p] = l.bytes(r.capacity);
        assert!(r.indices >= v && r.prims >= r.indices + i && r.bytes >= r.prims + p);
        assert!(r.bytes <= BYTES, "{r:?}");
        assert_eq!(r.index_bytes, i);
    }
}
