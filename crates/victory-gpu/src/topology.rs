//! Expands indexed topologies after earlier GPU writes in the same list.
//! IA reads the source just like an ordinary indexed draw. A point-list
//! stream-output pass captures the original indices, and compute expands and
//! packs those indices into a private buffer for the title draw.

use std::mem::ManuallyDrop;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_POINTLIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R16_UINT, DXGI_SAMPLE_DESC};
use windows::core::{PCSTR, s};

use crate::host::{Host, HostList, transition};
use crate::primitive::ExpandedTopology;

const CAPTURE: &str = r#"
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | ALLOW_STREAM_OUTPUT)")]
uint main(uint index : SV_VertexID) : VALUE { return index; }
"#;

// Work is split into bounded chunks. When restart is enabled, an inclusive
// maximum scan gives each chunk its preceding restart and a sum scan gives
// its output offset. The final pass packs the result and pads its tail.

const EXPAND: &str = r#"
cbuffer Draw : register(b0) {
    uint length;
    uint chunks;
    uint kind;
    uint cut;
    uint restart_enabled;
    uint capacity;
    uint fallback;
    uint packed16;
    uint mode;
    uint src_bank;
    uint dst_bank;
    uint step;
    uint cut_bank;
    uint count_bank;
    uint expanded;
    uint groups_width;
};
ByteAddressBuffer captured : register(t0);
RWByteAddressBuffer work : register(u0);
RWByteAddressBuffer output : register(u1);
static const uint CHUNK_SIZE = 256;

uint index_at(uint i) { return captured.Load(i * 4); }

// Number of outputs before a chunk begins when there are no cuts. A loop's
// final edge is emitted at the end of the full run, not at each boundary.
uint no_restart_prefix(uint first) {
    if (kind == 0) return (max(first, 2u) - 2) * 3;
    if (kind == 1) return (max(first, 1u) - 1) * 2;
    if (kind == 2) return (first / 4) * 6;
    return ((max(first, 2u) - 2) / 2) * 6;
}

void emit_pair(uint2 pair, inout uint out_at, bool emit) {
    if (emit) work.Store2(expanded + out_at * 4, pair);
    out_at += 2;
}

void emit_triangle(uint3 vertices, inout uint out_at, bool emit) {
    if (emit) work.Store3(expanded + out_at * 4, vertices);
    out_at += 3;
}

void expand_chunk(uint block, bool emit) {
    uint first = block * CHUNK_SIZE;
    uint end = min(first + CHUNK_SIZE, length);
    uint start = 0;
    uint out_at = 0;
    if (restart_enabled != 0 && block != 0) {
        start = work.Load(cut_bank + (block - 1) * 4);
        if (emit) out_at = work.Load(count_bank + (block - 1) * 4);
    }
    if (restart_enabled == 0) out_at = no_restart_prefix(first);

    for (uint i = first; i < end; ++i) {
        uint value = index_at(i);
        if (restart_enabled != 0 && value == cut) {
            start = i + 1;
        } else {
            uint at = i - start;
            if (kind == 0 && at >= 2) {
                emit_triangle(uint3(index_at(start), index_at(i - 1), value), out_at, emit);
            } else if (kind == 1 && at >= 1) {
                emit_pair(uint2(index_at(i - 1), value), out_at, emit);
                bool last = i + 1 == length;
                if (restart_enabled != 0 && !last) last = index_at(i + 1) == cut;
                if (last) emit_pair(uint2(value, index_at(start)), out_at, emit);
            } else if (kind == 2 && at % 4 == 3) {
                uint a = index_at(i - 3);
                uint b = index_at(i - 2);
                uint c = index_at(i - 1);
                emit_triangle(uint3(a, b, c), out_at, emit);
                emit_triangle(uint3(a, c, value), out_at, emit);
            } else if (kind == 3 && at >= 3 && (at & 1) != 0) {
                uint a = index_at(i - 3);
                uint b = index_at(i - 2);
                uint c = index_at(i - 1);
                emit_triangle(uint3(a, b, value), out_at, emit);
                emit_triangle(uint3(a, value, c), out_at, emit);
            }
        }
    }
    if (!emit) work.Store(dst_bank + block * 4, out_at);
}

uint output_index(uint i, uint count, uint pad) {
    if (i < count) return work.Load(expanded + i * 4);
    return pad;
}

[RootSignature("RootConstants(num32BitConstants=16, b0), SRV(t0), UAV(u0), UAV(u1)")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint id = tid.x + tid.y * groups_width * 64;
    if (mode == 5) {
        uint per_word = packed16 != 0 ? 2 : 1;
        uint first = id * per_word;
        if (first >= capacity) return;
        uint live = capacity;
        if (restart_enabled != 0) live = work.Load(count_bank + (chunks - 1) * 4);
        uint pad = fallback;
        if (live != 0) pad = work.Load(expanded);
        uint value = output_index(first, live, pad);
        if (packed16 != 0 && first + 1 < capacity) {
            value |= output_index(first + 1, live, pad) << 16;
        }
        output.Store(id * 4, value);
        return;
    }
    if (id >= chunks) return;
    if (mode == 0) {
        uint first = id * CHUNK_SIZE;
        uint end = min(first + CHUNK_SIZE, length);
        uint latest = 0;
        for (uint i = first; i < end; ++i) {
            if (index_at(i) == cut) latest = i + 1;
        }
        work.Store(dst_bank + id * 4, latest);
    } else if (mode == 1 || mode == 3) {
        uint value = work.Load(src_bank + id * 4);
        if (id >= step) {
            uint previous = work.Load(src_bank + (id - step) * 4);
            value = mode == 1 ? max(value, previous) : value + previous;
        }
        work.Store(dst_bank + id * 4, value);
    } else {
        expand_chunk(id, mode == 4);
    }
}
"#;

const CHUNK_SIZE: u32 = 256;

struct Pipelines {
    capture_root: ID3D12RootSignature,
    capture: ID3D12PipelineState,
    expand_root: ID3D12RootSignature,
    expand: ID3D12PipelineState,
    zero: ID3D12Resource,
}

// SAFETY: the D3D12 objects are free-threaded; zero is initialized before
// publication and never mapped again.
unsafe impl Send for Pipelines {}
// SAFETY: as above.
unsafe impl Sync for Pipelines {}

fn compile(source: &str, profile: PCSTR) -> windows::core::Result<ID3DBlob> {
    let (mut code, mut errors) = (None, None);
    // SAFETY: the source and names and both output pointers are live.
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("topology"),
            None,
            None,
            s!("main"),
            profile,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(error) = result {
        if let Some(errors) = errors {
            // SAFETY: the error blob owns this range.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("topology shader: {}", String::from_utf8_lossy(bytes));
        }
        return Err(error);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn make(host: &Host) -> windows::core::Result<Pipelines> {
    let vs = compile(CAPTURE, s!("vs_5_1"))?;
    let cs = compile(EXPAND, s!("cs_5_1"))?;
    let (zero, zero_ptr) = host
        .upload_buffer(8)
        .ok_or_else(windows::core::Error::empty)?;
    let output = D3D12_SO_DECLARATION_ENTRY {
        SemanticName: s!("VALUE"),
        ComponentCount: 1,
        ..Default::default()
    };
    let stride = 4;
    // SAFETY: setup inputs stay alive through object creation; zero owns an
    // eight-byte mapped range and nobody may see it before this returns.
    unsafe {
        std::ptr::write_bytes(zero_ptr, 0, 8);
        let vs_bytes =
            std::slice::from_raw_parts(vs.GetBufferPointer().cast::<u8>(), vs.GetBufferSize());
        let capture_root: ID3D12RootSignature = host.device.CreateRootSignature(0, vs_bytes)?;
        let capture =
            host.device
                .CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&capture_root),
                    VS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: vs_bytes.as_ptr().cast(),
                        BytecodeLength: vs_bytes.len(),
                    },
                    StreamOutput: D3D12_STREAM_OUTPUT_DESC {
                        pSODeclaration: &output,
                        NumEntries: 1,
                        pBufferStrides: &stride,
                        NumStrides: 1,
                        RasterizedStream: D3D12_SO_NO_RASTERIZED_STREAM,
                    },
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        ..Default::default()
                    },
                    SampleMask: u32::MAX,
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                })?;
        let cs_bytes =
            std::slice::from_raw_parts(cs.GetBufferPointer().cast::<u8>(), cs.GetBufferSize());
        let expand_root: ID3D12RootSignature = host.device.CreateRootSignature(0, cs_bytes)?;
        let expand =
            host.device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&expand_root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: cs_bytes.as_ptr().cast(),
                        BytecodeLength: cs_bytes.len(),
                    },
                    ..Default::default()
                })?;
        Ok(Pipelines {
            capture_root,
            capture,
            expand_root,
            expand,
            zero,
        })
    }
}

fn pipelines(host: &Host) -> Option<&'static Pipelines> {
    static PIPELINES: OnceLock<Option<Pipelines>> = OnceLock::new();
    PIPELINES
        .get_or_init(|| match make(host) {
            Ok(pipelines) => Some(pipelines),
            Err(error) => {
                tracing::warn!(%error, "indexed topology expansion unavailable");
                None
            }
        })
        .as_ref()
}

#[derive(Default)]
pub(crate) struct Scratch {
    capture: Option<(ID3D12Resource, u64)>,
    work: Option<(ID3D12Resource, u64)>,
    output: Option<(ID3D12Resource, u64)>,
}

fn buffer(
    host: &Host,
    list: &HostList,
    reuse: &mut Option<(ID3D12Resource, u64)>,
    size: u64,
    uav: bool,
) -> Option<ID3D12Resource> {
    if let Some((ready, capacity)) = reuse
        && *capacity >= size
    {
        return Some(ready.clone());
    }
    // Avoid reallocating when a later draw grows within the same heap page.
    let size = size.next_multiple_of(64 * 1024);
    let mut resource: Option<ID3D12Resource> = None;
    let (flags, initial) = if uav {
        (
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )
    } else {
        (D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)
    };
    // SAFETY: plain buffer allocation with a writable COM output parameter.
    unsafe {
        host.device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            },
            initial,
            None,
            &mut resource,
        )
    }
    .ok()?;
    let resource = resource?;
    if let Some((previous, _)) = reuse.replace((resource.clone(), size)) {
        // Earlier draws in this recording can still reference it.
        list.keep(&previous);
    }
    Some(resource)
}

pub(crate) struct Draw {
    pub(crate) source: D3D12_INDEX_BUFFER_VIEW,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) kind: ExpandedTopology,
    pub(crate) expanded: u32,
    pub(crate) cut: Option<u32>,
    pub(crate) pad: u32,
}

pub(crate) struct Expanded {
    pub(crate) view: D3D12_INDEX_BUFFER_VIEW,
    output: ID3D12Resource,
}

impl Expanded {
    pub(crate) fn finish(self, list: &HostList) {
        // SAFETY: the expansion left its private output in INDEX_BUFFER.
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &self.output,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }
}

fn work_barrier(work: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: the borrowed pointer lives beyond the barrier call.
                pResource: unsafe { std::mem::transmute_copy(work) },
            }),
        },
    }
}

fn compute(
    l: &ID3D12GraphicsCommandList,
    pipelines: &Pipelines,
    draw: &Draw,
    captured: &ID3D12Resource,
    work: &ID3D12Resource,
    output: &ID3D12Resource,
) {
    let chunks = draw.count.div_ceil(CHUNK_SIZE);
    let bank_bytes = chunks * 4;
    let expanded = bank_bytes * 4;
    let mut cut_bank = 0;
    let mut count_bank = 2 * bank_bytes;
    let packed16 = draw.source.Format == DXGI_FORMAT_R16_UINT;
    let kind = match draw.kind {
        ExpandedTopology::TriangleFan => 0,
        ExpandedTopology::LineLoop => 1,
        ExpandedTopology::Quads => 2,
        ExpandedTopology::QuadStrip => 3,
    };
    // SAFETY: the capture has finished and the private buffers remain bound
    // for the lifetime of the executing list.
    unsafe {
        l.SetComputeRootSignature(&pipelines.expand_root);
        l.SetPipelineState(&pipelines.expand);
        l.SetComputeRootShaderResourceView(1, captured.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(2, work.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(3, output.GetGPUVirtualAddress());
    }
    let dispatch = |mode, threads: u32, src, dst, step, cuts, counts| {
        let groups = threads.div_ceil(64);
        let width = groups.min(65_535);
        let params = [
            draw.count,
            chunks,
            kind,
            draw.cut.unwrap_or(0),
            u32::from(draw.cut.is_some()),
            draw.expanded,
            draw.pad,
            u32::from(packed16),
            mode,
            src,
            dst,
            step,
            cuts,
            counts,
            expanded,
            width,
        ];
        // SAFETY: each dispatch uses the compute bindings set above.
        unsafe {
            l.SetComputeRoot32BitConstants(0, params.len() as u32, params.as_ptr().cast(), 0);
            l.Dispatch(width, groups.div_ceil(width), 1);
        }
    };
    // SAFETY: work stayed in UAV state across the stages and previous use.
    let order = || unsafe { l.ResourceBarrier(&[work_barrier(work)]) };
    order();
    if draw.cut.is_some() {
        dispatch(0, chunks, 0, cut_bank, 0, cut_bank, count_bank);
        order();
        let scan = |mode, first_bank, second_bank, cuts, counts| {
            let (mut read, mut write) = (first_bank, second_bank);
            let mut step = 1;
            while step < chunks {
                dispatch(mode, chunks, read, write, step, cuts, counts);
                order();
                std::mem::swap(&mut read, &mut write);
                step *= 2;
            }
            read
        };
        cut_bank = scan(1, cut_bank, bank_bytes, cut_bank, count_bank);
        dispatch(2, chunks, 0, count_bank, 0, cut_bank, count_bank);
        order();
        count_bank = scan(3, count_bank, 3 * bank_bytes, cut_bank, count_bank);
    }
    dispatch(4, chunks, 0, 0, 0, cut_bank, count_bank);
    order();
    let words = if packed16 {
        draw.expanded.div_ceil(2)
    } else {
        draw.expanded
    };
    dispatch(5, words, 0, 0, 0, cut_bank, count_bank);
}

/// Records the capture and expansion before the caller rebinds its draw.
/// The caller restores the output to its reusable state with Expanded::finish.
pub(crate) fn expand(list: &HostList, draw: &Draw) -> Option<Expanded> {
    let chunks = draw.count.div_ceil(CHUNK_SIZE);
    let captured_bytes = u64::from(draw.count) * 4;
    let counter = captured_bytes.next_multiple_of(8);
    let work_size = u64::from(chunks) * 16 + u64::from(draw.expanded) * 4;
    let bytes = u64::from(draw.expanded)
        * if draw.source.Format == DXGI_FORMAT_R16_UINT {
            2
        } else {
            4
        };
    // Raw shader buffer offsets and the stream-output counter are 32-bit.
    if counter + 8 > u64::from(u32::MAX) || work_size > u64::from(u32::MAX) {
        return None;
    }

    let host = crate::host::host()?;
    let p = pipelines(host)?;
    let (captured, work, output) = {
        let mut scratch = list
            .topology_scratch
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        (
            buffer(host, list, &mut scratch.capture, counter + 8, false)?,
            buffer(host, list, &mut scratch.work, work_size, true)?,
            buffer(
                host,
                list,
                &mut scratch.output,
                bytes.next_multiple_of(4),
                true,
            )?,
        )
    };

    let l = &list.list;
    // SAFETY: these are live buffers with the documented stable states, and
    // the index buffer is valid for an ordinary IA read on this list.
    let view = unsafe {
        l.CopyBufferRegion(&captured, counter, &p.zero, 0, 8);
        l.ResourceBarrier(&[transition(
            &captured,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_STREAM_OUT,
        )]);
        let gpu = captured.GetGPUVirtualAddress();
        l.SetGraphicsRootSignature(&p.capture_root);
        l.SetPipelineState(&p.capture);
        l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
        l.IASetIndexBuffer(Some(&draw.source));
        l.SOSetTargets(
            0,
            Some(&[D3D12_STREAM_OUTPUT_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: captured_bytes,
                BufferFilledSizeLocation: gpu + counter,
            }]),
        );
        l.DrawIndexedInstanced(draw.count, 1, draw.first, 0, 0);
        l.SOSetTargets(0, Some(&[D3D12_STREAM_OUTPUT_BUFFER_VIEW::default()]));
        l.ResourceBarrier(&[transition(
            &captured,
            D3D12_RESOURCE_STATE_STREAM_OUT,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )]);
        compute(l, p, draw, &captured, &work, &output);
        l.ResourceBarrier(&[
            transition(
                &captured,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ),
            transition(
                &output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDEX_BUFFER,
            ),
        ]);
        D3D12_INDEX_BUFFER_VIEW {
            BufferLocation: output.GetGPUVirtualAddress(),
            SizeInBytes: bytes as u32,
            Format: draw.source.Format,
        }
    };
    Some(Expanded { view, output })
}
