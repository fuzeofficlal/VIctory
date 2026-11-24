//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler; a 5.1 shader with its root signature inside).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint unused; };
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// Clamps the record at `at` from `src` into `dst`; its cost.
uint clamped(uint at) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances, plus start offsets.
        uint count = min(src.Load(at), 1u << 24);
        uint instances = min(src.Load(at + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dst.Store4(at, uint4(count, instances, src.Load(at + 8), src.Load(at + 12)));
        if (kind == 1) {
            dst.Store(at + 16, src.Load(at + 16));
        }
    } else if (kind == 2) {
        // D3D12 allows at most 65,535 groups per dimension; anything above is
        // unwritten/garbage memory (pointers, floats, -1), not a valid count.
        uint3 g = src.Load3(at);
        if (any(g > 65535u) || (g.z != 0 && g.x * g.y > GROUPS / g.z)) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dst.Store3(at, g);
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = min(src.Load3(at), uint3(65535, 1, 1));
        cost = g.x * g.y * g.z * 128;
        dst.Store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget.Load(0), BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * stride;
        uint cost = 0;
        if (record < records) {
            cost = clamped(at);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            dst.Store(at, 0);
        }
    }
    if (lane == 0) {
        budget.Store(0, BUDGET - left);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_on(device)
}

fn make_on(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart from host address `src_gpu` into
/// `list`'s own indirect-argument buffer, and leaves that output buffer ready
/// to be read by `ExecuteIndirect` (returning `(buffer, offset)`). Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). `None` if the clamp is not available or misaligned.
pub(crate) fn bound(
    list: &HostList,
    src_gpu: u64,
    (kind, stride, records): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    let (c, budget) = (clamp()?, list.budget()?);
    if !src_gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let bytes = u64::from(stride) * u64::from(records.max(1));
    let (dst, dst_offset, dst_gpu) = list.indirect_args(bytes)?;
    record_bound(
        &list.list,
        c,
        src_gpu,
        (&dst, dst_gpu),
        budget,
        (kind, stride, records),
    );
    Some((dst, dst_offset))
}

fn record_bound(
    l: &ID3D12GraphicsCommandList,
    c: &Clamp,
    src_gpu: u64,
    (dst, dst_gpu): (&ID3D12Resource, u64),
    budget: u64,
    (kind, stride, records): (u32, u32, u32),
) {
    let constants = [kind, stride, records, 0];
    // SAFETY: an open list; our root signature, pipeline and live buffers.
    unsafe {
        // Earlier shader writes (the title's argument passes) finish before
        // `src` is read as `t0`, without transitioning the shared shadow buffer.
        l.ResourceBarrier(&[uav_barrier()]);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootUnorderedAccessView(2, dst_gpu);
        l.SetComputeRootUnorderedAccessView(3, budget);
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
}

/// After the execution: the list's clamped argument buffer returns to
/// unordered access for later indirect calls in the same list.
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    done_on(&list.list, buffer);
}

fn done_on(l: &ID3D12GraphicsCommandList, buffer: &ID3D12Resource) {
    // SAFETY: an open list and a live buffer.
    unsafe {
        l.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )])
    };
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word after (zero when
    /// it runs empty). Both the records in the source buffer and one record
    /// more after them must stay as written. `None` without a host GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let out_bytes = (records as usize * per * 4) as u64;
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let (readback, out_data) = host.shared_buffer(out_bytes.max(16)).unwrap();
        let data = data.cast::<u32>();
        let out_data = out_data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffers, written before the list runs and read
        // after it has; live buffers.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut last = None;
            for _ in 0..calls {
                let (dst, off) = super::bound(&list, gpu, spec).expect("bound");
                super::done(&list, &dst);
                last = Some((dst, off));
            }
            let (dst, off) = last.unwrap();
            list.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.list
                .CopyBufferRegion(&readback, 0, &dst, off, out_bytes);
            list.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()]);
            }
            let first = |i: usize| out_data.add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    /// End-to-end indirect dispatch (`kind == 2`) across every SM 6.6 hardware
    /// adapter and WARP under the D3D12 debug layer:
    /// 1) Rejects unwritten/garbage records (`> 65535` float/pointer/`-1` and
    ///    `> GROUPS` totals) to `(0, 0, 0)` instead of clamping to `65535`.
    /// 2) Preserves same-list GPU UAV writes into a shared buffer that was
    ///    earlier promoted to `VERTEX_AND_CONSTANT_BUFFER`, with zero debug
    ///    layer barrier errors and without mutating the source buffer.
    /// 3) Zeroes output records past `BUDGET` while leaving the source intact.
    #[test]
    fn indirect_dispatch_guards_garbage_and_preserves_same_list_writes() {
        const CHILD_ENV: &str = "VICTORY_INDIRECT_DEBUG_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let exe = std::env::current_exe().expect("current_exe");
            let out = std::process::Command::new(exe)
                .env(CHILD_ENV, "1")
                .args([
                    "--exact",
                    "indirect::tests::indirect_dispatch_guards_garbage_and_preserves_same_list_writes",
                    "--nocapture",
                ])
                .output()
                .expect("spawn debug-layer child");
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);
            eprint!("{stdout}{stderr}");
            assert!(
                out.status.success(),
                "debug-layer child failed ({}):\n{stdout}\n{stderr}",
                out.status
            );
            return;
        }

        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS,
            IDXGIAdapter, IDXGIFactory4,
        };
        use windows::core::Interface;

        let mut debug_enabled = false;
        // SAFETY: called in a fresh child process before any D3D12 device exists.
        unsafe {
            let mut dbg: Option<ID3D12Debug> = None;
            if D3D12GetDebugInterface(&mut dbg).is_ok()
                && let Some(d) = dbg
            {
                d.EnableDebugLayer();
                debug_enabled = true;
            }
        }
        eprintln!("D3D12 debug layer enabled in child: {debug_enabled}");

        // Compile small synthetic producer & consumer CS 5.1 shaders via D3DCompile:
        // - Producer writes `(3, 2, 1)` into record 0 of `u0` (shared buffer).
        // - Consumer (`[numthreads(4, 1, 1)]`) atomically adds 1 per thread to `u0[0]`.
        const PRODUCER_HLSL: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer out_args : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    out_args.Store3(0, uint3(3, 2, 1));
}
"#;
        const CONSUMER_HLSL: &str = r#"
#define RS "UAV(u0)"
globallycoherent RWByteAddressBuffer counter : register(u0);
[RootSignature(RS)]
[numthreads(4, 1, 1)]
void main() {
    uint old;
    counter.InterlockedAdd(0, 1u, old);
}
"#;
        let compile_cs =
            |dev: &ID3D12Device, hlsl: &str| -> (ID3D12RootSignature, ID3D12PipelineState) {
                // SAFETY: valid HLSL slice and D3D12 root/PSO creation.
                unsafe {
                    let mut code: Option<ID3DBlob> = None;
                    D3DCompile(
                        hlsl.as_ptr().cast(),
                        hlsl.len(),
                        s!("test_cs"),
                        None,
                        None,
                        s!("main"),
                        s!("cs_5_1"),
                        0,
                        0,
                        &mut code,
                        None,
                    )
                    .expect("D3DCompile test_cs");
                    let code = code.unwrap();
                    let bytes = std::slice::from_raw_parts(
                        code.GetBufferPointer().cast::<u8>(),
                        code.GetBufferSize(),
                    );
                    let root: ID3D12RootSignature = dev.CreateRootSignature(0, bytes).unwrap();
                    let pso = dev
                        .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                            pRootSignature: std::mem::transmute_copy(&root),
                            CS: D3D12_SHADER_BYTECODE {
                                pShaderBytecode: bytes.as_ptr().cast(),
                                BytecodeLength: bytes.len(),
                            },
                            ..Default::default()
                        })
                        .unwrap();
                    (root, pso)
                }
            };

        let mut adapters: Vec<(String, ID3D12Device)> = Vec::new();
        // SAFETY: plain DXGI adapter enumeration and D3D12 device creation.
        unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            let mut i = 0;
            while let Ok(ad) = factory.EnumAdapters1(i) {
                i += 1;
                let Ok(desc) = ad.GetDesc1() else { continue };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0
                    || desc.DedicatedVideoMemory == 0
                {
                    continue;
                }
                let len = desc
                    .Description
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..len]);
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&ad, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok()
                    && let Some(d) = dev
                {
                    adapters.push((name, d));
                }
            }
            if let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() {
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_12_0, &mut dev).is_ok()
                    && let Some(d) = dev
                {
                    adapters.push(("WARP".into(), d));
                }
            }
        }
        assert!(!adapters.is_empty(), "no D3D12 adapter available");

        for (label, dev) in &adapters {
            let clamp_pso = make_on(dev).unwrap_or_else(|e| panic!("{label}: make_on: {e}"));
            let (prod_rs, prod_pso) = compile_cs(dev, PRODUCER_HLSL);
            let (cons_rs, cons_pso) = compile_cs(dev, CONSUMER_HLSL);

            // SAFETY: valid D3D12 resource, command signature, dispatch, and info-queue calls.
            unsafe {
                let iq = dev.cast::<ID3D12InfoQueue>().ok();
                if let Some(iq) = &iq {
                    iq.ClearStoredMessages();
                }

                let make_buf = |heap_props, width, flags, state| -> ID3D12Resource {
                    let mut out: Option<ID3D12Resource> = None;
                    dev.CreateCommittedResource(
                        &heap_props,
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: width,
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
                        state,
                        None,
                        &mut out,
                    )
                    .expect("CreateCommittedResource");
                    out.unwrap()
                };
                let shared_heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_CUSTOM,
                    CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                    MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                    ..Default::default()
                };
                let default_heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                };
                let readback_heap = D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                };

                // Shared shadow-like buffer holding:
                // - [0..72):   6 compute dispatch records (`kind == 2`)
                // - [80..116): 3 mesh dispatch records (`kind == 10`)
                // - [120..140): 1 indexed draw record (`kind == 1`)
                // - [192..256): vertex buffer slice bound simultaneously
                let src_buf = make_buf(
                    shared_heap,
                    256,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                let mut sp = std::ptr::null_mut();
                src_buf.Map(0, None, Some(&mut sp)).unwrap();
                let initial_records: [[u32; 3]; 6] = [
                    // Record 0: garbage overwritten in-list by producer CS -> (3, 2, 1) = 6 groups.
                    [0xDEAD_BEEF, 0xCAFE_BABE, 0xFFFF_FFFF],
                    // Record 1: unwritten float garbage (1.0f, 1, 1) -> must zero to (0, 0, 0).
                    [0x3F80_0000, 1, 1],
                    // Record 2: unwritten 64-bit VA garbage -> must zero to (0, 0, 0).
                    [0xC799_B400, 0xD, 1],
                    // Record 3: unwritten -1 sentinel -> must zero to (0, 0, 0).
                    [u32::MAX, u32::MAX, u32::MAX],
                    // Record 4: > GROUPS (2048 * 1024 * 1 > 2^20) -> must zero to (0, 0, 0).
                    [2048, 1024, 1],
                    // Record 5: valid CPU-initialized count (2, 1, 1) = 2 groups.
                    [2, 1, 1],
                ];
                let mesh_records: [[u32; 3]; 3] = [[4, 1, 1], [4, 2, 1], [70_000, 1, 1]];
                let indexed_record: [u32; 5] = [100, 2, 10, 20, 30];
                sp.cast::<[[u32; 3]; 6]>().write(initial_records);
                sp.cast::<u8>()
                    .add(80)
                    .cast::<[[u32; 3]; 3]>()
                    .write(mesh_records);
                sp.cast::<u8>()
                    .add(120)
                    .cast::<[u32; 5]>()
                    .write(indexed_record);
                src_buf.Unmap(0, None);

                let dst_buf = make_buf(
                    default_heap,
                    512,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                let budget_buf = make_buf(
                    shared_heap,
                    256,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                let mut bp = std::ptr::null_mut();
                budget_buf.Map(0, None, Some(&mut bp)).unwrap();
                bp.cast::<[u32; 4]>().write([0, (1u32 << 27) - 64, 0, 0]);
                budget_buf.Unmap(0, None);

                let counter_buf = make_buf(
                    default_heap,
                    16,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                let rb_buf = make_buf(
                    readback_heap,
                    512,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );

                let arg_desc = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
                    ..Default::default()
                };
                let mut cmd_sig: Option<ID3D12CommandSignature> = None;
                dev.CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 12,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &arg_desc,
                        NodeMask: 0,
                    },
                    None::<&ID3D12RootSignature>,
                    &mut cmd_sig,
                )
                .expect("CreateCommandSignature");
                let cmd_sig = cmd_sig.unwrap();

                let queue: ID3D12CommandQueue = dev
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .unwrap();
                let alloc: ID3D12CommandAllocator = dev
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = dev
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                    .unwrap();

                // 1) Producer CS writes `(3, 2, 1)` into record 0 of `src_buf`.
                list.SetComputeRootSignature(&prod_rs);
                list.SetPipelineState(&prod_pso);
                list.SetComputeRootUnorderedAccessView(0, src_buf.GetGPUVirtualAddress());
                list.Dispatch(1, 1, 1);

                // 2) Clamp 6 records from `src_buf` into `dst_buf[0..72)` with fresh budget,
                //    then feed every clamped slot individually to `ExecuteIndirect(max = 1)`
                //    (AMD's Windows D3D12 driver executes 1 dispatch per call) so all 6 slots
                //    are genuinely consumed by the command processor.
                record_bound(
                    &list,
                    &clamp_pso,
                    src_buf.GetGPUVirtualAddress(),
                    (&dst_buf, dst_buf.GetGPUVirtualAddress()),
                    budget_buf.GetGPUVirtualAddress(),
                    (2, 12, 6),
                );
                list.SetComputeRootSignature(&cons_rs);
                list.SetPipelineState(&cons_pso);
                list.SetComputeRootUnorderedAccessView(0, counter_buf.GetGPUVirtualAddress());
                for slot in 0..6u64 {
                    list.ExecuteIndirect(
                        &cmd_sig,
                        1,
                        &dst_buf,
                        slot * 12,
                        None::<&ID3D12Resource>,
                        0,
                    );
                }
                done_on(&list, &dst_buf);

                // 3) Second indirect call in the SAME list with budget at `BUDGET - 64`
                //    (room for 1 group = 64 threads; record 5 asks for 2 groups = 128 -> zeroed!).
                //    First bind `src_buf + 192` as a Vertex Buffer so `src_buf` holds a
                //    read state alongside `SRV(t0)`.
                list.IASetVertexBuffers(
                    0,
                    Some(&[D3D12_VERTEX_BUFFER_VIEW {
                        BufferLocation: src_buf.GetGPUVirtualAddress() + 192,
                        SizeInBytes: 64,
                        StrideInBytes: 16,
                    }]),
                );
                record_bound(
                    &list,
                    &clamp_pso,
                    src_buf.GetGPUVirtualAddress() + 60,
                    (&dst_buf, dst_buf.GetGPUVirtualAddress() + 128),
                    budget_buf.GetGPUVirtualAddress() + 4,
                    (2, 12, 1),
                );
                list.SetComputeRootSignature(&cons_rs);
                list.SetComputeRootUnorderedAccessView(0, counter_buf.GetGPUVirtualAddress());
                list.SetPipelineState(&cons_pso);
                list.ExecuteIndirect(&cmd_sig, 1, &dst_buf, 128, None::<&ID3D12Resource>, 0);
                done_on(&list, &dst_buf);

                // 4) Clamp mesh dispatch (`kind == 10`) and indexed draw (`kind == 1`) records.
                record_bound(
                    &list,
                    &clamp_pso,
                    src_buf.GetGPUVirtualAddress() + 80,
                    (&dst_buf, dst_buf.GetGPUVirtualAddress() + 192),
                    budget_buf.GetGPUVirtualAddress() + 8,
                    (10, 12, 3),
                );
                done_on(&list, &dst_buf);
                record_bound(
                    &list,
                    &clamp_pso,
                    src_buf.GetGPUVirtualAddress() + 120,
                    (&dst_buf, dst_buf.GetGPUVirtualAddress() + 256),
                    budget_buf.GetGPUVirtualAddress() + 12,
                    (1, 20, 1),
                );
                done_on(&list, &dst_buf);

                list.ResourceBarrier(&[
                    transition(
                        &counter_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                    transition(
                        &dst_buf,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ]);
                list.CopyBufferRegion(&rb_buf, 0, &counter_buf, 0, 4);
                list.CopyBufferRegion(&rb_buf, 16, &dst_buf, 0, 72);
                list.CopyBufferRegion(&rb_buf, 128, &dst_buf, 128, 12);
                list.CopyBufferRegion(&rb_buf, 192, &dst_buf, 192, 36);
                list.CopyBufferRegion(&rb_buf, 256, &dst_buf, 256, 20);
                list.Close().unwrap();

                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                queue.Signal(&fence, 1).unwrap();
                while fence.GetCompletedValue() < 1 {
                    std::thread::yield_now();
                }

                assert!(
                    dev.GetDeviceRemovedReason().is_ok(),
                    "{label}: device was removed"
                );
                if let Some(iq) = &iq {
                    let n = iq.GetNumStoredMessages();
                    let mut bad = Vec::new();
                    for i in 0..n {
                        let mut len = 0usize;
                        let _ = iq.GetMessage(i, None, &mut len);
                        if len == 0 {
                            continue;
                        }
                        let mut buf = vec![0u8; len];
                        let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                        if iq.GetMessage(i, Some(msg), &mut len).is_ok()
                            && matches!(
                                (*msg).Severity,
                                D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR
                            )
                        {
                            let d = std::slice::from_raw_parts(
                                (*msg).pDescription,
                                (*msg).DescriptionByteLength.saturating_sub(1),
                            );
                            bad.push(String::from_utf8_lossy(d).into_owned());
                        }
                    }
                    assert!(
                        bad.is_empty(),
                        "{label}: D3D12 debug layer reported errors: {bad:#?}"
                    );
                }

                let mut rp = std::ptr::null_mut();
                rb_buf.Map(0, None, Some(&mut rp)).unwrap();
                let threads_ran = rp.cast::<u32>().read();
                let clamped = rp.cast::<u8>().add(16).cast::<[[u32; 3]; 6]>().read();
                let over_budget = rp.cast::<u8>().add(128).cast::<[u32; 3]>().read();
                let clamped_mesh = rp.cast::<u8>().add(192).cast::<[[u32; 3]; 3]>().read();
                let clamped_indexed = rp.cast::<u8>().add(256).cast::<[u32; 5]>().read();
                rb_buf.Unmap(0, None);

                let mut sp2 = std::ptr::null_mut();
                src_buf.Map(0, None, Some(&mut sp2)).unwrap();
                let src_after = sp2.cast::<[[u32; 3]; 6]>().read();
                src_buf.Unmap(0, None);

                eprintln!(
                    "indirect guard on {label} (debug_layer={}, info_queue={}): threads_ran={threads_ran}, clamped={clamped:?}, over_budget={over_budget:?}, mesh={clamped_mesh:?}, indexed={clamped_indexed:?}",
                    debug_enabled,
                    iq.is_some()
                );

                // (6 + 0 + 0 + 0 + 0 + 2) * 4 threads/group = 32 threads total;
                // second ExecuteIndirect exceeded budget so added 0 threads.
                assert_eq!(threads_ran, 32, "{label}: consumer thread count");
                assert_eq!(
                    clamped,
                    [
                        [3, 2, 1],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [2, 1, 1],
                    ],
                    "{label}: clamped dispatch records"
                );
                assert_eq!(over_budget[0], 0, "{label}: over-budget x must be zeroed");
                assert_eq!(
                    clamped_mesh,
                    [[4, 1, 1], [4, 1, 1], [65_535, 1, 1]],
                    "{label}: clamped mesh records"
                );
                assert_eq!(
                    clamped_indexed,
                    [100, 2, 10, 20, 30],
                    "{label}: clamped indexed draw record"
                );
                // Source buffer was NOT overwritten by clamp or budget exhaustion!
                assert_eq!(src_after[0], [3, 2, 1], "{label}: producer write in src");
                assert_eq!(src_after[1], [0x3F80_0000, 1, 1], "{label}: src[1] intact");
                assert_eq!(
                    src_after[5],
                    [2, 1, 1],
                    "{label}: src[5] intact after budget cap"
                );
            }
        }
    }
}
