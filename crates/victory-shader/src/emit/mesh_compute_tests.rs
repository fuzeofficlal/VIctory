//! A mesh program run as compute ([`super::translate_mesh_as_compute`]):
//! a synthetic GFX10 mesh program lowered both ways; the compute form
//! dispatched to read back what it wrote, and drawn with the fallback
//! vertex shader to compare with the mesh shader's own image.

use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::bitcode::{Attr, Term};
use super::container::SigElement;
use super::host::Host;
use super::shader::Shader;
use super::{
    HostShader, MESH_OUT_SPACE, MeshLayout, TranslateOptions, Wave64Mode, lower, mesh_vs, validator,
};
use crate::native::{HwStage, MeshIo, NativeProgram, StageIo, Usage, VsOutput};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// Inline constants as operands.
const ZERO: u16 = 0x80;
const ONE: u16 = 0x81;
const MINUS_ONE: u16 = 0xC1;
const ONE_F: u16 = 0xF2;
const TWO_F: u16 = 0xF4;
/// A literal dword follows the instruction.
const LITERAL: u16 = 0xFF;

/// SOP1 `op sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u16) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | (src0 as u32 & 0xFF)
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u16, src1: u16) -> u32 {
    0x8000_0000 | (op as u32) << 23 | (sdst as u32) << 16 | (src1 as u32) << 8 | (src0 as u32)
}

/// VOPC `op src0, vsrc1`.
const fn vopc(op: u16, src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | (op as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP1 `op vdst, src0` (`src0`: below 256 SGPRs and constants, 256 on VGPRs).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// `exp target, v[srcs]` with the channels `srcs` names enabled.
fn exp(target: u8, srcs: [Option<u8>; 4], done: bool) -> [u32; 2] {
    let mut en = 0;
    let mut w1 = 0;
    for (k, s) in srcs.iter().enumerate() {
        if let Some(v) = s {
            en |= 1 << k;
            w1 |= u32::from(*v) << (8 * k);
        }
    }
    [
        0xF800_0000 | u32::from(done) << 11 | u32::from(target) << 4 | en,
        w1,
    ]
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// One triangle per group from three lanes. Lane n exports vertex n at
/// (n - 1, 2 (n & 1) - 1, 0, 1) with `TEXCOORD0` (n, 1, 2, 0), and
/// primitive n: lane 0 the triangle (0, 1, 2), lanes 1 and 2 culled ones;
/// with `prims`, primitive n also has the per-primitive `COLOR0`
/// (1, 2, n, 0).
fn program(prims: bool) -> NativeProgram {
    let mut w = Vec::new();
    w.extend(vop3(0x365, 0, MINUS_ONE, ZERO, 0)); // v_mbcnt_lo_u32_b32 v0, -1, 0: the lane
    w.push(vop1(0x006, 1, vgpr(0))); // v_cvt_f32_u32 v1, v0
    w.push(vop1(0x001, 2, TWO_F)); // v_mov_b32 v2, 2.0
    w.push(vop1(0x001, 3, ZERO)); // v_mov_b32 v3, 0
    w.push(vop1(0x001, 4, ONE_F)); // v_mov_b32 v4, 1.0
    w.push(vop2(0x05, 7, ONE_F, 1)); // v_subrev_f32 v7, 1.0, v1: x
    w.push(vop2(0x1b, 8, ONE, 0)); // v_and_b32 v8, 1, v0
    w.push(vop1(0x006, 8, vgpr(8))); // v_cvt_f32_u32 v8, v8
    w.push(vop2(0x08, 8, TWO_F, 8)); // v_mul_f32 v8, 2.0, v8
    w.push(vop2(0x05, 8, ONE_F, 8)); // v_subrev_f32 v8, 1.0, v8: y
    // v6 = 0x00200400 + lane * 0x7FDFFC00: lane 0 the indices 0, 1, 2;
    // lanes 1 and 2 have bit 31 set (culled).
    w.extend([s_mov_b32(10, LITERAL), 0x7FDF_FC00]);
    w.extend([s_mov_b32(11, LITERAL), 0x0020_0400]);
    w.extend(vop3(0x169, 6, vgpr(0), 10, 0)); // v_mul_lo_u32 v6, v0, s10
    w.push(vop2(0x25, 6, 11, 6)); // v_add_nc_u32 v6, s11, v6
    w.extend(exp(20, [Some(6), None, None, None], true)); // exp prim
    w.extend(exp(32, [Some(1), Some(4), Some(2), Some(3)], false)); // exp param0
    if prims {
        w.extend(exp(33, [Some(4), Some(2), Some(1), Some(3)], false)); // exp param1
    }
    w.extend(exp(12, [Some(7), Some(8), Some(3), Some(4)], true)); // exp pos0
    w.push(S_ENDPGM);
    let output = |row, param| VsOutput {
        usage: Usage::Generic,
        row,
        param,
        fp16: false,
        pack2: false,
    };
    let mut outputs = vec![output(1, 0)];
    if prims {
        // Bit 7: per-primitive, row 0 of the primitive signature.
        outputs.push(output(0, 0x81));
    }
    NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 12,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs,
            clip_cntl: 0,
            threads: 3,
            max_vertices: 3,
            max_primitives: 3,
            topology: 2,
            lds_bytes: 0,
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn element(name: &str, kind: u8, interp: u8, row: u32) -> SigElement {
    SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: false,
    }
}

fn outputs() -> Vec<SigElement> {
    vec![
        element("SV_Position", 3, 4, 0),
        element("TEXCOORD", 0, 2, 1),
    ]
}

/// The primitive signature (constant, as `emit` declares it).
fn prim_outputs(prims: bool) -> Vec<SigElement> {
    if prims {
        vec![element("COLOR", 0, 1, 0)]
    } else {
        Vec::new()
    }
}

fn validate(shader: &HostShader) {
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
}

fn shader(stage: HwStage, dxil: Vec<u8>) -> HostShader {
    HostShader {
        stage,
        dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
    }
}

/// The program lowered as a mesh shader or, with `as_compute`, a compute
/// shader with its layout; validated when `dxil.dll` is there.
fn lowered(as_compute: bool) -> (HostShader, MeshLayout) {
    lowered_with(as_compute, false)
}

fn lowered_with(as_compute: bool, prims: bool) -> (HostShader, MeshLayout) {
    lowered_with_max_primitives(as_compute, prims, 3)
}

fn lowered_with_max_primitives(
    as_compute: bool,
    prims: bool,
    max_primitives: u32,
) -> (HostShader, MeshLayout) {
    let mut native = program(prims);
    let StageIo::Ms(mesh) = &mut native.io else {
        unreachable!()
    };
    mesh.max_primitives = max_primitives;
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let outputs = outputs();
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: outputs.clone(),
        prims: prim_outputs(prims),
        mesh_stride: max_primitives,
        options: super::TranslateOptions::default(),
        mesh_as_compute: as_compute,
    })
    .expect("lower");
    let s = shader(HwStage::Ms, l.dxil);
    validate(&s);
    let layout = MeshLayout {
        threads: 3,
        max_vertices: 3,
        max_primitives,
        vertex_dwords: 4 * outputs.len() as u32,
        primitive_dwords: 4 * prim_outputs(prims).len() as u32,
    };
    (s, layout)
}

/// The fallback's vertex shader for the program.
fn fallback_vs(layout: &MeshLayout) -> HostShader {
    fallback_vs_with(layout, false)
}

fn fallback_vs_with(layout: &MeshLayout, prims: bool) -> HostShader {
    let vs = shader(
        HwStage::Vs,
        mesh_vs::vertex_shader(
            &outputs(),
            &prim_outputs(prims),
            layout.vertex_dwords,
            layout.primitive_dwords,
        ),
    );
    validate(&vs);
    vs
}

/// A pixel shader writing `TEXCOORD0` to target 0 or, with `prims`,
/// (`TEXCOORD0.xy`, `COLOR0.xy`): `COLOR0` a constant input after the
/// vertex outputs, linked to the per-primitive output by its semantic.
fn pixel_shader(prims: bool) -> HostShader {
    let mut s = Shader::new(true);
    let mut inputs = outputs();
    inputs[1].usage = if prims { 0x3 } else { 0xF };
    if prims {
        let mut color = element("COLOR", 0, 1, 2);
        color.usage = 0x3;
        inputs.push(color);
    }
    s.inputs = inputs;
    let e = s.entry;
    for col in 0..4u8 {
        // (input element, its channel) for target channel `col`.
        let (id, ch) = match (prims, col) {
            (true, 2 | 3) => (2, col - 2),
            _ => (1, col),
        };
        let args = [s.m.i32(id), s.m.i32(0), s.m.i8(ch), s.m.undef(s.t.i32)];
        let x = s.op(e, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
        let args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), x];
        s.op(e, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
    }
    s.outputs = vec![SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }];
    s.m.terminate(e, Term::Ret);
    let ps = shader(HwStage::Ps, s.finish());
    validate(&ps);
    ps
}

/// The program kind in the DXIL part's version word (1 vertex, 5 compute,
/// 13 mesh).
fn program_kind(container: &[u8]) -> u32 {
    let c = crate::dxbc::Container::parse(container).expect("container");
    let dxil = c.part(crate::dxbc::FourCC::DXIL).expect("DXIL part");
    u32::from_le_bytes(dxil.data[0..4].try_into().unwrap()) >> 16
}

#[test]
fn a_mesh_program_lowers_as_compute_too() {
    let (mesh, _) = lowered(false);
    let (cs, layout) = lowered(true);
    assert_eq!(program_kind(&mesh.dxil), 13);
    assert_eq!(program_kind(&cs.dxil), 5);
    assert_eq!(program_kind(&fallback_vs(&layout).dxil), 1);
    assert_eq!(layout.bytes(2), [2 * 3 * 8 * 4, 2 * 3 * 12, 0]);
}

/// D3D12 plumbing for the tests: one device, its queue, and helpers.
struct Gpu<'a> {
    dev: &'a ID3D12Device,
    queue: ID3D12CommandQueue,
    root: ID3D12RootSignature,
}

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
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

const SIZE: u32 = 32;
/// Root parameters: `u0`-`u2` (0-2) and `t0`-`t2` (3-5) in
/// [`MESH_OUT_SPACE`].
const VERTICES_SRV: u32 = 3;

impl<'a> Gpu<'a> {
    fn new(host: &'a Host) -> Result<Self, String> {
        let dev = &host.device;
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = (0..3)
            .map(|r| D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                        ShaderRegister: r,
                        RegisterSpace: MESH_OUT_SPACE,
                        Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            })
            .collect();
        params.extend((0..3).map(|r| D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: r,
                    RegisterSpace: MESH_OUT_SPACE,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }));
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                        | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                        | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
                    ..Default::default()
                },
            },
        };
        // SAFETY: plain D3D12 calls with descriptions that outlive them.
        unsafe {
            let mut blob = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(err)?;
            let blob: ID3DBlob = blob.ok_or("no root signature blob")?;
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            Ok(Gpu {
                dev,
                queue: dev
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .map_err(err)?,
                root: dev.CreateRootSignature(0, bytes).map_err(err)?,
            })
        }
    }

    fn resource(
        &self,
        heap: D3D12_HEAP_TYPE,
        desc: &D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
        clear: Option<&D3D12_CLEAR_VALUE>,
    ) -> Result<ID3D12Resource, String> {
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: a plain D3D12 call with a live description.
        unsafe {
            self.dev
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: heap,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    desc,
                    state,
                    clear.map(std::ptr::from_ref),
                    &mut r,
                )
                .map_err(err)?;
        }
        r.ok_or_else(|| "no resource".to_string())
    }

    fn buffer(
        &self,
        heap: D3D12_HEAP_TYPE,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<ID3D12Resource, String> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(4),
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
        };
        self.resource(heap, &desc, state, None)
    }

    /// A fresh command list (with no pipeline set).
    fn list(&self) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList6), String> {
        // SAFETY: plain D3D12 calls.
        unsafe {
            let alloc: ID3D12CommandAllocator = self
                .dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(err)?;
            let list: ID3D12GraphicsCommandList = self
                .dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .map_err(err)?;
            Ok((alloc, list.cast().map_err(err)?))
        }
    }

    /// Closes `list`, runs it and waits for the GPU.
    fn run(&self, list: &ID3D12GraphicsCommandList6) -> Result<(), String> {
        // SAFETY: plain D3D12 calls; the caller keeps the resources alive.
        unsafe {
            list.Close().map_err(err)?;
            self.queue
                .ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = self
                .dev
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .map_err(err)?;
            self.queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = self.dev.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    fn read(r: &ID3D12Resource, dwords: usize) -> Result<Vec<u32>, String> {
        // SAFETY: a mapped readback buffer of at least `dwords` dwords.
        unsafe {
            let mut p = std::ptr::null_mut();
            r.Map(0, None, Some(&mut p)).map_err(err)?;
            let v = std::slice::from_raw_parts(p.cast::<u32>(), dwords).to_vec();
            r.Unmap(0, None);
            Ok(v)
        }
    }

    /// The three mesh output buffers, in UNORDERED_ACCESS.
    fn mesh_buffers(
        &self,
        layout: &MeshLayout,
        groups: u32,
    ) -> Result<Vec<ID3D12Resource>, String> {
        layout
            .bytes(groups)
            .into_iter()
            .map(|size| {
                self.buffer(
                    D3D12_HEAP_TYPE_DEFAULT,
                    size,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )
            })
            .collect()
    }

    /// Records the compute pass writing `outs`.
    fn record_compute(
        &self,
        list: &ID3D12GraphicsCommandList6,
        pso: &ID3D12PipelineState,
        outs: &[ID3D12Resource],
        groups: u32,
    ) {
        // SAFETY: plain D3D12 recording; the caller keeps the objects alive.
        unsafe {
            list.SetPipelineState(pso);
            list.SetComputeRootSignature(&self.root);
            for (k, out) in outs.iter().enumerate() {
                list.SetComputeRootUnorderedAccessView(k as u32, out.GetGPUVirtualAddress());
            }
            list.Dispatch(groups, 1, 1);
        }
    }
}

/// Runs the compute form for `groups` groups and returns `u0` and `u1`.
fn run_compute(
    host: &Host,
    cs: &HostShader,
    layout: &MeshLayout,
    groups: u32,
) -> Result<[Vec<u32>; 2], String> {
    let gpu = Gpu::new(host)?;
    let pso = host.compute_pipeline_with(&gpu.root, &cs.dxil)?;
    let outs = gpu.mesh_buffers(layout, groups)?;
    let sizes = layout.bytes(groups);
    let readbacks: Vec<ID3D12Resource> = sizes
        .iter()
        .map(|&s| {
            gpu.buffer(
                D3D12_HEAP_TYPE_READBACK,
                s,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )
        })
        .collect::<Result<_, _>>()?;
    let (_alloc, list) = gpu.list()?;
    let index_seed = gpu.buffer(
        D3D12_HEAP_TYPE_UPLOAD,
        sizes[1],
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    )?;
    // SAFETY: the upload and index buffers have the same size and are live through submission.
    unsafe {
        let mut p = std::ptr::null_mut();
        index_seed.Map(0, None, Some(&mut p)).map_err(err)?;
        std::slice::from_raw_parts_mut(p.cast::<u32>(), sizes[1] as usize / 4).fill(1);
        index_seed.Unmap(0, None);
        list.ResourceBarrier(&[transition(
            &outs[1],
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )]);
        list.CopyResource(&outs[1], &index_seed);
        list.ResourceBarrier(&[transition(
            &outs[1],
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
    }
    gpu.record_compute(&list, &pso, &outs, groups);
    // SAFETY: plain D3D12 recording of live resources.
    unsafe {
        for (out, readback) in outs.iter().zip(&readbacks) {
            list.ResourceBarrier(&[transition(
                out,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyResource(readback, out);
        }
    }
    gpu.run(&list)?;
    Ok([
        Gpu::read(&readbacks[0], sizes[0] as usize / 4)?,
        Gpu::read(&readbacks[1], sizes[1] as usize / 4)?,
    ])
}

#[test]
fn a_mesh_program_run_as_compute_writes_its_vertices_and_triangles() {
    let (cs, layout) = lowered(true);
    let groups = 2;
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let [vertices, indices] =
            run_compute(&host, &cs, &layout, groups).unwrap_or_else(|e| panic!("{label}: {e}"));
        let f = f32::to_bits;
        let mut want_vertices = Vec::new();
        let mut want_indices = Vec::new();
        for g in 0..groups {
            for lane in 0..3u32 {
                let n = lane as f32;
                let y = 2.0 * (lane & 1) as f32 - 1.0;
                // SV_Position, then TEXCOORD0.
                want_vertices.extend([f(n - 1.0), f(y), f(0.0), f(1.0)]);
                want_vertices.extend([f(n), f(1.0), f(2.0), f(0.0)]);
                want_indices.extend(match lane {
                    0 => [3 * g, 3 * g + 1, 3 * g + 2],
                    _ => [0, 0, 0],
                });
            }
        }
        assert_eq!(vertices, want_vertices, "{label}: vertices");
        assert_eq!(indices, want_indices, "{label}: indices");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn mesh_compute_initializes_more_primitive_slots_than_lanes() {
    let (cs, layout) = lowered_with_max_primitives(true, false, 4);
    let host = Host::new(false)
        .or_else(|_| Host::new(true))
        .expect("an SM 6.6 hardware or WARP device");
    let [_, indices] = run_compute(&host, &cs, &layout, 2).unwrap();
    assert_eq!(indices.len(), 24);
    for (group, records) in indices.as_chunks::<12>().0.iter().enumerate() {
        let base = 3 * group as u32;
        assert_eq!(&records[..3], &[base, base + 1, base + 2]);
        assert_eq!(&records[3..], &[0; 9], "unused indices, group {group}");
    }
}

/// How [`render`] draws the program.
enum Draw<'a> {
    /// As a mesh shader.
    Mesh(&'a HostShader),
    /// Run as compute, then drawn by the fallback vertex shader (unindexed
    /// when `expanded`).
    Fallback {
        cs: &'a HostShader,
        vs: &'a HostShader,
        layout: &'a MeshLayout,
        expanded: bool,
    },
}

fn has_mesh_shaders(host: &Host) -> bool {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    // SAFETY: a feature query with a matching out structure.
    let ok = unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS7,
            (&raw mut options).cast(),
            size_of_val(&options) as u32,
        )
    };
    ok.is_ok() && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
}

/// Draws `groups` groups of the program with `ps` into a [`SIZE`]² RGBA32F
/// target cleared to -1, and returns its texels row by row.
fn render(host: &Host, draw: &Draw, ps: &HostShader, groups: u32) -> Result<Vec<[f32; 4]>, String> {
    let gpu = Gpu::new(host)?;
    let dev = gpu.dev;
    let format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let clear = D3D12_CLEAR_VALUE {
        Format: format,
        Anonymous: D3D12_CLEAR_VALUE_0 { Color: [-1.0; 4] },
    };
    let target = gpu.resource(
        D3D12_HEAP_TYPE_DEFAULT,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIZE),
            Height: SIZE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        Some(&clear),
    )?;
    let pitch = SIZE * 16;
    let readback = gpu.buffer(
        D3D12_HEAP_TYPE_READBACK,
        u64::from(pitch * SIZE),
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let (_alloc, list) = gpu.list()?;
    let mut keep: Vec<ID3D12PipelineState> = Vec::new();
    let mut keep_buffers: Vec<ID3D12Resource> = Vec::new();
    // SAFETY: D3D12 calls with descriptions that outlive them; every object
    // stays alive (`keep`, `keep_buffers`) until `run` has waited for the GPU.
    unsafe {
        let rtvs: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .map_err(err)?;
        let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&target, None, rtv);
        let draw_pso = match *draw {
            Draw::Mesh(ms) => host.mesh_pipeline_with(&gpu.root, &ms.dxil, &ps.dxil)?,
            Draw::Fallback { cs, vs, layout, .. } => {
                let cs_pso = host.compute_pipeline_with(&gpu.root, &cs.dxil)?;
                let outs = gpu.mesh_buffers(layout, groups)?;
                gpu.record_compute(&list, &cs_pso, &outs, groups);
                let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                    | D3D12_RESOURCE_STATE_INDEX_BUFFER;
                let barriers: Vec<_> = outs
                    .iter()
                    .map(|o| transition(o, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, read))
                    .collect();
                list.ResourceBarrier(&barriers);
                list.SetGraphicsRootSignature(&gpu.root);
                for (k, o) in outs.iter().enumerate() {
                    list.SetGraphicsRootShaderResourceView(
                        VERTICES_SRV + k as u32,
                        o.GetGPUVirtualAddress(),
                    );
                }
                list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                    BufferLocation: outs[1].GetGPUVirtualAddress(),
                    SizeInBytes: layout.bytes(groups)[1] as u32,
                    Format: DXGI_FORMAT_R32_UINT,
                }));
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                keep.push(cs_pso);
                keep_buffers.extend(outs);
                host.pipeline_with(&gpu.root, &vs.dxil, &ps.dxil)?
            }
        };
        list.SetPipelineState(&draw_pso);
        list.SetGraphicsRootSignature(&gpu.root);
        list.ClearRenderTargetView(rtv, &[-1.0; 4], None);
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.RSSetViewports(&[D3D12_VIEWPORT {
            Width: SIZE as f32,
            Height: SIZE as f32,
            MaxDepth: 1.0,
            ..Default::default()
        }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
            right: SIZE as i32,
            bottom: SIZE as i32,
            ..Default::default()
        }]);
        match *draw {
            Draw::Mesh(_) => list.DispatchMesh(groups, 1, 1),
            Draw::Fallback {
                layout, expanded, ..
            } => {
                let n = 3 * layout.max_primitives * groups;
                if expanded {
                    list.DrawInstanced(n, 1, 0, 0);
                } else {
                    list.DrawIndexedInstanced(n, 1, 0, 0, 0);
                }
            }
        }
        list.ResourceBarrier(&[transition(
            &target,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&readback),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: SIZE,
                        Height: SIZE,
                        Depth: 1,
                        RowPitch: pitch,
                    },
                },
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&target),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        gpu.run(&list)?;
        drop(draw_pso);
    }
    drop((keep, keep_buffers));
    let words = Gpu::read(&readback, (pitch * SIZE / 4) as usize)?;
    Ok(words
        .as_chunks::<4>()
        .0
        .iter()
        .map(|t| t.map(f32::from_bits))
        .collect())
}

/// Whether two images cover the same texels and agree within `tolerance`.
fn same_image(a: &[[f32; 4]], b: &[[f32; 4]], tolerance: f32) -> Result<(), String> {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let (covered_a, covered_b) = (x[1] != -1.0, y[1] != -1.0);
        if covered_a != covered_b || x.iter().zip(y).any(|(p, q)| (p - q).abs() > tolerance) {
            return Err(format!(
                "texel ({}, {}): {x:?} and {y:?}",
                i as u32 % SIZE,
                i as u32 / SIZE
            ));
        }
    }
    Ok(())
}

#[test]
fn a_mesh_program_run_as_compute_draws_what_the_mesh_shader_draws() {
    compare_with_the_mesh_shader(false);
}

#[test]
fn per_primitive_outputs_run_as_compute_draw_what_the_mesh_shader_draws() {
    compare_with_the_mesh_shader(true);
}

/// Draws the program (with `prims`, its per-primitive variant) with the
/// fallback on every device and compares with the first mesh-capable
/// device's mesh shader.
fn compare_with_the_mesh_shader(prims: bool) {
    let (ms, _) = lowered_with(false, prims);
    let (cs, layout) = lowered_with(true, prims);
    let vs = fallback_vs_with(&layout, prims);
    let ps = pixel_shader(prims);
    let fallback = Draw::Fallback {
        cs: &cs,
        vs: &vs,
        layout: &layout,
        expanded: prims,
    };
    let groups = 2;
    let mut reference = None;
    let mut images = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        if has_mesh_shaders(&host) && reference.is_none() {
            let native = render(&host, &Draw::Mesh(&ms), &ps, groups)
                .unwrap_or_else(|e| panic!("{label} mesh: {e}"));
            reference = Some((label, native));
        } else {
            eprintln!("{label}: no mesh shaders; the fallback only");
        }
        let image = render(&host, &fallback, &ps, groups)
            .unwrap_or_else(|e| panic!("{label} fallback: {e}"));
        // The triangle covers about a quarter of the target, with
        // TEXCOORD0.y = 1 wherever it is drawn.
        let covered: Vec<_> = image.iter().filter(|t| t[1] != -1.0).collect();
        assert!(
            covered.len() > (SIZE * SIZE / 5) as usize,
            "{label}: {} texels",
            covered.len()
        );
        // TEXCOORD0.z = 2, or the primitive's COLOR0.xy = (1, 2).
        assert!(
            covered.iter().all(|t| if prims {
                t[2] == 1.0 && t[3] == 2.0
            } else {
                t[2] == 2.0
            }),
            "{label}: {:?}",
            covered[0]
        );
        images.push((label, image));
    }
    assert!(
        !images.is_empty(),
        "neither hardware nor WARP SM 6.6 device available"
    );
    let Some((ref_label, native)) = reference else {
        panic!("no device with mesh shaders to compare with");
    };
    for (label, image) in &images {
        // The same device rasterizes both the same way; another device may
        // interpolate a little differently.
        let tolerance = if *label == ref_label { 1e-6 } else { 1e-3 };
        same_image(&native, image, tolerance)
            .unwrap_or_else(|e| panic!("{ref_label} mesh vs {label} fallback: {e}"));
    }
}

/// 128-thread Wave64 mesh shader (`v_mbcnt(-1)` + `v_readfirstlane` +
/// fullscreen triangle from threads 32..34).
fn wave64_host_lane_parity_program() -> NativeProgram {
    const TWO: u16 = 0x82;
    const VCC: u16 = 106;
    let mut w = Vec::new();
    w.extend(vop3(0x365, 0, MINUS_ONE, ZERO, 0)); // v_mbcnt_lo_u32_b32 v0, -1, 0
    w.extend(vop3(0x366, 0, MINUS_ONE, vgpr(0), 0)); // v_mbcnt_hi_u32_b32 v0, -1, v0
    w.push(sop2(0x20, 2, 3, 0x80 + 24)); // s_lshr_b32 s2, s3, 24
    w.push(vop1(0x002, 4, vgpr(0))); // v_readfirstlane_b32 s4, v0
    w.push(vop2(0x1B, 5, 0x80 + 31, 0)); // v_and_b32 v5, 31, v0
    w.push(vop2(0x26, 10, vgpr(0), 5)); // v_sub_nc_u32 v10, v0, v5
    w.push(vop1(0x006, 1, 4)); // v_cvt_f32_u32 v1, s4
    w.push(vop1(0x006, 2, vgpr(10))); // v_cvt_f32_u32 v2, v10
    w.push(vop1(0x006, 3, vgpr(5))); // v_cvt_f32_u32 v3, v5
    w.push(vop1(0x001, 4, ONE_F)); // v_mov_b32 v4, 1.0
    w.extend([vop1(0x001, 6, LITERAL), (-1.0f32).to_bits()]);
    w.extend([vop1(0x001, 7, LITERAL), 3.0f32.to_bits()]);
    w.push(vopc(0xC2, ONE, 5)); // v_cmp_eq_u32 1, v5
    w.extend(vop3(0x101, 6, vgpr(6), vgpr(7), VCC));
    w.extend([vop1(0x001, 11, LITERAL), (-1.0f32).to_bits()]);
    w.push(vopc(0xC2, TWO, 5)); // v_cmp_eq_u32 2, v5
    w.extend(vop3(0x101, 7, vgpr(11), vgpr(7), VCC));
    w.extend([vop1(0x001, 8, LITERAL), 0.5f32.to_bits()]);
    w.push(vop1(0x001, 9, ONE_F));
    w.extend(exp(32, [Some(1), Some(2), Some(3), Some(4)], false)); // TEXCOORD0
    w.extend(exp(12, [Some(6), Some(7), Some(8), Some(9)], false)); // SV_Position
    w.push(vop2(0x1C, 12, 2, 0)); // v_or_b32 v12, s2, v0
    w.extend([vop1(0x001, 14, LITERAL), 0x8000_0000]);
    w.extend([vop1(0x001, 15, LITERAL), 32 | (33 << 10) | (34 << 20)]);
    w.push(vopc(0xC2, ZERO, 12)); // v_cmp_eq_u32 0, v12
    w.extend(vop3(0x101, 13, vgpr(14), vgpr(15), VCC));
    w.extend(exp(20, [Some(13), None, None, None], true)); // prim
    w.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Ms,
        wave64: true,
        vgprs: 16,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ms(MeshIo {
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 1,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            clip_cntl: 0,
            threads: 128,
            max_vertices: 128,
            max_primitives: 128,
            topology: 2,
            lds_bytes: 0,
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn lower_wave64_mesh(as_compute: bool, mode: Wave64Mode) -> (HostShader, MeshLayout) {
    let native = wave64_host_lane_parity_program();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let outs = outputs();
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: outs.clone(),
        prims: Vec::new(),
        mesh_stride: 128,
        options: TranslateOptions { wave64: mode },
        mesh_as_compute: as_compute,
    })
    .expect("lower");
    let s = shader(HwStage::Ms, l.dxil);
    validate(&s);
    let layout = MeshLayout {
        threads: 128,
        max_vertices: 128,
        max_primitives: 128,
        vertex_dwords: 4 * outs.len() as u32,
        primitive_dwords: 0,
    };
    (s, layout)
}

#[test]
fn wave64_128_thread_mesh_compute_fallback_matches_mesh_shader() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ps = pixel_shader(false);
    let hosts = Host::all_test_hosts();
    assert!(!hosts.is_empty(), "expected at least one SM 6.6 host");

    for (desc, warp, host) in &hosts {
        let lanes = host.wave_lanes();
        let mesh_ok = has_mesh_shaders(host);
        let rec = host.recommended_wave64_mode();
        eprintln!(
            "[mesh_compute_tests] adapter={desc:?} warp={warp} wave_lanes={lanes:?} mesh={mesh_ok} rec={rec:?}"
        );
        let mut modes = vec![rec];
        if !modes.contains(&Wave64Mode::Emulate) {
            modes.push(Wave64Mode::Emulate);
        }
        if host.supports_wave32() && !modes.contains(&Wave64Mode::Wave32) {
            modes.push(Wave64Mode::Wave32);
        }

        for mode in modes {
            let (ms, _) = lower_wave64_mesh(false, mode);
            let (cs, layout) = lower_wave64_mesh(true, mode);
            let vs = fallback_vs(&layout);

            // 1. Verify u0 (vertices) & u1 (indices) UAV readback across all 128 threads.
            let [vertices, indices] = run_compute(host, &cs, &layout, 1)
                .unwrap_or_else(|e| panic!("{desc} ({mode:?}, warp={warp}) run_compute: {e}"));
            assert_eq!(vertices.len(), 128 * 8);
            assert_eq!(indices.len(), 128 * 3);
            let host_wave = if lanes.0 && lanes.1 == lanes.2 {
                lanes.1
            } else {
                32
            };
            for t in 0..128usize {
                let r = f32::from_bits(vertices[8 * t + 4]);
                let g = f32::from_bits(vertices[8 * t + 5]);
                let b = f32::from_bits(vertices[8 * t + 6]);
                let a = f32::from_bits(vertices[8 * t + 7]);
                assert_eq!(
                    r, 0.0,
                    "{desc} ({mode:?}, warp={warp}) thread {t}: expected R=0.0, got {r}"
                );
                assert_eq!(a, 1.0, "{desc} ({mode:?}, warp={warp}) thread {t}: A");
                if host_wave <= 32 {
                    assert_eq!(g, 0.0, "{desc} ({mode:?}, warp={warp}) thread {t}: G");
                    assert_eq!(
                        b,
                        (t as u32 % host_wave) as f32,
                        "{desc} ({mode:?}, warp={warp}) thread {t}: B"
                    );
                }
            }
            for p in 0..128usize {
                let tri = &indices[3 * p..3 * p + 3];
                let expect_tri = p < 64 && (p as u32).is_multiple_of(host_wave);
                if expect_tri {
                    assert_eq!(
                        tri,
                        &[32, 33, 34],
                        "{desc} ({mode:?}, warp={warp}) prim {p}"
                    );
                } else if host_wave <= 32 {
                    assert_eq!(tri, &[0, 0, 0], "{desc} ({mode:?}, warp={warp}) prim {p}");
                }
            }

            // 2. Render via Fallback and compare pixel-for-pixel with DispatchMesh.
            let fallback_pixels = render(
                host,
                &Draw::Fallback {
                    cs: &cs,
                    vs: &vs,
                    layout: &layout,
                    expanded: false,
                },
                &ps,
                1,
            )
            .unwrap_or_else(|e| panic!("{desc} ({mode:?}, warp={warp}) fallback render: {e}"));
            for px in &fallback_pixels {
                assert_eq!(px[0], 0.0, "{desc} ({mode:?}, warp={warp}) fallback R");
                assert_ne!(px[1], -1.0, "{desc} ({mode:?}, warp={warp}) uncovered");
                assert_eq!(px[3], 1.0, "{desc} ({mode:?}, warp={warp}) fallback A");
            }
            if mesh_ok {
                let mesh_pixels = render(host, &Draw::Mesh(&ms), &ps, 1)
                    .unwrap_or_else(|e| panic!("{desc} ({mode:?}, warp={warp}) mesh render: {e}"));
                same_image(&mesh_pixels, &fallback_pixels, 1e-6)
                    .unwrap_or_else(|e| panic!("{desc} ({mode:?}, warp={warp}) parity: {e}"));
            }
        }
    }
}

/// A full wave32 group for the benchmark: lane n exports vertex n near the
/// origin (triangles of under a pixel, so the raster costs little) with
/// `TEXCOORD0` as in [`program`], and the triangle (n, n + 1, n + 2), all
/// modulo 32.
fn bench_program() -> NativeProgram {
    const TINY: u32 = 0x3A00_0000; // 1/2048
    let mut w = Vec::new();
    w.extend(vop3(0x365, 0, MINUS_ONE, ZERO, 0)); // v_mbcnt_lo_u32_b32 v0, -1, 0: the lane
    w.push(vop1(0x006, 1, vgpr(0))); // v_cvt_f32_u32 v1, v0
    w.push(vop1(0x001, 2, TWO_F)); // v_mov_b32 v2, 2.0
    w.push(vop1(0x001, 3, ZERO)); // v_mov_b32 v3, 0
    w.push(vop1(0x001, 4, ONE_F)); // v_mov_b32 v4, 1.0
    w.extend([vop2(0x08, 7, LITERAL, 1), TINY]); // v_mul_f32 v7, tiny, v1: x
    w.push(vop2(0x1b, 8, ONE, 0)); // v_and_b32 v8, 1, v0
    w.push(vop1(0x006, 8, vgpr(8))); // v_cvt_f32_u32 v8, v8
    w.extend([vop2(0x08, 8, LITERAL, 8), TINY]); // v_mul_f32 v8, tiny, v8: y
    w.push(vop2(0x25, 9, ONE, 0)); // v_add_nc_u32 v9, 1, v0
    w.push(vop2(0x1b, 9, 0x9F, 9)); // v_and_b32 v9, 31, v9
    w.push(vop2(0x25, 10, 0x82, 0)); // v_add_nc_u32 v10, 2, v0
    w.push(vop2(0x1b, 10, 0x9F, 10)); // v_and_b32 v10, 31, v10
    w.push(vop2(0x1a, 9, 0x8A, 9)); // v_lshlrev_b32 v9, 10, v9
    w.push(vop2(0x1a, 10, 0x94, 10)); // v_lshlrev_b32 v10, 20, v10
    w.push(vop2(0x1c, 6, vgpr(9), 0)); // v_or_b32 v6, v9, v0
    w.push(vop2(0x1c, 6, vgpr(10), 6)); // v_or_b32 v6, v10, v6
    w.extend(exp(20, [Some(6), None, None, None], true)); // exp prim
    w.extend(exp(32, [Some(1), Some(4), Some(2), Some(3)], false)); // exp param0
    w.extend(exp(12, [Some(7), Some(8), Some(3), Some(4)], true)); // exp pos0
    w.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 12,
        layout: vec![],
        used: vec![],
        io: StageIo::Ms(MeshIo {
            outputs: vec![VsOutput {
                usage: Usage::Generic,
                row: 1,
                param: 0,
                fp16: false,
                pack2: false,
            }],
            clip_cntl: 0,
            threads: 32,
            max_vertices: 32,
            max_primitives: 32,
            topology: 2,
            lds_bytes: 0,
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// [`bench_program`] lowered as compute, with its layout.
fn bench_lowered() -> (HostShader, MeshLayout) {
    let native = bench_program();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: outputs(),
        prims: Vec::new(),
        mesh_stride: 32,
        options: super::TranslateOptions::default(),
        mesh_as_compute: true,
    })
    .expect("lower");
    let s = shader(HwStage::Ms, l.dxil);
    validate(&s);
    let layout = MeshLayout {
        threads: 32,
        max_vertices: 32,
        max_primitives: 32,
        vertex_dwords: 4 * outputs().len() as u32,
        primitive_dwords: 0,
    };
    (s, layout)
}

/// How a benchmark frame records its mesh draws.
#[derive(Clone, Copy, Debug)]
enum Frame {
    /// As `victory-gpu` does today: per draw, the compute pass, a
    /// transition to read, the draw, and a transition back.
    PerDraw,
    /// Every draw's compute pass first, one transition, then every draw.
    Batched,
    /// The draws alone, from what an earlier frame wrote: what drawing the
    /// same triangles with a vertex shader costs.
    DrawsOnly,
}

/// GPU time of one frame of `draws` mesh draws of `groups` groups each,
/// recorded as `frame`, in milliseconds (the median of a few runs).
fn bench_frame(
    host: &Host,
    (cs, vs, ps): (&HostShader, &HostShader, &HostShader),
    layout: &MeshLayout,
    (draws, groups): (u32, u32),
    frame: Frame,
) -> Result<f64, String> {
    let gpu = Gpu::new(host)?;
    let dev = gpu.dev;
    let cs_pso = host.compute_pipeline_with(&gpu.root, &cs.dxil)?;
    let gfx_pso = host.pipeline_with(&gpu.root, &vs.dxil, &ps.dxil)?;
    let [vertex_bytes, index_bytes, _] = layout.bytes(groups);
    let aligned = |n: u64| n.next_multiple_of(256);
    let at_indices = aligned(vertex_bytes);
    let region = at_indices + aligned(index_bytes) + 256;
    let scratch = gpu.buffer(
        D3D12_HEAP_TYPE_DEFAULT,
        region * u64::from(draws),
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    )?;
    let target = gpu.resource(
        D3D12_HEAP_TYPE_DEFAULT,
        &D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIZE),
            Height: SIZE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        },
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        None,
    )?;
    let stamps = gpu.buffer(
        D3D12_HEAP_TYPE_READBACK,
        16,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let indices = gpu.buffer(
        D3D12_HEAP_TYPE_READBACK,
        index_bytes,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )?;
    let (uav, read) = (
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_INDEX_BUFFER,
    );
    let mut times = Vec::new();
    // SAFETY: D3D12 calls with descriptions that outlive them; every object
    // lives until `run` has waited for the GPU.
    unsafe {
        let mut queries: Option<ID3D12QueryHeap> = None;
        dev.CreateQueryHeap(
            &D3D12_QUERY_HEAP_DESC {
                Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                Count: 2,
                NodeMask: 0,
            },
            &mut queries,
        )
        .map_err(err)?;
        let queries = queries.ok_or("no query heap")?;
        let rtvs: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .map_err(err)?;
        let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&target, None, rtv);
        let frequency = gpu.queue.GetTimestampFrequency().map_err(err)? as f64;
        let base = scratch.GetGPUVirtualAddress();
        let at = |d: u32| base + u64::from(d) * region;
        let compute = |list: &ID3D12GraphicsCommandList6, d: u32| {
            list.SetComputeRootUnorderedAccessView(0, at(d));
            list.SetComputeRootUnorderedAccessView(1, at(d) + at_indices);
            list.SetComputeRootUnorderedAccessView(2, at(d) + region - 256);
            list.Dispatch(groups, 1, 1);
        };
        let draw = |list: &ID3D12GraphicsCommandList6, d: u32| {
            list.SetGraphicsRootShaderResourceView(VERTICES_SRV, at(d));
            list.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: at(d) + at_indices,
                SizeInBytes: index_bytes as u32,
                Format: DXGI_FORMAT_R32_UINT,
            }));
            list.DrawIndexedInstanced(3 * layout.max_primitives * groups, 1, 0, 0, 0);
        };
        let graphics = |list: &ID3D12GraphicsCommandList6| {
            list.SetPipelineState(&gfx_pso);
            list.SetGraphicsRootSignature(&gpu.root);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIZE as f32,
                Height: SIZE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                right: SIZE as i32,
                bottom: SIZE as i32,
                ..Default::default()
            }]);
            list.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
        };
        let to = |b, a| [transition(&scratch, b, a)];
        // A first, untimed frame writes every region (the draws-only frames
        // read them) and checks the last draw's indices.
        for run in 0..=7 {
            let (_alloc, list) = gpu.list()?;
            let first = run == 0;
            list.ClearRenderTargetView(rtv, &[0.0; 4], None);
            list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            match frame {
                _ if first => {
                    list.SetPipelineState(&cs_pso);
                    list.SetComputeRootSignature(&gpu.root);
                    for d in 0..draws {
                        compute(&list, d);
                    }
                }
                Frame::PerDraw => {
                    for d in 0..draws {
                        list.SetPipelineState(&cs_pso);
                        list.SetComputeRootSignature(&gpu.root);
                        compute(&list, d);
                        list.ResourceBarrier(&to(uav, read));
                        graphics(&list);
                        draw(&list, d);
                        list.ResourceBarrier(&to(read, uav));
                    }
                }
                Frame::Batched => {
                    list.SetPipelineState(&cs_pso);
                    list.SetComputeRootSignature(&gpu.root);
                    for d in 0..draws {
                        compute(&list, d);
                    }
                    list.ResourceBarrier(&to(uav, read));
                    graphics(&list);
                    for d in 0..draws {
                        draw(&list, d);
                    }
                    list.ResourceBarrier(&to(read, uav));
                }
                Frame::DrawsOnly => {
                    list.ResourceBarrier(&to(uav, read));
                    graphics(&list);
                    for d in 0..draws {
                        draw(&list, d);
                    }
                    list.ResourceBarrier(&to(read, uav));
                }
            }
            list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &stamps, 0);
            let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
            if first {
                list.ResourceBarrier(&to(uav, copy));
                let last = u64::from(draws - 1) * region + at_indices;
                list.CopyBufferRegion(&indices, 0, &scratch, last, index_bytes);
                list.ResourceBarrier(&to(copy, uav));
            }
            gpu.run(&list)?;
            if first {
                let got = Gpu::read(&indices, index_bytes as usize / 4)?;
                let tail = &got[got.len() - 3 * 32..];
                let group = 32 * (groups - 1);
                for lane in [0, 1, 31u32] {
                    let k = 3 * lane as usize;
                    let want = [lane, (lane + 1) % 32, (lane + 2) % 32].map(|v| group + v);
                    assert_eq!(tail[k..k + 3], want, "lane {lane}'s triangle");
                }
                continue;
            }
            let t = Gpu::read(&stamps, 4)?;
            let stamp = |k: usize| u64::from(t[2 * k]) | u64::from(t[2 * k + 1]) << 32;
            times.push((stamp(1) - stamp(0)) as f64 / frequency * 1e3);
        }
    }
    times.sort_by(f64::total_cmp);
    Ok(times[times.len() / 2])
}

/// What drawing mesh programs through the compute fallback costs on the
/// GPU, against drawing the same triangles with a vertex shader alone. Run
/// by hand on a GPU worth measuring:
/// `cargo test -p victory-shader --lib fallback_frame_cost -- --ignored --nocapture`
#[test]
#[ignore = "a benchmark; prints GPU times"]
fn fallback_frame_cost() {
    let (cs, layout) = bench_lowered();
    let vs = fallback_vs(&layout);
    let ps = pixel_shader(false);
    let host = Host::new(false).expect("no SM 6.6 hardware device");
    // (draws, wave32 groups per draw): a group of the title's is 128
    // threads, four of these.
    for shape in [(1000, 32), (2000, 16), (7000, 4), (7000, 16)] {
        let triangles = u64::from(shape.0 * shape.1 * 32);
        let mut line = format!(
            "{:>5} draws x {:>2} groups ({triangles:>8} triangles):",
            shape.0, shape.1
        );
        for frame in [Frame::PerDraw, Frame::Batched, Frame::DrawsOnly] {
            let ms = bench_frame(&host, (&cs, &vs, &ps), &layout, shape, frame)
                .unwrap_or_else(|e| panic!("{frame:?}: {e}"));
            line += &format!("  {frame:?} {ms:7.2} ms");
        }
        println!("{line}");
    }
}
