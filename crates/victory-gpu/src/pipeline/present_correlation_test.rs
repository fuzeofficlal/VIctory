//! Synthetic offscreen D3D12 multi-pass pixel-readback and last-writer
//! correlation tests (no Win32 window or DXGI swap chain).

use super::*;
use crate::capture::{InputKind, PresentPlane, WriteKind};
use crate::object::{S_OK, slot};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
use windows::core::{GUID, Interface, PCSTR, s};

fn compile_hlsl(source: &str, entry: PCSTR, target: PCSTR) -> Vec<u8> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: static source and target/entry pointers live through the call.
    unsafe {
        let hr = D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        );
        if let Some(e) = &errors {
            let text =
                std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize());
            assert!(hr.is_ok(), "{}", String::from_utf8_lossy(text));
        }
        hr.unwrap();
        let code = code.unwrap();
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
            .to_vec()
    }
}

fn make_graphics_shader(
    stage: &'static str,
    hw_stage: victory_shader::HwStage,
    dxil: Vec<u8>,
) -> Arc<Shader> {
    let s = shader(stage, &dxil);
    s.translated.get_or_init(|| {
        Some(Translated {
            host: victory_shader::emit::HostShader {
                stage: hw_stage,
                dxil,
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
            },
            used: Vec::new(),
        })
    });
    s
}

fn make_graphics_pso(ps_rgba: [f32; 4], samples: u32) -> This {
    let vs_src = "float4 main(uint id : SV_VertexID) : SV_Position {
        return float4(id == 1 ? 3 : -1, id == 2 ? -3 : 1, 0.25, 1);
    }";
    let ps_src = format!(
        "float4 main() : SV_Target {{ return float4({}, {}, {}, {}); }}",
        ps_rgba[0], ps_rgba[1], ps_rgba[2], ps_rgba[3]
    );
    let vs = make_graphics_shader(
        "vertex",
        victory_shader::HwStage::Vs,
        compile_hlsl(vs_src, s!("main"), s!("vs_5_1")),
    );
    let ps = make_graphics_shader(
        "pixel",
        victory_shader::HwStage::Ps,
        compile_hlsl(&ps_src, s!("main"), s!("ps_5_1")),
    );
    build_graphics_stream_pso(vs, ps, samples)
}

fn build_graphics_stream_pso(vs: Arc<Shader>, ps: Arc<Shader>, samples: u32) -> This {
    #[repr(C, align(8))]
    struct Depth(u32, D3D12_DEPTH_STENCIL_DESC);
    #[repr(C)]
    struct Stream {
        vs_kind: u64,
        vs: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        depth: Depth,
        topology: [u32; 2],
        targets: [u32; 10],
        samples: [u32; 4],
    }
    const _: () = assert!(size_of::<Depth>() == 56);
    let mut targets = [0u32; 10];
    targets[0] = 15; // D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS
    targets[1] = DXGI_FORMAT_R8G8B8A8_UNORM.0 as u32;
    targets[9] = 1; // NumRenderTargets
    let stream = Stream {
        vs_kind: 1,
        vs: Bytecode {
            code: vs.container.as_ptr(),
            size: vs.container.len(),
        },
        ps_kind: 2,
        ps: Bytecode {
            code: ps.container.as_ptr(),
            size: ps.container.len(),
        },
        depth: Depth(
            11,
            D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: false.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                ..Default::default()
            },
        ),
        topology: [14, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32],
        targets,
        samples: [17, samples, 0, 0],
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let mut pso = std::ptr::null();
    assert_eq!(
        create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut pso),
        S_OK
    );
    assert!(
        host_pipeline(pso).is_some(),
        "host refused graphics pipeline (samples={samples})"
    );
    pso
}

fn dxbc_with_osg1(name: &str, system_value: u32) -> Vec<u8> {
    let mut osg1 = Vec::new();
    osg1.extend(1u32.to_le_bytes()); // element count
    osg1.extend(8u32.to_le_bytes()); // element offset
    for v in [0u32, 40, 0, system_value, 3 /* f32 */, 0] {
        osg1.extend(v.to_le_bytes());
    }
    osg1.extend([0xfu8, 0x0, 0, 0]);
    osg1.extend(0u32.to_le_bytes());
    osg1.extend_from_slice(name.as_bytes());
    osg1.push(0);

    let part_offset = 0x24u32;
    let total_size = part_offset + 8 + osg1.len() as u32;
    let mut out = vec![0u8; 0x20];
    out[0..4].copy_from_slice(b"DXBC");
    out[0x14..0x18].copy_from_slice(&1u32.to_le_bytes());
    out[0x18..0x1c].copy_from_slice(&total_size.to_le_bytes());
    out[0x1c..0x20].copy_from_slice(&1u32.to_le_bytes());
    out.extend(part_offset.to_le_bytes());
    out.extend_from_slice(b"OSG1");
    out.extend((osg1.len() as u32).to_le_bytes());
    out.extend_from_slice(&osg1);
    out
}

fn make_graphics_srv_read_pso() -> This {
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, emit,
        io, ir, native::StageIo, resource, structure,
    };

    const S_ENDPGM: u32 = 0xBF81_0000;
    const S_WAITCNT_0: u32 = 0xBF8C_0000;
    const ZERO: u32 = 0x80;
    const LITERAL: u32 = 0xFF;
    let s_mov = |d: u8, src: u32| 0xBE80_0300 | u32::from(d) << 16 | src;
    let s_load_x4 = |d: u8, base: u8| -> [u32; 2] {
        [
            0xF408_0000 | u32::from(d) << 6 | u32::from(base / 2),
            0xFA00_0000,
        ]
    };
    let s_buffer_load_x4 = |d: u8, base: u8| -> [u32; 2] {
        [
            0xF428_0000 | u32::from(d) << 6 | u32::from(base / 2),
            0xFA00_0000,
        ]
    };
    let v_mov_s = |d: u8, s: u8| 0x7E00_0200 | u32::from(d) << 17 | u32::from(s);
    let v_mov_lit = |d: u8, v: f32| [0x7E00_0200 | u32::from(d) << 17 | LITERAL, v.to_bits()];

    // Fullscreen triangle VS from SV_VertexID (v0):
    // x = (v0 & 1) * 4.0 - 1.0, y = 1.0 - (v0 >> 1) * 4.0, z = 0.25, w = 1.0.
    let mut vs_words = vec![
        0x1B << 25 | 2 << 17 | 0x81,                  // v_and_b32 v2, 1, v0
        0x16 << 25 | 3 << 17 | 0x81,                  // v_lshrrev_b32 v3, 1, v0
        0x7E00_0000 | 2 << 17 | 6 << 9 | (0x100 + 2), // v_cvt_f32_u32 v2, v2
        0x7E00_0000 | 3 << 17 | 6 << 9 | (0x100 + 3), // v_cvt_f32_u32 v3, v3
        0x08 << 25 | 2 << 17 | 2 << 9 | LITERAL,
        4.0f32.to_bits(),
        0x03 << 25 | 2 << 17 | 2 << 9 | LITERAL,
        (-1.0f32).to_bits(),
        0x08 << 25 | 3 << 17 | 3 << 9 | LITERAL,
        4.0f32.to_bits(),
        0x04 << 25 | 3 << 17 | 3 << 9 | LITERAL,
        1.0f32.to_bits(), // v_sub_f32 v3, 1.0, v3
    ];
    vs_words.extend(v_mov_lit(4, 0.25));
    vs_words.extend(v_mov_lit(5, 1.0));
    vs_words.push(0xF800_08CF);
    vs_words.push(0x0504_0302); // exp pos0, v2, v3, v4, v5 done
    vs_words.push(S_ENDPGM);

    let vs_native = NativeProgram {
        stage: HwStage::Vs,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Vs {
            inputs: Vec::new(),
            outputs: Vec::new(),
        },
        code: vs_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let vs_ir = ir::build(&vs_native.code, vs_native.wave64, vs_native.vgprs).unwrap();
    let vs_st = structure::structure(&vs_ir).unwrap();
    let vs_res = resource::track(&vs_ir, &vs_native.layout, vs_native.stage, Some(&vs_st)).unwrap();
    let vs_io = io::raise(&vs_ir, &vs_native).unwrap();
    let vs_dxbc = dxbc_with_osg1("SV_Position", 1);
    let vs_container = Container::parse(&vs_dxbc).unwrap();
    let vs_host = emit::emit(&vs_container, &vs_native, &vs_ir, &vs_st, &vs_io, &vs_res).unwrap();

    // PS: s0 is SRV table pointer (1 SGPR). Loads SRV descriptor to s[12:15],
    // reads 4 floats to s[16:19], exports as mrt0 color.
    let mut words = Vec::new();
    words.push(s_mov(10, 0));
    words.push(s_mov(11, ZERO));
    words.extend(s_load_x4(12, 10));
    words.push(S_WAITCNT_0);
    words.extend(s_buffer_load_x4(16, 12));
    words.push(S_WAITCNT_0);
    for i in 0..4u8 {
        words.push(v_mov_s(2 + i, 16 + i));
    }
    words.push(0xF800_180F);
    words.push(0x0504_0302); // exp mrt0, v2, v3, v4, v5 done vm
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io_raised = io::raise(&p, &native).unwrap();
    let ps_dxbc = dxbc_with_osg1("SV_Target", 64);
    let container = Container::parse(&ps_dxbc).unwrap();
    let host_shader = emit::emit(&container, &native, &p, &st, &io_raised, &res).unwrap();

    let vs = shader("vertex", b"vs-srv-read-correlation-test");
    vs.translated.get_or_init(|| {
        Some(Translated {
            host: vs_host,
            used: Vec::new(),
        })
    });
    let ps = shader("pixel", b"ps-srv-read-correlation-test");
    ps.translated.get_or_init(|| {
        Some(Translated {
            host: host_shader,
            used: native.used.clone(),
        })
    });
    build_graphics_stream_pso(vs, ps, 1)
}

fn make_compute_uav_fill_pso(rgba: [f32; 4]) -> This {
    use victory_shader::{
        Container, HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, emit,
        io, ir,
        native::{ComputeIo, StageIo},
        resource, structure,
    };

    const S_ENDPGM: u32 = 0xBF81_0000;
    const S_WAITCNT_0: u32 = 0xBF8C_0000;
    const ZERO: u32 = 0x80;
    const LITERAL: u32 = 0xFF;
    const IMAGE_STORE: u32 = 0x08;

    let s_mov = |d: u8, src: u32| 0xBE80_0300 | u32::from(d) << 16 | src;
    let s_load_x8 = |d: u8, base: u8, offset: u32| -> [u32; 2] {
        [
            0xF400_0000 | 3 << 18 | u32::from(d) << 6 | u32::from(base / 2),
            0xFA00_0000 | offset,
        ]
    };
    let v_mov = |d: u8, src: u32| 0x7E00_0000 | u32::from(d) << 17 | 1 << 9 | src;
    let v_lshl =
        |d: u8, by: u32, v: u8| 0x1A << 25 | u32::from(d) << 17 | u32::from(v) << 9 | (0x80 + by);
    let v_add_vv = |d: u8, v0: u8, v1: u8| {
        0x25 << 25 | u32::from(d) << 17 | u32::from(v1) << 9 | (0x100 + u32::from(v0))
    };

    // s0: UAV table pointer (1 SGPR), s1: tgid_x, s2: tgid_y, v0: tid_x, v1: tid_y.
    // Copies s0 -> s10, loads UAV descriptor to s[12:19], computes (x, y) into v[2:3],
    // writes constant `rgba` via image_store (dmask=0xf, 2D).
    let mut words = Vec::new();
    words.push(s_mov(10, 0));
    words.push(s_mov(11, ZERO));
    words.extend(s_load_x8(12, 10, 0));
    words.push(S_WAITCNT_0);
    words.push(v_mov(2, 1));
    words.push(v_lshl(2, 3, 2));
    words.push(v_add_vv(2, 0, 2));
    words.push(v_mov(3, 2));
    words.push(v_lshl(3, 3, 3));
    words.push(v_add_vv(3, 1, 3));
    for (i, ch) in rgba.iter().enumerate() {
        words.push(v_mov(4 + i as u8, LITERAL));
        words.push(ch.to_bits());
    }
    words.push(0xF000_0000 | (IMAGE_STORE << 18) | (1 << 3) | (0xf << 8));
    words.push(2 | (4 << 8) | ((12 / 4) << 16));
    words.push(S_WAITCNT_0);
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            // 1 user SGPR (bit 1 -> 1), tgid_x (bit 7), tgid_y (bit 8), tid_x+tid_y (bit 11).
            rsrc2: (1 << 1) | (1 << 7) | (1 << 8) | (1 << 11),
            threads: [8, 8, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io_raised = io::raise(&p, &native).unwrap();
    let mut dxbc_bytes = vec![0u8; 0x20];
    dxbc_bytes[0..4].copy_from_slice(b"DXBC");
    dxbc_bytes[0x14..0x18].copy_from_slice(&1u32.to_le_bytes());
    dxbc_bytes[0x18..0x1c].copy_from_slice(&0x20u32.to_le_bytes());
    let container = Container::parse(&dxbc_bytes).unwrap();
    let host_shader = emit::emit(&container, &native, &p, &st, &io_raised, &res).unwrap();

    // Unique synthetic tag so the shared shader cache keeps this exact shader.
    let mut tag = b"uav-fill-64x64".to_vec();
    for ch in rgba {
        tag.extend_from_slice(&ch.to_le_bytes());
    }
    let cs = shader("compute", &tag);
    cs.translated.get_or_init(|| {
        Some(Translated {
            host: host_shader,
            used: native.used.clone(),
        })
    });
    let mut pso = std::ptr::null();
    assert_eq!(hand_out(vec![cs], None, &PIPELINE_IIDS[0], &mut pso), S_OK);
    assert!(
        host_pipeline(pso).is_some(),
        "host refused compute UAV fill pipeline"
    );
    pso
}

fn readback_rgba8(host: &crate::host::Host, guest_tex: This) -> Vec<u8> {
    let tex = crate::resource::host_resource(guest_tex).unwrap();
    // SAFETY: live texture.
    let d = unsafe { tex.GetDesc() };
    let list = host.new_list().unwrap();
    let (_, before) = crate::resource::host_transition(guest_tex, 0x800).unwrap();
    // SAFETY: open list, live texture.
    unsafe {
        if before != 0x800 {
            list.list.ResourceBarrier(&[crate::host::transition(
                tex,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
        }
    }
    let (buf, fp) = crate::capture::read_back(&host.device, &list.list, tex).unwrap();
    // SAFETY: open list, live texture.
    unsafe {
        if before != 0x800 {
            list.list.ResourceBarrier(&[crate::host::transition(
                tex,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATES(before as i32),
            )]);
        }
    }
    list.close();
    host.execute(&[&list]);
    crate::resource::host_transition(guest_tex, before);

    let f = fp.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: GPU copy finished; mapping spans the footprint rows.
    unsafe {
        buf.Map(0, None, Some(&mut data)).unwrap();
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
        let mut out = Vec::with_capacity(d.Width as usize * d.Height as usize * 4);
        for y in 0..d.Height as usize {
            out.extend_from_slice(&rows[y * pitch..][..d.Width as usize * 4]);
        }
        buf.Unmap(0, None);
        out
    }
}

fn assert_rgba_approx(actual: &[u8], expected: [u8; 4], label: &str) {
    assert!(!actual.is_empty());
    for (i, px) in actual.as_chunks::<4>().0.iter().enumerate() {
        for c in 0..4 {
            assert!(
                (i16::from(px[c]) - i16::from(expected[c])).abs() <= 1,
                "{label} pixel {i} channel {c}: got {px:?}, expected {expected:?}"
            );
        }
    }
}

#[test]
fn offscreen_multipass_readback_and_last_writer_correlation() {
    let Some(host) = crate::host::host() else {
        eprintln!("skipped offscreen multipass test: no host D3D12 device");
        return;
    };

    let null: This = std::ptr::null();
    let iid = |g: GUID| guid(g.data1, g.data2, g.data3, g.data4);
    let (mut queue, mut allocator, mut list1, mut list2) = (null, null, null, null);
    let queue_desc = [0u32; 4];
    assert_eq!(
        crate::queue::create(
            null,
            queue_desc.as_ptr().cast(),
            &iid(ID3D12CommandQueue::IID),
            &mut queue
        ),
        S_OK
    );
    assert_eq!(
        crate::list::create_allocator(null, 0, &iid(ID3D12CommandAllocator::IID), &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list1
        ),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list2
        ),
        S_OK
    );

    let make_tex = |flags: u32, samples: u32| -> This {
        #[repr(C)]
        struct ResourceDesc {
            dimension: u32,
            alignment: u64,
            width: u64,
            height: u32,
            depth_or_array: u16,
            mips: u16,
            format: u32,
            sample_count: u32,
            sample_quality: u32,
            layout: u32,
            flags: u32,
        }
        let desc = ResourceDesc {
            dimension: 3, // TEXTURE2D
            alignment: 0,
            width: 64,
            height: 64,
            depth_or_array: 1,
            mips: 1,
            format: 28, // R8G8B8A8_UNORM
            sample_count: samples,
            sample_quality: 0,
            layout: 0,
            flags,
        };
        let props = [1u32, 0, 0, 1, 1];
        let mut out = null;
        let create_committed = crate::resource::create_committed;
        assert_eq!(
            create_committed(
                null,
                props.as_ptr().cast(),
                0,
                (&raw const desc).cast(),
                0,
                null.cast(),
                &iid(ID3D12Resource::IID),
                &mut out
            ),
            S_OK
        );
        out
    };

    let rtv_tex = make_tex(1, 1);
    let uav_tex = make_tex(4, 1);
    let mid_tex = make_tex(0, 1);
    let copy_dst = make_tex(0, 1);
    let msaa_tex = make_tex(1, 4);
    let resolve_dst = make_tex(0, 1);
    let dropped_tex = make_tex(1, 1);
    let reset_tex = make_tex(1, 1);
    let reset_after_tex = make_tex(1, 1);

    // Descriptor heaps: RTV heap (kind=2, count=5) and shader-visible view heap (kind=0, flags=1, count=4).
    let (mut rtv_heap, mut view_heap) = (null, null);
    let rtv_heap_desc = [2u32, 5, 0, 1];
    let view_heap_desc = [0u32, 4, 1, 1];
    assert_eq!(
        crate::descriptor::create_heap(
            null,
            rtv_heap_desc.as_ptr().cast(),
            &iid(ID3D12DescriptorHeap::IID),
            &mut rtv_heap
        ),
        S_OK
    );
    assert_eq!(
        crate::descriptor::create_heap(
            null,
            view_heap_desc.as_ptr().cast(),
            &iid(ID3D12DescriptorHeap::IID),
            &mut view_heap
        ),
        S_OK
    );
    let rtv_base = crate::descriptor::heap(rtv_heap).start;
    let view_base = crate::descriptor::heap(view_heap).start;

    crate::descriptor::create_rtv(null, rtv_tex, null.cast(), rtv_base);
    crate::descriptor::create_rtv(null, msaa_tex, null.cast(), rtv_base + 128);
    crate::descriptor::create_rtv(null, dropped_tex, null.cast(), rtv_base + 256);
    crate::descriptor::create_rtv(null, reset_tex, null.cast(), rtv_base + 384);
    crate::descriptor::create_rtv(null, reset_after_tex, null.cast(), rtv_base + 512);
    crate::descriptor::create_uav(null, uav_tex, null, null.cast(), view_base);

    let gfx_pso_rtv = make_graphics_pso([0.25, 0.5, 0.75, 1.0], 1);
    let gfx_pso_msaa = make_graphics_pso([0.2, 0.6, 0.8, 1.0], 4);
    let cs_pso = make_compute_uav_fill_pso([0.8, 0.2, 0.4, 1.0]);

    type ClearRtv = extern "system" fn(This, u64, *const [f32; 4], u32, *const u8);
    type Targets = extern "system" fn(This, u32, *const u64, i32, *const u64);
    type SetPipeline = extern "system" fn(This, This);
    type Topology = extern "system" fn(This, u32);
    type Viewports = extern "system" fn(This, u32, *const D3D12_VIEWPORT);
    type Scissors = extern "system" fn(This, u32, *const RECT);
    type Draw = extern "system" fn(This, u32, u32, u32, u32);
    type SetHeaps = extern "system" fn(This, u32, *const This);
    type ComputeTable = extern "system" fn(This, u32, u64);
    type Dispatch = extern "system" fn(This, u32, u32, u32);
    type CopyRes = extern "system" fn(This, This, This);
    type Resolve = extern "system" fn(This, This, u32, This, u32, u32);
    type Close = extern "system" fn(This) -> i32;
    type Reset = extern "system" fn(This, This, This) -> i32;
    type Execute = extern "system" fn(This, u32, *const This);
    type Release = extern "system" fn(This) -> u32;

    #[repr(C)]
    struct CopyLocation {
        resource: This,
        kind: u32,
        subresource: u64,
        _rest: [u32; 6],
    }
    type CopyRegion = extern "system" fn(
        This,
        *const CopyLocation,
        u32,
        u32,
        u32,
        *const CopyLocation,
        *const u8,
    );

    let vp = D3D12_VIEWPORT {
        Width: 64.0,
        Height: 64.0,
        MaxDepth: 1.0,
        ..Default::default()
    };
    let sc = RECT {
        left: 0,
        top: 0,
        right: 64,
        bottom: 64,
    };

    // Pass 1: ClearRenderTargetView + Graphics DrawInstanced into rtv_tex.
    slot::<SetHeaps>(list1, 0x100)(list1, 1, &view_heap);
    slot::<ClearRtv>(list1, 0x1a0)(list1, rtv_base, &[0.0, 0.0, 0.0, 1.0], 0, null.cast());
    slot::<Targets>(list1, 0x190)(list1, 1, &rtv_base, 0, null.cast());
    slot::<Topology>(list1, 0xa0)(list1, 4);
    slot::<Viewports>(list1, 0xa8)(list1, 1, &vp);
    slot::<Scissors>(list1, 0xb0)(list1, 1, &sc);
    slot::<SetPipeline>(list1, 0xc8)(list1, gfx_pso_rtv);
    slot::<Draw>(list1, 0x60)(list1, 3, 1, 0, 0);

    // Pass 2: Compute Dispatch writing into uav_tex.
    slot::<SetPipeline>(list1, 0xc8)(list1, cs_pso);
    slot::<ComputeTable>(list1, 0x118)(list1, 0, view_base);
    slot::<Dispatch>(list1, 0x70)(list1, 8, 8, 1);

    // Pass 3: CopyTextureRegion (uav_tex -> mid_tex) + CopyResource (mid_tex -> copy_dst).
    let to_mid = CopyLocation {
        resource: mid_tex,
        kind: 0,
        subresource: 0,
        _rest: [0; 6],
    };
    let from_uav = CopyLocation {
        resource: uav_tex,
        kind: 0,
        subresource: 0,
        _rest: [0; 6],
    };
    slot::<CopyRegion>(list1, 0x80)(list1, &to_mid, 0, 0, 0, &from_uav, null.cast());
    slot::<CopyRes>(list1, 0x88)(list1, copy_dst, mid_tex);

    // Pass 4: MSAA 4x RTV draw into msaa_tex + ResolveSubresource -> resolve_dst.
    let msaa_rtv = rtv_base + 128;
    slot::<Targets>(list1, 0x190)(list1, 1, &msaa_rtv, 0, null.cast());
    slot::<SetPipeline>(list1, 0xc8)(list1, gfx_pso_msaa);
    slot::<Draw>(list1, 0x60)(list1, 3, 1, 0, 0);
    slot::<Resolve>(list1, 0x98)(list1, resolve_dst, 0, msaa_tex, 0, 28);

    // Pass 5 (negative): dropped draw (null pipeline) targeting dropped_tex.
    let dropped_rtv = rtv_base + 256;
    slot::<Targets>(list1, 0x190)(list1, 1, &dropped_rtv, 0, null.cast());
    slot::<SetPipeline>(list1, 0xc8)(list1, null);
    slot::<Draw>(list1, 0x60)(list1, 3, 1, 0, 0);

    // Negative 2: unsubmitted list2 cleared, closed, reset() without execute(),
    // then reused for `reset_after_tex` so pre-reset writes must not leak.
    let reset_rtv = rtv_base + 384;
    let reset_after_rtv = rtv_base + 512;
    slot::<ClearRtv>(list2, 0x1a0)(list2, reset_rtv, &[1.0, 1.0, 1.0, 1.0], 0, null.cast());
    assert_eq!(slot::<Close>(list2, 0x48)(list2), S_OK);
    assert_eq!(slot::<Reset>(list2, 0x50)(list2, allocator, null), S_OK);
    slot::<ClearRtv>(list2, 0x1a0)(
        list2,
        reset_after_rtv,
        &[0.0, 1.0, 0.0, 1.0],
        0,
        null.cast(),
    );
    assert_eq!(slot::<Close>(list2, 0x48)(list2), S_OK);

    assert_eq!(slot::<Close>(list1, 0x48)(list1), S_OK);
    let lists = [list1, list2];
    slot::<Execute>(queue, 0x50)(queue, 2, lists.as_ptr());
    assert_eq!(slot::<Reset>(list1, 0x50)(list1, allocator, null), S_OK);
    assert_eq!(slot::<Reset>(list2, 0x50)(list2, allocator, null), S_OK);

    // Verify GPU pixel readback on rtv_tex, copy_dst, and resolve_dst.
    assert_rgba_approx(
        &readback_rgba8(host, rtv_tex),
        [64, 128, 191, 255],
        "rtv_tex",
    );
    assert_rgba_approx(
        &readback_rgba8(host, copy_dst),
        [204, 51, 102, 255],
        "copy_dst",
    );
    assert_rgba_approx(
        &readback_rgba8(host, resolve_dst),
        [51, 153, 204, 255],
        "resolve_dst",
    );

    // Verify last_writer and writer_chain provenance.
    let rtv_writer = crate::capture::last_writer(rtv_tex as usize).expect("rtv_tex writer");
    assert_eq!(rtv_writer.kind, WriteKind::Draw);
    assert_eq!(rtv_writer.slot, Some(0));
    assert_eq!(
        (rtv_writer.width, rtv_writer.height, rtv_writer.format),
        (64, 64, 28)
    );

    let copy_chain = crate::capture::writer_chain(copy_dst as usize, 8);
    assert_eq!(copy_chain.len(), 3, "copy_dst chain: {copy_chain:?}");
    assert_eq!(copy_chain[0].kind, WriteKind::CopyResource);
    assert_eq!(copy_chain[0].target, copy_dst as usize);
    assert_eq!(copy_chain[0].source, Some(mid_tex as usize));
    assert_eq!(copy_chain[1].kind, WriteKind::CopyTextureRegion);
    assert_eq!(copy_chain[1].target, mid_tex as usize);
    assert_eq!(copy_chain[1].source, Some(uav_tex as usize));
    assert_eq!(copy_chain[2].kind, WriteKind::Dispatch);
    assert_eq!(copy_chain[2].target, uav_tex as usize);

    let resolve_chain = crate::capture::writer_chain(resolve_dst as usize, 8);
    assert_eq!(
        resolve_chain.len(),
        2,
        "resolve_dst chain: {resolve_chain:?}"
    );
    assert_eq!(resolve_chain[0].kind, WriteKind::ResolveSubresource);
    assert_eq!(resolve_chain[0].target, resolve_dst as usize);
    assert_eq!(resolve_chain[0].source, Some(msaa_tex as usize));
    assert_eq!(resolve_chain[1].kind, WriteKind::Draw);
    assert_eq!(resolve_chain[1].target, msaa_tex as usize);

    assert!(
        crate::capture::last_writer(dropped_tex as usize).is_none(),
        "dropped draw must not record a writer"
    );
    assert!(
        crate::capture::last_writer(reset_tex as usize).is_none(),
        "reset-without-execute list must not record a writer"
    );
    let after_writer =
        crate::capture::last_writer(reset_after_tex as usize).expect("reused list writer");
    assert_eq!(after_writer.kind, WriteKind::ClearRtv);

    // SAFETY: plain query.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    assert_eq!(
        host.log_messages(),
        None,
        "D3D12 debug layer reported an error"
    );

    for obj in [
        rtv_tex,
        uav_tex,
        mid_tex,
        copy_dst,
        msaa_tex,
        resolve_dst,
        dropped_tex,
        reset_tex,
        reset_after_tex,
        rtv_heap,
        view_heap,
        gfx_pso_rtv,
        gfx_pso_msaa,
        cs_pso,
        list1,
        list2,
        allocator,
        queue,
    ] {
        assert_eq!(slot::<Release>(obj, 0x10)(obj), 0);
    }
}

#[test]
fn present_source_trace_env_correlates_in_isolated_child() {
    const CHILD: &str = "VICTORY_PRESENT_TRACE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "pipeline::present_correlation_test::present_source_trace_env_correlates_in_isolated_child",
                "--exact",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("VICTORY_PRESENT_SOURCE_TRACE", "1")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "child test failed:\n{text}");
        assert!(text.contains("1 passed"), "child ran no test:\n{text}");
        return;
    }

    assert!(crate::capture::present_source_trace_on());
    let tex_a = 0x5000_0001usize;
    let tex_b = 0x5000_0002usize;
    crate::capture::commit_writes(
        1,
        1,
        0,
        &[
            crate::capture::WriteRecord {
                target: tex_a,
                source: None,
                kind: WriteKind::Draw,
                slot: Some(0),
                subresource: 0,
                width: 64,
                height: 64,
                format: 28,
                pipeline: "vs_ps".into(),
                submission: 0,
                list_index: 0,
                op_index: 0,
                frame: 0,
            },
            crate::capture::WriteRecord {
                target: tex_b,
                source: Some(tex_a),
                kind: WriteKind::CopyResource,
                slot: None,
                subresource: 0,
                width: 64,
                height: 64,
                format: 28,
                pipeline: String::new(),
                submission: 0,
                list_index: 0,
                op_index: 1,
                frame: 0,
            },
        ],
    );
    crate::capture::correlate_present(
        1,
        &[PresentPlane {
            plane: 0,
            composition_index: 0,
            resource: tex_b,
            width: 64,
            height: 64,
            format: 28,
        }],
    );
    let chain = crate::capture::writer_chain(tex_b, 8);
    assert_eq!(chain.len(), 2);
    assert_eq!(chain[0].kind, WriteKind::CopyResource);
    assert_eq!(chain[1].kind, WriteKind::Draw);
    crate::capture::end_frame(1);
}

#[test]
fn dropped_compute_translation_marks_bound_input_never_host_written_in_isolated_child() {
    const CHILD: &str = "VICTORY_DROPPED_CS_INPUT_TRACE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "pipeline::present_correlation_test::dropped_compute_translation_marks_bound_input_never_host_written_in_isolated_child",
                "--exact",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("VICTORY_PRESENT_SOURCE_TRACE", "1")
            .output()
            .unwrap();
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(out.status.success(), "child test failed:\n{text}");
        assert!(text.contains("1 passed"), "child ran no test:\n{text}");
        return;
    }

    let Some(host) = crate::host::host() else {
        eprintln!("skipped dropped compute correlation test: no host D3D12 device");
        return;
    };

    let null: This = std::ptr::null();
    let iid = |g: GUID| guid(g.data1, g.data2, g.data3, g.data4);
    let (mut queue, mut allocator, mut list1) = (null, null, null);
    let queue_desc = [0u32; 4];
    assert_eq!(
        crate::queue::create(
            null,
            queue_desc.as_ptr().cast(),
            &iid(ID3D12CommandQueue::IID),
            &mut queue
        ),
        S_OK
    );
    assert_eq!(
        crate::list::create_allocator(null, 0, &iid(ID3D12CommandAllocator::IID), &mut allocator),
        S_OK
    );
    assert_eq!(
        crate::list::create_list(
            null,
            0,
            0,
            allocator,
            null,
            &iid(ID3D12GraphicsCommandList::IID),
            &mut list1
        ),
        S_OK
    );

    #[repr(C)]
    struct ResourceDesc {
        dimension: u32,
        alignment: u64,
        width: u64,
        height: u32,
        depth_or_array: u16,
        mips: u16,
        format: u32,
        sample_count: u32,
        sample_quality: u32,
        layout: u32,
        flags: u32,
    }
    let props = [1u32, 0, 0, 1, 1];
    let tex_desc = ResourceDesc {
        dimension: 3, // TEXTURE2D
        alignment: 0,
        width: 64,
        height: 64,
        depth_or_array: 1,
        mips: 1,
        format: 28,
        sample_count: 1,
        sample_quality: 0,
        layout: 0,
        flags: 1, // ALLOW_RENDER_TARGET
    };
    let buf_desc = ResourceDesc {
        dimension: 1, // BUFFER
        alignment: 0,
        width: 64,
        height: 1,
        depth_or_array: 1,
        mips: 1,
        format: 0,
        sample_count: 1,
        sample_quality: 0,
        layout: 1, // ROW_MAJOR
        flags: 4,  // ALLOW_UNORDERED_ACCESS
    };
    let (mut out_rtv, mut unwritten_buf) = (null, null);
    assert_eq!(
        crate::resource::create_committed(
            null,
            props.as_ptr().cast(),
            0,
            (&raw const tex_desc).cast(),
            0,
            null.cast(),
            &iid(ID3D12Resource::IID),
            &mut out_rtv
        ),
        S_OK
    );
    assert_eq!(
        crate::resource::create_committed(
            null,
            props.as_ptr().cast(),
            0,
            (&raw const buf_desc).cast(),
            0,
            null.cast(),
            &iid(ID3D12Resource::IID),
            &mut unwritten_buf
        ),
        S_OK
    );
    let buf_addr = crate::resource::buffer_bytes(unwritten_buf)
        .expect("unwritten_buf bytes")
        .0 as usize;

    let (mut rtv_heap, mut view_heap) = (null, null);
    let rtv_heap_desc = [2u32, 1, 0, 1];
    let view_heap_desc = [0u32, 2, 1, 1];
    assert_eq!(
        crate::descriptor::create_heap(
            null,
            rtv_heap_desc.as_ptr().cast(),
            &iid(ID3D12DescriptorHeap::IID),
            &mut rtv_heap
        ),
        S_OK
    );
    assert_eq!(
        crate::descriptor::create_heap(
            null,
            view_heap_desc.as_ptr().cast(),
            &iid(ID3D12DescriptorHeap::IID),
            &mut view_heap
        ),
        S_OK
    );
    let rtv_base = crate::descriptor::heap(rtv_heap).start;
    let view_base = crate::descriptor::heap(view_heap).start;

    crate::descriptor::create_rtv(null, out_rtv, null.cast(), rtv_base);
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: 0,
                NumElements: 4,
                StructureByteStride: 0,
                Flags: D3D12_BUFFER_SRV_FLAG_NONE,
            },
        },
    };
    const _: () = assert!(size_of::<D3D12_SHADER_RESOURCE_VIEW_DESC>() == 40);
    crate::descriptor::create_srv(null, unwritten_buf, (&raw const srv_desc).cast(), view_base);

    // Upstream compute pipeline whose container fails victory_shader translation.
    let bad_cs_bytes = b"not-a-valid-gcn-dxbc-container";
    let bad_cs_hash = blake3::hash(bad_cs_bytes).to_hex()[..16].to_owned();
    let cs_desc = ComputeHead {
        root: null,
        stage: Bytecode {
            code: bad_cs_bytes.as_ptr(),
            size: bad_cs_bytes.len(),
        },
    };
    let mut bad_cs_pso = null;
    assert_eq!(
        create_compute(null, &cs_desc, &PIPELINE_IIDS[0], &mut bad_cs_pso),
        S_OK
    );

    let gfx_srv_pso = make_graphics_srv_read_pso();

    type ClearRtv = extern "system" fn(This, u64, *const [f32; 4], u32, *const u8);
    type Targets = extern "system" fn(This, u32, *const u64, i32, *const u64);
    type SetPipeline = extern "system" fn(This, This);
    type Topology = extern "system" fn(This, u32);
    type Viewports = extern "system" fn(This, u32, *const D3D12_VIEWPORT);
    type Scissors = extern "system" fn(This, u32, *const RECT);
    type Draw = extern "system" fn(This, u32, u32, u32, u32);
    type SetHeaps = extern "system" fn(This, u32, *const This);
    type ComputeTable = extern "system" fn(This, u32, u64);
    type GraphicsTable = extern "system" fn(This, u32, u64);
    type Dispatch = extern "system" fn(This, u32, u32, u32);
    type Close = extern "system" fn(This) -> i32;
    type Reset = extern "system" fn(This, This, This) -> i32;
    type Execute = extern "system" fn(This, u32, *const This);
    type Release = extern "system" fn(This) -> u32;

    let vp = D3D12_VIEWPORT {
        Width: 64.0,
        Height: 64.0,
        MaxDepth: 1.0,
        ..Default::default()
    };
    let sc = RECT {
        left: 0,
        top: 0,
        right: 64,
        bottom: 64,
    };

    // Upstream compute dispatch drops because its CS translation failed.
    slot::<SetHeaps>(list1, 0x100)(list1, 1, &view_heap);
    slot::<SetPipeline>(list1, 0xc8)(list1, bad_cs_pso);
    slot::<ComputeTable>(list1, 0x118)(list1, 0, view_base);
    slot::<Dispatch>(list1, 0x70)(list1, 1, 1, 1);

    // Downstream fullscreen graphics PSO binds `unwritten_buf` as SRV and draws into `out_rtv`.
    slot::<ClearRtv>(list1, 0x1a0)(list1, rtv_base, &[0.0, 0.0, 0.0, 1.0], 0, null.cast());
    slot::<Targets>(list1, 0x190)(list1, 1, &rtv_base, 0, null.cast());
    slot::<Topology>(list1, 0xa0)(list1, 4);
    slot::<Viewports>(list1, 0xa8)(list1, 1, &vp);
    slot::<Scissors>(list1, 0xb0)(list1, 1, &sc);
    slot::<SetPipeline>(list1, 0xc8)(list1, gfx_srv_pso);
    slot::<GraphicsTable>(list1, 0x120)(list1, 0, view_base);
    slot::<Draw>(list1, 0x60)(list1, 3, 1, 0, 0);

    assert_eq!(slot::<Close>(list1, 0x48)(list1), S_OK);
    slot::<Execute>(queue, 0x50)(queue, 1, &list1);
    assert_eq!(slot::<Reset>(list1, 0x50)(list1, allocator, null), S_OK);

    // Pixel readback confirms output stays 0 (unwritten buffer reads zeros).
    assert_rgba_approx(&readback_rgba8(host, out_rtv), [0, 0, 0, 0], "out_rtv");

    let frame = crate::queue::presented() + 1;
    let reports = crate::capture::input_reports(frame);
    let report = reports
        .iter()
        .find(|r| r.targets.contains(&(out_rtv as usize)))
        .expect("downstream draw input provenance report missing");
    assert_eq!(report.inputs.len(), 1);
    assert_eq!(report.inputs[0].input.kind, InputKind::Srv);
    assert_eq!(report.inputs[0].input.resource, buf_addr);
    assert!(report.inputs[0].writer.is_none());
    assert!(
        report.inputs[0].status.contains("never host-written"),
        "expected 'never host-written', got: {}",
        report.inputs[0].status
    );
    assert!(
        report.inputs[0]
            .status
            .contains("dropped upstream CS translation"),
        "expected 'dropped upstream CS translation', got: {}",
        report.inputs[0].status
    );
    assert!(
        report.inputs[0].status.contains(&bad_cs_hash),
        "expected failed CS hash {bad_cs_hash}, got: {}",
        report.inputs[0].status
    );
    assert!(
        report.inputs[0]
            .status
            .contains("dropped upstream dispatch"),
        "expected 'dropped upstream dispatch', got: {}",
        report.inputs[0].status
    );

    crate::capture::correlate_present(
        frame,
        &[PresentPlane {
            plane: 0,
            composition_index: 0,
            resource: out_rtv as usize,
            width: 64,
            height: 64,
            format: 28,
        }],
    );
    crate::capture::end_frame(frame);

    // SAFETY: plain query.
    assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    assert_eq!(
        host.log_messages(),
        None,
        "D3D12 debug layer reported an error"
    );

    for obj in [
        out_rtv,
        unwritten_buf,
        rtv_heap,
        view_heap,
        bad_cs_pso,
        gfx_srv_pso,
        list1,
        allocator,
        queue,
    ] {
        assert_eq!(slot::<Release>(obj, 0x10)(obj), 0);
    }
}
