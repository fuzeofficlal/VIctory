//! End-to-end GPU tests for sub-dword LDS reads and writes (`ds_write_b8*`,
//! `ds_write_b16*`, `ds_read_{u8,i8,u16,i16}`, and `_d16` / `_d16_hi`),
//! validated with `dxil.dll` when available and executed across every SM 6.6
//! hardware adapter and WARP.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_BARRIER: u32 = 0xBF8A_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// `v_cmpx_eq_u32 0, v0`: narrows EXEC to lane 0 so a 64-thread wave32 group
/// (2 waves -> no `[WaveSize(32)]` attribute, compatible with WARP's 4-wide
/// SIMD as well as NVIDIA/AMD hardware) runs single-writer LDS sequences
/// through the guarded execution path.
const V_CMPX_LANE0: u32 = 0x7DA4_0080;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 128..192 = inline int, 256.. = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    ((op as u32 & 0x3F) << 25)
        | ((vdst as u32) << 17)
        | ((vsrc1 as u32) << 9)
        | (src0 as u32 & 0x1FF)
}

/// VOP1 `v_mov_b32 vdst, literal` (64-bit encoding with 32-bit literal).
const fn v_mov_lit(vdst: u8, lit: u32) -> [u32; 2] {
    [vop1(0x001, vdst, 255), lit]
}

/// VOP2 `op vdst, literal, vsrc1`.
const fn vop2_lit(op: u8, vdst: u8, lit: u32, vsrc1: u8) -> [u32; 2] {
    [vop2(op, vdst, 255, vsrc1), lit]
}

/// GFX10 single-offset DS instruction.
const fn ds(op: u16, offset: u16, addr: u8, data0: u8, dst: u8) -> [u32; 2] {
    [
        0xD800_0000 | ((op as u32) << 18) | (offset as u32),
        (addr as u32) | ((data0 as u32) << 8) | ((dst as u32) << 24),
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (1 << 12),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | (vaddr as u32),
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

const fn inline_u32(n: u8) -> u16 {
    128 + n as u16
}

/// Builds, lowers, and `dxil.dll`-validates a synthetic compute shader with
/// 4 root constants in `s0..s3`, descriptor table at `s4` (raw UAV at index 0),
/// `threads` group size, and `lds_blocks` (512 bytes per block) of LDS.
fn build_lds_cs(words: &[u32], threads: u32, lds_blocks: u32) -> HostShader {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (lds_blocks << 15) | (5 << 1),
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    let dxil_path = validator::Validator::default_path();
    if let Ok(mut v) = validator::Validator::load(dxil_path.as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        eprintln!("validated DXIL via {:?}", dxil_path);
    } else {
        eprintln!("no dxil.dll found; validator skipped");
    }

    shader
}

/// Runs `shader` on every available SM 6.6 adapter (including NVIDIA/AMD GPUs
/// and WARP) under the global GPU test lock.
fn run_on_all_adapters(
    shader: &HostShader,
    uav_dwords: u32,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], &[u32]),
) {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let adapters = Host::all_adapters();
    assert!(
        !adapters.is_empty(),
        "no SM 6.6 D3D12 adapter or WARP found"
    );
    for (name, host) in &adapters {
        eprintln!("executing LDS compute shader on adapter: {name}");
        host.clear_debug_messages();
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{name}: compute_pipeline failed: {e}"));
        assert_eq!(host.take_debug_error(), None, "{name}: PSO debug error");
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            host.clear_debug_messages();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(uav_dwords)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{name}: dispatch failed: {e}"));
            assert_eq!(
                host.take_debug_error(),
                None,
                "{name}: dispatch debug error"
            );
            check(name, i, inputs, &out[0]);
        }
    }
}

#[test]
fn subdword_writes_and_reads_roundtrip_across_offsets() {
    // Assemble LDS bytes 0..8 via ds_write_b8, ds_write_b8_d16_hi,
    // ds_write_b16 (odd offset 1 and straddling offset 3), and
    // ds_write_b16_d16_hi, plus a straddling write at the very last byte of
    // the 512-byte LDS allocation (byte 511), then read back with ds_read_u8,
    // ds_read_i8, ds_read_u16, and ds_read_i16.
    let mut w = vec![
        s_mov_b32(5, 0x80),
        s_load_dwordx4(8, 4)[0],
        s_load_dwordx4(8, 4)[1],
        V_CMPX_LANE0,
        vop1(0x001, 1, sgpr(0)),       // v1 = c0 (0x9B80_7FFE)
        vop1(0x001, 2, sgpr(1)),       // v2 = c1 (0xC3D4_A1B2)
        vop1(0x001, 3, sgpr(2)),       // v3 = dynamic read offset
        vop1(0x001, 5, inline_u32(0)), // v5 = 0 (LDS base)
    ];
    // Clear dwords 0, 1, and last dword 127 (offset 508).
    w.extend_from_slice(&ds(0x00d, 0, 5, 5, 0));
    w.extend_from_slice(&ds(0x00d, 4, 5, 5, 0));
    w.extend_from_slice(&ds(0x00d, 508, 5, 5, 0));
    w.push(S_WAITCNT_ALL);

    // Byte 0 = v1[7:0] (0xFE) via ds_write_b8
    w.extend_from_slice(&ds(0x01e, 0, 5, 1, 0));
    // Bytes 1..3 = v2[15:0] (0xB2, 0xA1) at odd offset 1 via ds_write_b16
    w.extend_from_slice(&ds(0x01f, 1, 5, 2, 0));
    // Bytes 3..5 = v2[31:16] (0xD4, 0xC3) straddling dwords 0 & 1 via ds_write_b16_d16_hi
    w.extend_from_slice(&ds(0x0a1, 3, 5, 2, 0));
    // Byte 5 = v1[23:16] (0x80) via ds_write_b8_d16_hi
    w.extend_from_slice(&ds(0x0a0, 5, 5, 1, 0));
    // Byte 511 (last byte of 512-byte LDS): write 0xA1B2 via ds_write_b16;
    // low byte 0xB2 lands at byte 511, high byte 0xA1 is dropped out of bounds.
    w.extend_from_slice(&ds(0x01f, 511, 5, 2, 0));
    w.push(S_WAITCNT_ALL);

    // Resulting LDS bytes 0..6: [0xFE, 0xB2, 0xA1, 0xD4, 0xC3, 0x80]
    w.extend_from_slice(&ds(0x03a, 0, 3, 0, 6)); // ds_read_u8 v6, v3
    w.extend_from_slice(&ds(0x039, 0, 3, 0, 7)); // ds_read_i8 v7, v3
    w.extend_from_slice(&ds(0x03c, 0, 3, 0, 8)); // ds_read_u16 v8, v3
    w.extend_from_slice(&ds(0x03b, 0, 3, 0, 9)); // ds_read_i16 v9, v3
    w.extend_from_slice(&ds(0x03c, 511, 5, 0, 10)); // ds_read_u16 v10, v5 offset:511
    w.push(S_WAITCNT_ALL);

    w.extend_from_slice(&buffer_store_dword(6, 8, 0));
    w.extend_from_slice(&buffer_store_dword(7, 8, 4));
    w.extend_from_slice(&buffer_store_dword(8, 8, 8));
    w.extend_from_slice(&buffer_store_dword(9, 8, 12));
    w.extend_from_slice(&buffer_store_dword(10, 8, 16));
    w.push(S_ENDPGM);

    let shader = build_lds_cs(&w, 64, 1);
    let bytes = [0xFEu8, 0xB2, 0xA1, 0xD4, 0xC3, 0x80, 0x00];
    let cases: Vec<[u32; 4]> = (0..5u32)
        .map(|off| [0x9B80_7FFE, 0xC3D4_A1B2, off, 0])
        .collect();

    run_on_all_adapters(&shader, 5, &cases, |adapter, off, _, got| {
        let b0 = bytes[off];
        let b1 = bytes[off + 1];
        let u16_val = u16::from_le_bytes([b0, b1]);
        assert_eq!(got[0], u32::from(b0), "{adapter} off={off} ds_read_u8");
        assert_eq!(
            got[1],
            (b0 as i8 as i32) as u32,
            "{adapter} off={off} ds_read_i8"
        );
        assert_eq!(
            got[2],
            u32::from(u16_val),
            "{adapter} off={off} ds_read_u16"
        );
        assert_eq!(
            got[3],
            (u16_val as i16 as i32) as u32,
            "{adapter} off={off} ds_read_i16"
        );
        assert_eq!(got[4], 0x00B2, "{adapter} off={off} last-byte straddle");
    });
}

#[test]
fn d16_and_d16_hi_preserve_the_untouched_destination_half() {
    // Preload each destination VGPR from `c2` (0xCAFE_1357) while encoding a
    // distinct decoy register `v2` (0xDEAD_BEEF) in the DS instruction's
    // `data0` field, verifying the preserved 16-bit half comes strictly from
    // the destination's prior SSA value across aligned, odd, and straddling
    // byte offsets.
    let mut w = vec![
        s_mov_b32(5, 0x80),
        s_load_dwordx4(8, 4)[0],
        s_load_dwordx4(8, 4)[1],
        V_CMPX_LANE0,
        vop1(0x001, 1, sgpr(0)),       // v1 = LDS dword 0 (0xAB12_80FE)
        vop1(0x001, 2, sgpr(1)),       // v2 = LDS dword 1 / decoy data0 (0xDEAD_CDBE)
        vop1(0x001, 3, sgpr(2)),       // v3 = preload (0xCAFE_1357)
        vop1(0x001, 4, sgpr(3)),       // v4 = byte address
        vop1(0x001, 5, inline_u32(0)), // v5 = 0
    ];
    w.extend_from_slice(&ds(0x00d, 0, 5, 1, 0)); // LDS[0..4] = v1
    w.extend_from_slice(&ds(0x00d, 4, 5, 2, 0)); // LDS[4..8] = v2
    w.push(S_WAITCNT_ALL);

    // Preload v6..v11 with v3 (0xCAFE_1357), then issue all 6 _d16/_d16_hi ops
    // with `data0 = v2` (0xDEAD_BEEF) so reading `data0` by mistake would fail.
    for (reg, op) in [
        (6u8, 0x0a2u16), // ds_read_u8_d16
        (7, 0x0a4),      // ds_read_i8_d16
        (8, 0x0a3),      // ds_read_u8_d16_hi
        (9, 0x0a5),      // ds_read_i8_d16_hi
        (10, 0x0a6),     // ds_read_u16_d16
        (11, 0x0a7),     // ds_read_u16_d16_hi
    ] {
        w.push(vop1(0x001, reg, vgpr(3)));
        w.extend_from_slice(&ds(op, 0, 4, 2, reg));
    }
    w.push(S_WAITCNT_ALL);

    for i in 0..6u16 {
        w.extend_from_slice(&buffer_store_dword(6 + i as u8, 8, i * 4));
    }
    w.push(S_ENDPGM);

    let shader = build_lds_cs(&w, 64, 1);
    let d0 = 0xAB12_80FE_u32;
    let d1 = 0xDEAD_CDBE_u32;
    let preload = 0xCAFE_1357_u32;
    let mut lds = [0u8; 8];
    lds[..4].copy_from_slice(&d0.to_le_bytes());
    lds[4..].copy_from_slice(&d1.to_le_bytes());

    let cases: Vec<[u32; 4]> = (0..4u32).map(|at| [d0, d1, preload, at]).collect();
    run_on_all_adapters(&shader, 6, &cases, |adapter, at, _, got| {
        let b = lds[at];
        let h = u16::from_le_bytes([lds[at], lds[at + 1]]);
        let hi_keep = preload & 0xFFFF_0000;
        let lo_keep = preload & 0x0000_FFFF;
        let i8_16 = (b as i8 as i16 as u16) as u32;

        assert_eq!(got[0], hi_keep | u32::from(b), "{adapter} at={at} u8_d16");
        assert_eq!(got[1], hi_keep | i8_16, "{adapter} at={at} i8_d16");
        assert_eq!(
            got[2],
            lo_keep | (u32::from(b) << 16),
            "{adapter} at={at} u8_d16_hi"
        );
        assert_eq!(
            got[3],
            lo_keep | (i8_16 << 16),
            "{adapter} at={at} i8_d16_hi"
        );
        assert_eq!(got[4], hi_keep | u32::from(h), "{adapter} at={at} u16_d16");
        assert_eq!(
            got[5],
            lo_keep | (u32::from(h) << 16),
            "{adapter} at={at} u16_d16_hi"
        );
    });
}

#[test]
fn cross_lane_subdword_lds_exchange_across_threads() {
    // 64-thread workgroup (`v0 = SV_GroupThreadID.x`, 2 wave32s):
    // 1. Concurrent sub-dword writes into shared 32-bit LDS dwords:
    //    - Each lane writes byte `(0x80 | tid)` to `LDS[tid]` via `ds_write_b8`
    //      (4 lanes share each 32-bit dword).
    //    - Each lane writes halfword `0x5000 | (tid * 0x101)` to
    //      `LDS[128 + tid * 2]` via `ds_write_b16` (2 lanes share each dword).
    // 2. `s_barrier` synchronizes the entire 64-thread group across waves.
    // 3. Cross-lane reads & chained half-register packing:
    //    - Reads partner `tid ^ 3`'s signed byte (`ds_read_i8`) and partner
    //      `tid ^ 1`'s unsigned byte (`ds_read_u8`).
    //    - Packs partner `(63 - tid)`'s halfword into low 16 bits via
    //      `ds_read_u16_d16` and partner `(tid ^ 2)`'s halfword into high 16
    //      bits of the SAME VGPR via `ds_read_u16_d16_hi`.
    let mut w = vec![
        s_mov_b32(5, 0x80),
        s_load_dwordx4(8, 4)[0],
        s_load_dwordx4(8, 4)[1],
    ];
    // v1 = 0x80 | v0 (byte value for LDS[tid])
    w.extend_from_slice(&vop2_lit(0x1c, 1, 0x80, 0)); // v_or_b32 v1, 0x80, v0
    w.extend_from_slice(&ds(0x01e, 0, 0, 1, 0)); // ds_write_b8 v0, v1

    // v2 = tid << 1 (halfword byte offset)
    w.push(vop2(0x1a, 2, inline_u32(1), 0)); // v_lshlrev_b32 v2, 1, v0
    // v3 = (tid << 8) | tid; v3 = v3 | 0x5000
    w.push(vop2(0x1a, 3, inline_u32(8), 0)); // v_lshlrev_b32 v3, 8, v0
    w.push(vop2(0x1c, 3, vgpr(0), 3)); // v_or_b32 v3, v0, v3
    w.extend_from_slice(&vop2_lit(0x1c, 3, 0x5000, 3)); // v_or_b32 v3, 0x5000, v3
    w.extend_from_slice(&ds(0x01f, 128, 2, 3, 0)); // ds_write_b16 v2, v3 offset:128

    w.push(S_WAITCNT_ALL);
    w.push(S_BARRIER);

    // Partner addresses:
    // v4 = tid ^ 3; v5 = tid ^ 1
    w.push(vop2(0x1d, 4, inline_u32(3), 0)); // v_xor_b32 v4, 3, v0
    w.push(vop2(0x1d, 5, inline_u32(1), 0)); // v_xor_b32 v5, 1, v0
    w.extend_from_slice(&ds(0x039, 0, 4, 0, 6)); // ds_read_i8 v6, v4 (sign-extended negative!)
    w.extend_from_slice(&ds(0x03a, 0, 5, 0, 7)); // ds_read_u8 v7, v5

    // v8 = (63 - tid) << 1 = (tid ^ 63) << 1; v9 = (tid ^ 2) << 1
    w.push(vop2(0x1d, 8, inline_u32(63), 0)); // v_xor_b32 v8, 63, v0
    w.push(vop2(0x1a, 8, inline_u32(1), 8)); // v_lshlrev_b32 v8, 1, v8
    w.push(vop2(0x1d, 9, inline_u32(2), 0)); // v_xor_b32 v9, 2, v0
    w.push(vop2(0x1a, 9, inline_u32(1), 9)); // v_lshlrev_b32 v9, 1, v9

    // Seed v10 with 0xFFFF_FFFF, overwrite low half via ds_read_u16_d16, then
    // overwrite high half via ds_read_u16_d16_hi (preserving the low half!).
    w.extend_from_slice(&v_mov_lit(10, 0xFFFF_FFFF));
    w.extend_from_slice(&ds(0x0a6, 128, 8, 0, 10)); // ds_read_u16_d16 v10, v8 offset:128
    w.extend_from_slice(&ds(0x0a7, 128, 9, 0, 10)); // ds_read_u16_d16_hi v10, v9 offset:128
    w.push(S_WAITCNT_ALL);

    // Combine v6 ^ v7 into v11, store v10 at out[tid * 2] and v11 at out[tid * 2 + 1].
    w.push(vop2(0x1d, 11, vgpr(6), 7)); // v_xor_b32 v11, v6, v7
    w.push(vop2(0x1a, 12, inline_u32(3), 0)); // v_lshlrev_b32 v12, 3, v0 (tid * 8 bytes)
    w.extend_from_slice(&buffer_store_dword_offen(10, 12, 8));
    w.push(vop2(0x25, 13, inline_u32(4), 12)); // v_add_nc_u32 v13, 4, v12
    w.extend_from_slice(&buffer_store_dword_offen(11, 13, 8));
    w.push(S_ENDPGM);

    let shader = build_lds_cs(&w, 64, 1);
    run_on_all_adapters(&shader, 128, &[[0; 4]], |adapter, _, _, got| {
        for tid in 0..64u32 {
            let half_of = |t: u32| 0x5000 | (t << 8) | t;
            let want_packed = half_of(63 - tid) | (half_of(tid ^ 2) << 16);
            let i8_ext = ((0x80 | (tid ^ 3)) as u8 as i8 as i32) as u32;
            let u8_ext = 0x80 | (tid ^ 1);
            let want_bytes = i8_ext ^ u8_ext;

            assert_eq!(
                got[(tid * 2) as usize],
                want_packed,
                "{adapter} lane {tid} chained d16+d16_hi halfword exchange"
            );
            assert_eq!(
                got[(tid * 2 + 1) as usize],
                want_bytes,
                "{adapter} lane {tid} cross-lane i8/u8 byte exchange"
            );
        }
    });
}
