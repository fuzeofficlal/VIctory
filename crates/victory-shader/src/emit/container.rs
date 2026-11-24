//! The PC container around emitted DXIL: feature flags (`SFI0`), the
//! signatures (`ISG1`, `OSG1`), pipeline state validation (`PSV0`) and the
//! program (`DXIL`), laid out as validator version 1.6 lays them out, plus
//! the container hash that D3D12 checks.

/// One signature element: one row (D3D12 links stages by semantic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SigElement {
    pub name: String,
    pub index: u32,
    /// DXIL semantic kind (0 arbitrary, 1 vertex id, 3 position, 16 target...).
    pub kind: u8,
    /// DXIL component type (4 i32, 5 u32, 9 f32, 1 i1).
    pub comp: u8,
    /// DXIL interpolation mode (0 for none).
    pub interp: u8,
    /// Start row, if the element has a register.
    pub row: Option<u32>,
    pub col: u8,
    pub cols: u8,
    /// Components read (inputs) or written (outputs), relative to `col`.
    pub usage: u8,
    /// 16-bit on the console (declared 32-bit here).
    pub half: bool,
}

impl SigElement {
    fn system_value(&self) -> u32 {
        match self.kind {
            1 => 6,  // vertex id
            2 => 8,  // instance id
            3 => 1,  // position
            4 => 4,  // render target array index
            5 => 5,  // viewport array index
            6 => 2,  // clip distance
            7 => 3,  // cull distance
            10 => 7, // primitive id
            12 => 10,
            13 => 9, // front face
            14 => 66,
            15 => 70,
            16 => 64, // target
            17 => 65, // depth
            18 => 68,
            19 => 67,
            20 => 69,
            30 => 25, // cull primitive
            _ => 0,
        }
    }

    fn sig_comp(&self) -> u32 {
        match self.comp {
            1 | 5 => 1, // bool and u32
            4 => 2,
            9 => 3,
            _ => 0,
        }
    }

    fn mask(&self) -> u8 {
        ((1u8 << self.cols) - 1) << self.col
    }
}

/// `ISG1`/`OSG1`: 32-byte rows, then the names.
pub(crate) fn signature(elements: &[SigElement], input: bool) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((elements.len() as u32).to_le_bytes());
    out.extend(8u32.to_le_bytes());
    // Validator 1.6 writes one name per element in element order, repeats
    // included, except that system values share theirs (it keys names by
    // the semantic's string, which is one static string per system value);
    // the rows sorted by register (unallocated last).
    let mut offsets: Vec<u32> = Vec::new();
    let mut next = 8 + 32 * elements.len() as u32;
    let mut names = Vec::new();
    for (i, e) in elements.iter().enumerate() {
        let shared = (e.kind != 0)
            .then(|| (0..i).find(|&j| elements[j].kind == e.kind))
            .flatten();
        match shared {
            Some(j) => offsets.push(offsets[j]),
            None => {
                offsets.push(next);
                next += e.name.len() as u32 + 1;
                names.push(&e.name);
            }
        }
    }
    let mut order: Vec<usize> = (0..elements.len()).collect();
    order.sort_by_key(|&i| (elements[i].row.unwrap_or(u32::MAX), elements[i].col));
    for i in order {
        let (e, offset) = (&elements[i], offsets[i]);
        let used = e.usage << e.col;
        let rw = if input { used } else { !used & 0xf };
        for v in [
            0,
            offset,
            e.index,
            e.system_value(),
            e.sig_comp(),
            e.row.unwrap_or(u32::MAX),
        ] {
            out.extend(v.to_le_bytes());
        }
        out.extend([e.mask(), rw, 0, 0]);
        out.extend(0u32.to_le_bytes());
    }
    for name in names {
        out.extend(name.as_bytes());
        out.push(0);
    }
    out
}

/// A resource bound by register, as `PSV0` lists it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Binding {
    /// PSV resource type: 1 sampler, 2 CBV.
    pub kind: u32,
    pub space: u32,
    pub register: u32,
    /// DXIL resource kind (13 cbuffer, 14 sampler).
    pub dxil_kind: u32,
}

pub(crate) struct Psv<'a> {
    /// PSV shader kind: 0 pixel, 1 vertex, 5 compute.
    pub stage: u8,
    /// Vertex: writes `SV_Position`. Pixel: writes depth.
    pub flag: bool,
    pub inputs: &'a [SigElement],
    pub outputs: &'a [SigElement],
    /// Mesh: per-primitive outputs.
    pub prims: &'a [SigElement],
    /// Mesh: its state and the group-shared bytes it declares.
    pub mesh: Option<(super::shader::Mesh, u32)>,
    pub bindings: &'a [Binding],
    /// Compute: thread-group size and required wave size.
    pub threads: [u32; 3],
    pub wave_size: Option<u32>,
}

/// `PSV0` with runtime info version 2 (validator 1.6).
pub(crate) fn psv(p: &Psv) -> Vec<u8> {
    let vectors = |els: &[SigElement]| {
        els.iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0)
    };
    let (in_vectors, out_vectors) = (vectors(p.inputs), vectors(p.outputs));
    let mut out = Vec::new();
    let u32s = |out: &mut Vec<u8>, vs: &[u32]| {
        for v in vs {
            out.extend(v.to_le_bytes());
        }
    };
    u32s(&mut out, &[48]);
    let mut info = [0u8; 48];
    info[0] = u8::from(p.flag);
    if p.stage == 0 {
        // PSInfo.SampleFrequency: an input interpolated per sample, or the
        // sample index.
        info[1] = u8::from(
            p.inputs
                .iter()
                .any(|e| matches!(e.interp, 6 | 7) || e.kind == 12),
        );
    }
    let (min, max) = p.wave_size.map_or((0, u32::MAX), |n| (n, n));
    info[16..20].copy_from_slice(&min.to_le_bytes());
    info[20..24].copy_from_slice(&max.to_le_bytes());
    info[24] = p.stage;
    for (k, n) in p.threads.into_iter().enumerate() {
        info[36 + 4 * k..40 + 4 * k].copy_from_slice(&n.to_le_bytes());
    }
    info[28] = p.inputs.len() as u8;
    info[29] = p.outputs.len() as u8;
    info[31] = in_vectors as u8;
    info[32] = out_vectors as u8;
    if let Some((mesh, shared)) = p.mesh {
        info[0..4].copy_from_slice(&shared.to_le_bytes());
        info[12..14].copy_from_slice(&(mesh.max_vertices as u16).to_le_bytes());
        info[14..16].copy_from_slice(&(mesh.max_primitives as u16).to_le_bytes());
        info[26] = vectors(p.prims) as u8;
        info[27] = mesh.topology;
        info[30] = p.prims.len() as u8;
    }
    out.extend(info);
    u32s(&mut out, &[p.bindings.len() as u32]);
    if !p.bindings.is_empty() {
        u32s(&mut out, &[24]);
        for b in p.bindings {
            u32s(
                &mut out,
                &[b.kind, b.space, b.register, b.register, b.dxil_kind, 0],
            );
        }
    }
    // Names of arbitrary semantics (system values use ""), then indices.
    let mut strings = vec![0u8];
    let mut indices: Vec<u32> = Vec::new();
    let mut elements = Vec::new();
    for e in p.inputs.iter().chain(p.outputs).chain(p.prims) {
        let name = if e.kind == 0 {
            let needle: Vec<u8> = e.name.bytes().chain([0]).collect();
            match find(&strings, &needle) {
                Some(at) => at,
                None => {
                    strings.extend(&needle);
                    strings.len() - needle.len()
                }
            }
        } else {
            0
        };
        let index = match indices.iter().position(|&i| i == e.index) {
            Some(at) => at,
            None => {
                indices.push(e.index);
                indices.len() - 1
            }
        };
        let (start, alloc) = match e.row {
            Some(r) => (r as u8, 0x40 | (e.col << 4)),
            None => (0, 0),
        };
        elements.push((name as u32, index as u32, start, e.cols | alloc, e));
    }
    strings.resize(strings.len().next_multiple_of(4), 0);
    u32s(&mut out, &[strings.len() as u32]);
    out.extend(&strings);
    u32s(&mut out, &[indices.len() as u32]);
    u32s(&mut out, &indices);
    if !elements.is_empty() {
        u32s(&mut out, &[16]);
        for (name, index, start, cols, e) in elements {
            u32s(&mut out, &[name, index]);
            out.extend([1, start, cols, e.kind, e.sig_comp() as u8, e.interp, 0, 0]);
        }
    }
    // Input-to-output dependencies, as the view-id state declares them:
    // every output scalar on every input scalar.
    if in_vectors > 0 && out_vectors > 0 {
        let scalars = |els: &[SigElement]| {
            els.iter()
                .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                .max()
                .unwrap_or(0)
        };
        let (ni, no) = (scalars(p.inputs), scalars(p.outputs));
        let per = (out_vectors * 4).div_ceil(32);
        for i in 0..in_vectors * 4 {
            for k in 0..per {
                let bits = if i < ni {
                    no.saturating_sub(32 * k).min(32)
                } else {
                    0
                };
                let mask = if bits == 32 {
                    u32::MAX
                } else {
                    (1u32 << bits) - 1
                };
                u32s(&mut out, &[mask]);
            }
        }
    }
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The container: parts in order, hashed.
pub(crate) fn build(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let table = 0x20 + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend(b"DXBC");
    out.extend([0; 16]);
    out.extend(1u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend((offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend(*fourcc);
        out.extend((d.len() as u32).to_le_bytes());
        out.extend(*d);
    }
    let digest = hash(&out[20..]);
    out[4..20].copy_from_slice(&digest);
    out
}

/// The container hash D3D12 checks: MD5 whose last block carries the bit
/// count first and `2 × bytes + 1` last instead of standard padding.
pub(crate) fn hash(data: &[u8]) -> [u8; 16] {
    let len = data.len();
    let left = len % 64;
    let mut tail = Vec::with_capacity(128);
    if left < 56 {
        tail.extend(((len as u32) << 3).to_le_bytes());
        tail.extend(&data[len - left..]);
        tail.push(0x80);
        tail.resize(60, 0);
    } else {
        tail.extend(&data[len - left..]);
        tail.push(0x80);
        tail.resize(64, 0);
        tail.extend(((len as u32) << 3).to_le_bytes());
        tail.resize(124, 0);
    }
    tail.extend(((len as u32) << 1 | 1).to_le_bytes());
    let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    for block in data[..len - left]
        .as_chunks::<64>()
        .0
        .iter()
        .chain(tail.as_chunks::<64>().0)
    {
        md5_block(&mut state, block);
    }
    let mut out = [0; 16];
    for (o, s) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        o.copy_from_slice(&s.to_le_bytes());
    }
    out
}

fn md5_block(state: &mut [u32; 4], block: &[u8; 64]) {
    const S: [u32; 16] = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    let x: Vec<u32> = block
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let [mut a, mut b, mut c, mut d] = *state;
    for i in 0..64 {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let k = ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32;
        let rotated = a
            .wrapping_add(f)
            .wrapping_add(k)
            .wrapping_add(x[g])
            .rotate_left(S[(i / 16) * 4 + i % 4]);
        (a, b, c, d) = (d, b.wrapping_add(rotated), b, c);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d]) {
        *s = s.wrapping_add(v);
    }
}

#[cfg(test)]
mod tests {
    use super::super::shader::Mesh;
    use super::*;

    fn u32_words(bytes: &[u8]) -> Vec<u32> {
        let (chunks, rem) = bytes.as_chunks::<4>();
        assert!(rem.is_empty());
        chunks.iter().copied().map(u32::from_le_bytes).collect()
    }

    fn sig_el(name: &str, row: Option<u32>, col: u8, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row,
            col,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        }
    }

    #[test]
    fn md5_core_matches_the_standard() {
        // Standard MD5 of "abc" with standard padding, through md5_block.
        let mut block = b"abc".to_vec();
        block.push(0x80);
        block.resize(56, 0);
        block.extend(24u64.to_le_bytes());
        let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
        md5_block(&mut state, block.as_slice().try_into().unwrap());
        let hex: String = state
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn retail_hash_matches_reference_padding_across_block_boundaries() {
        // Independent reference construction of the retail DXBC padding layout:
        // left < 56 puts bit_len at byte 0 of the single tail block; left >= 56
        // spills bit_len and `(len << 1) | 1` into a second 64-byte tail block.
        let ref_hash = |input: &[u8]| -> [u8; 16] {
            let n = input.len();
            let rem = n % 64;
            let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
            for full in input[..n - rem].as_chunks::<64>().0 {
                md5_block(&mut state, full);
            }
            let bits = ((n as u32) << 3).to_le_bytes();
            let count_tag = (((n as u32) << 1) | 1).to_le_bytes();
            if rem < 56 {
                let mut blk = [0u8; 64];
                blk[..4].copy_from_slice(&bits);
                blk[4..4 + rem].copy_from_slice(&input[n - rem..]);
                blk[4 + rem] = 0x80;
                blk[60..64].copy_from_slice(&count_tag);
                md5_block(&mut state, &blk);
            } else {
                let mut blk0 = [0u8; 64];
                blk0[..rem].copy_from_slice(&input[n - rem..]);
                blk0[rem] = 0x80;
                md5_block(&mut state, &blk0);
                let mut blk1 = [0u8; 64];
                blk1[..4].copy_from_slice(&bits);
                blk1[60..64].copy_from_slice(&count_tag);
                md5_block(&mut state, &blk1);
            }
            let mut out = [0u8; 16];
            for (dst, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            out
        };

        let sample: Vec<u8> = (0..130).map(|i| (i as u8).wrapping_mul(37)).collect();
        for len in [0, 1, 54, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128] {
            assert_eq!(hash(&sample[..len]), ref_hash(&sample[..len]), "len {len}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn retail_hash_matches_dxil_validator_for_all_64_remainders() {
        use super::super::validator::Validator;
        let Ok(mut validator) = Validator::load(Validator::default_path().as_deref()) else {
            return;
        };
        let mut s = super::super::shader::Shader::new(true);
        let entry = s.entry;
        s.m.terminate(entry, super::super::bitcode::Term::Ret);
        let base = s.finish();
        let parsed = crate::dxbc::Container::parse(&base).unwrap();

        for pad in 0..64usize {
            let priv_data = vec![0x5au8; pad];
            let mut parts: Vec<(&[u8; 4], &[u8])> =
                parsed.parts.iter().map(|p| (&p.fourcc.0, p.data)).collect();
            parts.push((b"PRIV", &priv_data));
            let built = build(&parts);
            let ours = built[4..20].to_vec();

            let mut unsigned = built;
            unsigned[4..20].fill(0);
            validator.validate(&mut unsigned).unwrap();
            assert_eq!(&unsigned[4..20], &ours[..], "padding length {pad}");
        }
    }

    #[test]
    fn container_header_and_chunk_offsets_handle_unaligned_parts() {
        let empty = build(&[]);
        assert_eq!(&empty[..4], b"DXBC");
        assert_eq!(u32_words(&empty[20..]), [1, 32, 0]);
        assert_eq!(&empty[4..20], &hash(&empty[20..]));

        let c = build(&[(b"SFI0", &[1, 2, 3]), (b"ISG1", &[]), (b"OSG1", &[9])]);
        assert_eq!(&c[..4], b"DXBC");
        assert_eq!(u32_words(&c[20..44]), [1, 72, 3, 44, 55, 63]);
        assert_eq!(&c[44..55], b"SFI0\x03\0\0\0\x01\x02\x03");
        assert_eq!(&c[55..63], b"ISG1\0\0\0\0");
        assert_eq!(&c[63..72], b"OSG1\x01\0\0\0\x09");
        assert_eq!(&c[4..20], &hash(&c[20..]));
    }

    #[test]
    fn signature_orders_rows_shares_system_names_and_encodes_masks() {
        assert_eq!(u32_words(&signature(&[], true)), [0, 8]);

        let els = [
            SigElement {
                index: 2,
                usage: 0b01,
                ..sig_el("TEXCOORD", Some(1), 2, 2)
            },
            SigElement {
                index: 1,
                kind: 16,
                ..sig_el("SV_Target", Some(2), 0, 4)
            },
            SigElement {
                index: 0,
                comp: 4,
                usage: 0b10,
                ..sig_el("TEXCOORD", Some(0), 0, 2)
            },
            SigElement {
                index: 0,
                kind: 16,
                ..sig_el("SV_Target", Some(1), 0, 4)
            },
            SigElement {
                kind: 17,
                ..sig_el("SV_Depth", None, 0, 1)
            },
        ];

        let isg = signature(&els, true);
        let osg = signature(&els, false);
        assert_eq!(u32_words(&isg[..8]), [5, 8]);
        assert_eq!(&isg[168..], b"TEXCOORD\0SV_Target\0TEXCOORD\0SV_Depth\0");

        // Sorted by (row.unwrap_or(MAX), col): (0,0), (1,0), (1,2), (2,0), (MAX,0).
        let expected_in = [
            [0, 187, 0, 0, 2, 0, 0x0203, 0],
            [0, 177, 0, 64, 3, 1, 0x0f0f, 0],
            [0, 168, 2, 0, 3, 1, 0x040c, 0],
            [0, 177, 1, 64, 3, 2, 0x0f0f, 0],
            [0, 196, 0, 65, 3, u32::MAX, 0x0101, 0],
        ];
        for (idx, row) in expected_in.into_iter().enumerate() {
            let at = 8 + idx * 32;
            assert_eq!(u32_words(&isg[at..at + 32]), row);
            let used = (row[6] >> 8) & 0xff;
            let mut out_row = row;
            out_row[6] = (row[6] & 0xff) | (((!used) & 0x0f) << 8);
            assert_eq!(u32_words(&osg[at..at + 32]), out_row);
        }
    }

    #[test]
    fn psv_encodes_compute_pixel_vertex_and_mesh_metadata() {
        let cs = psv(&Psv {
            stage: 5,
            flag: false,
            inputs: &[],
            outputs: &[],
            prims: &[],
            mesh: None,
            bindings: &[],
            threads: [64, 2, 1],
            wave_size: Some(64),
        });
        assert_eq!(
            u32_words(&cs),
            [48, 0, 0, 0, 0, 64, 64, 5, 0, 0, 64, 2, 1, 0, 4, 0, 0]
        );

        // Pixel sample-frequency bit activates for sample interpolation (6/7) or SV_SampleIndex (12).
        for (interp, kind, sample_freq) in [(2, 0, 0u8), (6, 0, 1), (7, 0, 1), (0, 12, 1)] {
            let inp = [SigElement {
                interp,
                kind,
                ..sig_el("ATTR", Some(0), 0, 2)
            }];
            let ps = psv(&Psv {
                stage: 0,
                flag: true,
                inputs: &inp,
                outputs: &[],
                prims: &[],
                mesh: None,
                bindings: &[],
                threads: [0; 3],
                wave_size: None,
            });
            assert_eq!(&ps[4..6], &[1, sample_freq]);
            assert_eq!(u32_words(&ps[20..28]), [0, u32::MAX]);
        }

        // Vertex with >8 output rows spans two 32-bit words per input scalar in the dependency mask.
        let vs_in = [sig_el("POS", Some(0), 0, 2)];
        let vs_out = [sig_el("OUT", Some(8), 0, 2)];
        let vs = psv(&Psv {
            stage: 1,
            flag: true,
            inputs: &vs_in,
            outputs: &vs_out,
            prims: &[],
            mesh: None,
            bindings: &[Binding {
                kind: 2,
                space: 0,
                register: 0,
                dxil_kind: 13,
            }],
            threads: [0; 3],
            wave_size: None,
        });
        // 1 input vector * 4 scalars * 2 mask dwords = 8 trailing u32s: first 2 active, next 2 zero.
        let tail = u32_words(&vs[vs.len() - 32..]);
        assert_eq!(tail, [u32::MAX, 0b11, u32::MAX, 0b11, 0, 0, 0, 0]);

        let ms_out = [SigElement {
            kind: 3,
            ..sig_el("SV_Position", Some(0), 0, 4)
        }];
        let ms_prim = [SigElement {
            kind: 30,
            comp: 1,
            ..sig_el("SV_CullPrimitive", Some(0), 0, 1)
        }];
        let ms = psv(&Psv {
            stage: 13,
            flag: false,
            inputs: &[],
            outputs: &ms_out,
            prims: &ms_prim,
            mesh: Some((
                Mesh {
                    max_vertices: 96,
                    max_primitives: 64,
                    topology: 2,
                },
                2048,
            )),
            bindings: &[],
            threads: [128, 1, 1],
            wave_size: None,
        });
        assert_eq!(u32_words(&ms[..20]), [48, 2048, 0, 0, (64 << 16) | 96]);
        assert_eq!(&ms[28..35], &[13, 0, 1, 2, 0, 1, 1]);
    }
}
