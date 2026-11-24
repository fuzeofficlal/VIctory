//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the VIctory build
//! that translated it (the BLAKE3 of our executable), so a rebuilt
//! translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{
    HostShader, MeshFallback, MeshLayout, StaticSampler, TranslateOptions, UserData, Wave64Mode,
};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static WAVE64_MODE: AtomicU8 = AtomicU8::new(0);

fn mode_byte(mode: Wave64Mode) -> u8 {
    match mode {
        Wave64Mode::Native => 0,
        Wave64Mode::Wave32 => 1,
        Wave64Mode::Emulate => 2,
    }
}

/// Records the host device's preferred Wave64 compute lowering strategy.
#[cfg(windows)]
pub(crate) fn set_wave64_mode(mode: Wave64Mode) {
    WAVE64_MODE.store(mode_byte(mode), Ordering::Relaxed);
}

fn effective_wave64_mode() -> Wave64Mode {
    if let Ok(v) = std::env::var("VICTORY_WAVE64_EMULATE") {
        match v.trim().to_ascii_lowercase().as_str() {
            "0" | "native" => return Wave64Mode::Native,
            "wave32" | "32" => return Wave64Mode::Wave32,
            "1" | "emulate" | "force" => return Wave64Mode::Emulate,
            "" | "auto" => {}
            other => tracing::warn!(value = other, "unknown VICTORY_WAVE64_EMULATE; using auto"),
        }
    }
    match WAVE64_MODE.load(Ordering::Relaxed) {
        1 => Wave64Mode::Wave32,
        2 => Wave64Mode::Emulate,
        _ => Wave64Mode::Native,
    }
}

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies this build: the BLAKE3 of our executable file and of
/// the translation settings it runs with (the loop limit, which a
/// diagnostics build can change at run time).
fn build() -> &'static [u8; 32] {
    static BUILD: OnceLock<[u8; 32]> = OnceLock::new();
    BUILD.get_or_init(|| {
        let exe = std::env::current_exe().and_then(std::fs::read);
        let limit = victory_shader::emit::loop_limit();
        if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
            tracing::warn!(
                limit,
                "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"
            );
        }
        match exe {
            Ok(mut bytes) => {
                bytes.extend_from_slice(&limit.to_le_bytes());
                *blake3::hash(&bytes).as_bytes()
            }
            Err(e) => {
                // Without it no entry can be trusted: a fresh key per run.
                tracing::warn!(error = %e, "cannot read our executable; shader cache unused");
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                *blake3::hash(&now.as_nanos().to_le_bytes()).as_bytes()
            }
        }
    })
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    let wave64 = effective_wave64_mode();
    let options = TranslateOptions { wave64 };
    let translate =
        |c: &[u8]| victory_shader::emit::translate_with(c, options).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => cached(cache, build(), wave64, container, translate),
        None => translate(container),
    }
}

/// A container's mesh program run as compute (hosts without mesh shaders,
/// issue #3), from the cache when there is one.
///
/// Threads `effective_wave64_mode()` into
/// [`translate_mesh_as_compute_with`](victory_shader::emit::translate_mesh_as_compute_with)
/// and the 33-byte `[build, Wave64Mode, container]` cache prefix. `StageIo::Ms`
/// lowering currently ignores `options.wave64` until mesh wave64 LDS emulation
/// (`#189` follow-on) is integrated, while the `Wave64Mode` byte keeps disk
/// cache keys forward-compatible and consistent with `ms.translated()`.
pub(crate) fn mesh_fallback(container: &[u8]) -> Result<MeshFallback, String> {
    let wave64 = effective_wave64_mode();
    let options = TranslateOptions { wave64 };
    let translate = |c: &[u8]| {
        victory_shader::emit::translate_mesh_as_compute_with(c, options).map_err(|e| e.to_string())
    };
    match CACHE.get() {
        Some(cache) => cached_as(cache, build(), wave64, container, &MESH_FALLBACK, translate),
        None => translate(container),
    }
}

/// How one kind of entry is named and written.
struct Kind<T> {
    id: &'static str,
    encode: fn(&T) -> Vec<u8>,
    decode: fn(&[u8]) -> Option<T>,
}

const SHADER: Kind<HostShader> = Kind {
    id: "shader.dxil",
    encode,
    decode,
};

const MESH_FALLBACK: Kind<MeshFallback> = Kind {
    id: "shader.mesh-fallback",
    encode: encode_fallback,
    decode: decode_fallback,
};

/// The cache entry's converter: the build, the lowering mode and the
/// container in, the encoded translation out. `ran` tells a miss from a hit.
struct Translate<'k, F, T> {
    translate: F,
    kind: &'k Kind<T>,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<T, String> + Send + Sync, T> Converter for Translate<'_, F, T> {
    fn id(&self) -> &'static str {
        self.kind.id
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let value = (self.translate)(&input[33..]).map_err(anyhow::Error::msg)?;
        Ok((self.kind.encode)(&value))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    mode: Wave64Mode,
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    cached_as(cache, build, mode, container, &SHADER, translate)
}

fn cached_as<T>(
    cache: &Cache,
    build: &[u8; 32],
    mode: Wave64Mode,
    container: &[u8],
    kind: &Kind<T>,
    translate: impl Fn(&[u8]) -> Result<T, String> + Send + Sync,
) -> Result<T, String> {
    let mb = [mode_byte(mode)];
    let input = [&build[..], &mb[..], container].concat();
    let converter = Translate {
        translate,
        kind,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| (kind.decode)(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
    })
}

const FALLBACK_MAGIC: &[u8; 4] = b"VMF1";

/// The compute and vertex shaders as [`encode`]s, length-prefixed, then
/// `expanded` and the layout's words.
fn encode_fallback(f: &MeshFallback) -> Vec<u8> {
    let mut out = FALLBACK_MAGIC.to_vec();
    for s in [&f.compute, &f.vertex] {
        let bytes = encode(s);
        out.extend((bytes.len() as u32).to_le_bytes());
        out.extend(bytes);
    }
    let l = &f.layout;
    for v in [
        u32::from(f.expanded),
        l.threads,
        l.max_vertices,
        l.max_primitives,
        l.vertex_dwords,
        l.primitive_dwords,
    ] {
        out.extend(v.to_le_bytes());
    }
    out
}

fn decode_fallback(bytes: &[u8]) -> Option<MeshFallback> {
    let mut rest = bytes.strip_prefix(FALLBACK_MAGIC)?;
    let mut take = |n: usize| -> Option<&[u8]> {
        let (head, tail) = rest.split_at_checked(n)?;
        rest = tail;
        Some(head)
    };
    let word = |b: &[u8]| -> Option<u32> { Some(u32::from_le_bytes(b.try_into().ok()?)) };
    let n = word(take(4)?)? as usize;
    let compute = decode(take(n)?)?;
    let n = word(take(4)?)? as usize;
    let vertex = decode(take(n)?)?;
    let mut words = [0u32; 6];
    for w in &mut words {
        *w = word(take(4)?)?;
    }
    let [
        expanded,
        threads,
        max_vertices,
        max_primitives,
        vertex_dwords,
        primitive_dwords,
    ] = words;
    (rest.is_empty() && expanded <= 1).then_some(MeshFallback {
        compute,
        vertex,
        expanded: expanded == 1,
        layout: MeshLayout {
            threads,
            max_vertices,
            max_primitives,
            vertex_dwords,
            primitive_dwords,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            cached(&cache, &build, Wave64Mode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(
            cached(&cache, &build, Wave64Mode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another Wave64 mode translates separately.
        assert_eq!(
            cached(&cache, &build, Wave64Mode::Wave32, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // Another build translates again.
        assert_eq!(
            cached(&cache, &[8; 32], Wave64Mode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(
            cached(&cache, &build, Wave64Mode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 4);
        assert_eq!(
            cached(&cache, &build, Wave64Mode::Native, container, translate),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 4);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, Wave64Mode::Native, b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    fn fallback() -> MeshFallback {
        let mut vertex = shader();
        vertex.stage = HwStage::Vs;
        MeshFallback {
            compute: HostShader {
                stage: HwStage::Ms,
                ..shader()
            },
            vertex,
            expanded: true,
            layout: MeshLayout {
                threads: 64,
                max_vertices: 64,
                max_primitives: 126,
                vertex_dwords: 12,
                primitive_dwords: 4,
            },
        }
    }

    #[test]
    fn mesh_fallbacks_come_back_from_the_cache_apart_from_translations() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(fallback())
        };
        let get = || {
            cached_as(
                &cache,
                &build,
                Wave64Mode::Native,
                container,
                &MESH_FALLBACK,
                translate,
            )
        };
        assert_eq!(get(), Ok(fallback()));
        assert_eq!(get(), Ok(fallback()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");
        // The same container's plain translation is an entry of its own.
        assert_eq!(
            cached(&cache, &build, Wave64Mode::Native, container, |_| Ok(
                shader()
            )),
            Ok(shader())
        );
        assert_eq!(get(), Ok(fallback()));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn mesh_fallbacks_isolate_wave64_modes_and_preserve_container_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console mesh container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |c: &[u8]| {
            assert_eq!(c, container, "33-byte cache prefix must preserve container");
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(fallback())
        };
        for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
            assert_eq!(
                cached_as(&cache, &build, mode, container, &MESH_FALLBACK, translate),
                Ok(fallback())
            );
        }
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        for mode in [Wave64Mode::Native, Wave64Mode::Wave32, Wave64Mode::Emulate] {
            assert_eq!(
                cached_as(&cache, &build, mode, container, &MESH_FALLBACK, translate),
                Ok(fallback())
            );
        }
        assert_eq!(
            runs.load(Ordering::Relaxed),
            3,
            "repeat lookups per Wave64Mode must hit cache"
        );
    }

    #[test]
    fn mesh_fallback_entries_round_trip_and_reject_damage() {
        let bytes = encode_fallback(&fallback());
        assert_eq!(decode_fallback(&bytes), Some(fallback()));
        assert_eq!(decode_fallback(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode_fallback(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode_fallback(&encode(&shader())), None);
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
