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
use std::sync::atomic::{AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

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
    let translate = |c: &[u8]| victory_shader::emit::translate(c).map_err(|e| e.to_string());
    match CACHE.get() {
        Some(cache) => cached(cache, build(), container, translate),
        None => translate(container),
    }
}

const BUILD_LEN: usize = 32;
const VERSION: u32 = 2;

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `translated` keeps the in-memory result so a
/// miss neither re-translates nor fails when disk publication cannot write.
struct Translate<F> {
    translate: F,
    translated: OnceLock<HostShader>,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Translate<F> {
    fn run(&self, container: &[u8]) -> Result<&HostShader, String> {
        if let Some(shader) = self.translated.get() {
            return Ok(shader);
        }
        let shader = (self.translate)(container)?;
        Ok(self.translated.get_or_init(|| shader))
    }
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        VERSION
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        let (_, container) = input
            .split_at_checked(BUILD_LEN)
            .ok_or_else(|| anyhow::anyhow!("shader cache input shorter than {BUILD_LEN} bytes"))?;
        let shader = self.run(container).map_err(anyhow::Error::msg)?;
        Ok(encode(shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; BUILD_LEN],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let mut converter = Translate {
        translate,
        translated: OnceLock::new(),
    };
    for _ in 0..2 {
        let path = match cache.get_or_convert(&converter, &input) {
            Ok(path) => path,
            Err(e) => {
                if let Some(shader) = converter.translated.take() {
                    MISSES.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(error = %e, "shader cache write failed; using uncached shader");
                    return Ok(shader);
                }
                return Err(format!("{e:#}"));
            }
        };
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.translated.get().is_some() {
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
    match converter.translated.take() {
        Some(shader) => Ok(shader),
        None => (converter.translate)(container),
    }
}

const MAGIC: &[u8; 4] = b"VSH2";
const OTHER_ROOT: u32 = 6;
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
        RootType::Other(_) => OTHER_ROOT,
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
        for v in [u.index, u32::from(u.slot), root_code(u.root)] {
            word(&mut out, v);
        }
        if let RootType::Other(x) = u.root {
            word(&mut out, x);
        }
        word(&mut out, u.dword);
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
        let index = r.word()?;
        let slot = u8::try_from(r.word()?).ok()?;
        let root = match r.word()? {
            0 => RootType::Constants,
            1 => RootType::Descriptor,
            2 => RootType::Table,
            3 => RootType::ImmediateSampler,
            4 => RootType::ResourceHeap,
            5 => RootType::SamplerHeap,
            OTHER_ROOT => RootType::Other(r.word()?),
            _ => return None,
        };
        user_data.push(UserData {
            index,
            slot,
            root,
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
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH2 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
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

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
        assert_eq!(decode(b"VSH1"), None);
    }

    #[test]
    fn all_stages_and_root_types_round_trip_losslessly() {
        let roots = [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(0),
            RootType::Other(1),
            RootType::Other(6),
            RootType::Other(u32::MAX - 5),
            RootType::Other(u32::MAX),
        ];
        for stage in STAGES {
            let mut s = shader();
            s.stage = stage;
            s.user_data = roots
                .iter()
                .enumerate()
                .map(|(i, &root)| UserData {
                    index: i as u32,
                    slot: i as u8,
                    root,
                    dword: 100 + i as u32,
                })
                .collect();
            let bytes = encode(&s);
            assert_eq!(decode(&bytes), Some(s));
        }
    }

    #[test]
    fn decode_rejects_unknown_root_tags_and_truncated_other_payload() {
        let mut s = shader();
        s.user_data = vec![UserData {
            index: 0,
            slot: 0,
            root: RootType::Constants,
            dword: 7,
        }];
        let base = encode(&s);
        // Header: 4 (MAGIC) + 4 (stage) + 4 (dxil len) + dxil.len() + 4 (user_data len)
        // + 4 (index) + 4 (slot) => root tag starts at 24 + dxil.len().
        let tag_at = 24 + s.dxil.len();
        for bad_tag in [7u32, 42, u32::MAX] {
            let mut bad = base.clone();
            bad[tag_at..tag_at + 4].copy_from_slice(&bad_tag.to_le_bytes());
            assert_eq!(decode(&bad), None, "tag {bad_tag} must be rejected");
        }

        // Tag OTHER_ROOT (6) consumes one extra word, leaving `dword` missing.
        let mut truncated_other = base;
        truncated_other[tag_at..tag_at + 4].copy_from_slice(&OTHER_ROOT.to_le_bytes());
        assert_eq!(decode(&truncated_other), None);
    }

    #[test]
    fn unwritable_cache_keeps_in_memory_translation_without_retranslating() {
        use victory_pipeline::CacheKey;

        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [9u8; BUILD_LEN];
        let container = b"synthetic container a".as_slice();
        let probe = Translate {
            translate: |_: &[u8]| Ok(shader()),
            translated: OnceLock::new(),
        };

        // Case 1: target cache entry path is blocked by an existing directory, so
        // staging succeeds after `convert` and atomic rename fails.
        let input_a = [&build[..], container].concat();
        let blocked_entry = cache.path_for(&CacheKey::new(&probe, &input_a));
        std::fs::create_dir_all(&blocked_entry).unwrap();

        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(
            runs.load(Ordering::Relaxed),
            1,
            "failed rename must reuse the already-translated shader"
        );

        // Case 2: shard parent path is blocked by a regular file, so `create_dir_all`
        // fails after `convert` has already produced the shader.
        let container_b = b"synthetic container b".as_slice();
        let input_b = [&build[..], container_b].concat();
        let entry_b = cache.path_for(&CacheKey::new(&probe, &input_b));
        let shard_parent = entry_b.parent().unwrap();
        if shard_parent.exists() {
            std::fs::remove_dir_all(shard_parent).unwrap();
        }
        std::fs::write(shard_parent, b"not a directory").unwrap();

        runs.store(0, Ordering::Relaxed);
        assert_eq!(cached(&cache, &build, container_b, translate), Ok(shader()));
        assert_eq!(
            runs.load(Ordering::Relaxed),
            1,
            "failed directory creation must reuse the already-translated shader"
        );

        // Translation errors still propagate when `convert` itself failed.
        let fail = |_: &[u8]| Err::<HostShader, _>("bad shader".to_owned());
        let err = cached(&cache, &build, b"failing container", fail).unwrap_err();
        assert!(err.contains("bad shader"), "{err}");
    }

    #[test]
    fn convert_rejects_inputs_shorter_than_build_fingerprint() {
        let runs = AtomicU64::new(0);
        let seen_len = AtomicU64::new(u64::MAX);
        let conv = Translate {
            translate: |c: &[u8]| {
                runs.fetch_add(1, Ordering::Relaxed);
                seen_len.store(c.len() as u64, Ordering::Relaxed);
                Ok(shader())
            },
            translated: OnceLock::new(),
        };

        for len in [0usize, 1, 15, BUILD_LEN - 1] {
            let short = vec![0xAB; len];
            assert!(conv.convert(&short).is_err(), "len {len} should fail");
        }
        assert_eq!(runs.load(Ordering::Relaxed), 0);

        // Exact 32-byte input has an empty container slice and does not panic.
        let exact = [0xCD; BUILD_LEN];
        let encoded = conv.convert(&exact).unwrap();
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(seen_len.load(Ordering::Relaxed), 0);
        assert_eq!(decode(&encoded), Some(shader()));
    }
}
