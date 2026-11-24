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
//!
//! Containers that fail translation are also memoized in memory for the
//! current process and stored on disk as `VSF1` entries under the same
//! build-keyed input. Rebuilding the translator or changing
//! `VICTORY_SHADER_LOOP_LIMIT` changes [`build`], so fixed shaders are
//! automatically retried while repeated launches of the same build skip
//! re-translating known failures.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static FAILURES: Mutex<Option<HashMap<[u8; 32], String>>> = Mutex::new(None);
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

fn failure_key(build: &[u8; 32], container: &[u8]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(build);
    h.update(container);
    *h.finalize().as_bytes()
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    let translate = |c: &[u8]| victory_shader::emit::translate(c).map_err(|e| e.to_string());
    lookup(&FAILURES, CACHE.get(), build(), container, translate)
}

fn lookup(
    failures: &Mutex<Option<HashMap<[u8; 32], String>>>,
    cache: Option<&Cache>,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let key = failure_key(build, container);
    if let Some(err) = failures
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|map| map.get(&key).cloned())
    {
        return Err(err);
    }
    let result = match cache {
        Some(cache) => cached(cache, build, container, translate),
        None => translate(container),
    };
    if let Err(err) = &result {
        failures
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get_or_insert_with(HashMap::new)
            .insert(key, err.clone());
    }
    result
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation or failure entry out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        Ok(match (self.translate)(&input[32..]) {
            Ok(shader) => encode(&shader),
            Err(error) => encode_failure(&error),
        })
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        converter.ran.store(false, Ordering::Relaxed);
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        let entry = std::fs::read(&path)
            .ok()
            .and_then(|b| decode(&b).map(Ok).or_else(|| decode_failure(&b).map(Err)));
        if let Some(result) = entry {
            let ran = converter.ran.load(Ordering::Relaxed);
            if result.is_ok() {
                let counter = if ran { &MISSES } else { &HITS };
                counter.fetch_add(1, Ordering::Relaxed);
            }
            let what = if ran {
                "translated and cached"
            } else {
                "from the cache"
            };
            tracing::debug!(path = %path.display(), ok = result.is_ok(), "shader {what}");
            return result;
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    let result = (converter.translate)(container);
    if result.is_ok() {
        MISSES.fetch_add(1, Ordering::Relaxed);
    }
    result
}

const MAGIC: &[u8; 4] = b"VSH1";
const FAILED_MAGIC: &[u8; 4] = b"VSF1";

fn encode_failure(error: &str) -> Vec<u8> {
    let bytes = error.as_bytes();
    let mut out = Vec::with_capacity(8 + bytes.len());
    out.extend_from_slice(FAILED_MAGIC);
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
    out
}

fn decode_failure(bytes: &[u8]) -> Option<String> {
    let rest = bytes.strip_prefix(FAILED_MAGIC.as_slice())?;
    let (len, text) = rest.split_at_checked(4)?;
    let len = u32::from_le_bytes(len.try_into().ok()?) as usize;
    if text.len() != len {
        return None;
    }
    String::from_utf8(text.to_vec()).ok()
}
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
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn in_memory_negative_cache_skips_retranslating_without_disk_cache() {
        let failures = Mutex::new(None);
        let build = [3u8; 32];
        let container = b"synthetic unsupported container".as_slice();
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported instruction".to_owned())
        };

        assert_eq!(
            lookup(&failures, None, &build, container, fail),
            Err("unsupported instruction".to_owned())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // A second lookup with the same build and container hits the in-memory
        // negative cache and never invokes the translator again.
        assert_eq!(
            lookup(&failures, None, &build, container, fail),
            Err("unsupported instruction".to_owned())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // A new build fingerprint is not blocked by the old build's failure.
        let ok = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(
            lookup(&failures, None, &[4u8; 32], container, ok),
            Ok(shader())
        );
        assert_eq!(runs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn failures_persist_per_build_without_inflating_shader_counts() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"synthetic container that refuses translation".as_slice();
        let runs = AtomicU64::new(0);
        let fail = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Err::<HostShader, _>("unsupported atomic".to_owned())
        };
        let before = counts();
        let expected = Err("unsupported atomic".to_owned());

        assert_eq!(cached(&cache, &build, container, fail), expected);
        assert_eq!(cached(&cache, &build, container, fail), expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert_eq!(
            counts(),
            before,
            "refusals must not inflate shader hit/miss counts"
        );

        // Another build retries translation and succeeds once supported.
        let ok = |_: &[u8]| Ok(shader());
        assert_eq!(cached(&cache, &[8u8; 32], container, ok), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, fail), expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        // A corrupted failure entry on disk is discarded and retried.
        for path in walk(dir.path()) {
            if std::fs::read(&path).unwrap().starts_with(FAILED_MAGIC) {
                std::fs::write(path, b"VSF1\x10\0\0\0short").unwrap();
            }
        }
        assert_eq!(cached(&cache, &build, container, fail), expected);
        assert_eq!(runs.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn failure_entries_round_trip_and_reject_damage() {
        let full = encode_failure("unsupported resource binding");
        assert_eq!(
            decode_failure(&full).as_deref(),
            Some("unsupported resource binding")
        );
        assert_eq!(decode(&full), None);
        assert_eq!(decode_failure(&encode(&shader())), None);

        for cut in [
            0,
            FAILED_MAGIC.len(),
            FAILED_MAGIC.len() + 4,
            full.len() - 1,
        ] {
            assert_eq!(decode_failure(&full[..cut]), None, "truncated at {cut}");
        }
        let mut trailing = full.clone();
        trailing.push(b'!');
        assert_eq!(decode_failure(&trailing), None);
        assert_eq!(
            decode_failure(b"VSF1\x02\0\0\0\xff\xfe"),
            None,
            "invalid utf-8"
        );
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
    }
}
