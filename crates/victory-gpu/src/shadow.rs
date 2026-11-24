//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! in GPU-local memory (`D3D12_HEAP_TYPE_DEFAULT`) on first use, and the
//! ranges each submission reads (vertex, index and constant data, buffer
//! views) are brought up to date from title memory before it runs via pooled
//! `UPLOAD` staging copies: the 64 KiB pages the title wrote since they were
//! last copied ([`crate::watch`]: write watching, or read-only guards on
//! physical-page views), all of it where neither works. A title GPU address
//! inside the buffer is the same offset in its copy. Ranges shaders write
//! are read back through pooled `READBACK` staging and merged into title
//! memory after the submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// One staged upload copy to record onto the submission's upload list.
pub(crate) struct UploadCopy {
    pub(crate) shadow: Arc<Shadow>,
    pub(crate) src: ID3D12Resource,
    pub(crate) src_offset: u64,
    pub(crate) dst_offset: u64,
    pub(crate) len: u64,
}

/// A title buffer's host copy in `D3D12_HEAP_TYPE_DEFAULT` memory.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; mutable state is mutex-guarded.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

/// Coalesces overlapping or adjacent `[from, to)` byte ranges in place.
pub(crate) fn merge_ranges(ranges: &mut Vec<(usize, usize)>) {
    ranges.retain(|&(a, b)| a < b);
    if ranges.len() <= 1 {
        return;
    }
    ranges.sort_unstable();
    let mut out = 0;
    for i in 1..ranges.len() {
        let (a, b) = ranges[i];
        if a <= ranges[out].1 {
            ranges[out].1 = ranges[out].1.max(b);
        } else {
            out += 1;
            ranges[out] = (a, b);
        }
    }
    ranges.truncate(out + 1);
}

/// Subtracts sorted, disjoint `covered` intervals from `[start, end)`,
/// returning the remaining uncovered sub-ranges in ascending order.
fn subtract_covered(start: u64, end: u64, covered: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut kept = Vec::new();
    let mut cur = start;
    for &(c0, c1) in covered {
        if c1 <= cur {
            continue;
        }
        if c0 >= end {
            break;
        }
        if c0 > cur {
            kept.push((cur, c0));
        }
        cur = cur.max(c1);
        if cur >= end {
            break;
        }
    }
    if cur < end {
        kept.push((cur, end));
    }
    kept
}

/// Merges `[start, end)` into a sorted, disjoint interval list in place.
fn insert_covered(covered: &mut Vec<(u64, u64)>, start: u64, end: u64) {
    if start >= end {
        return;
    }
    covered.push((start, end));
    covered.sort_unstable();
    let mut out = 0;
    for i in 1..covered.len() {
        let (a, b) = covered[i];
        if a <= covered[out].1 {
            covered[out].1 = covered[out].1.max(b);
        } else {
            out += 1;
            covered[out] = (a, b);
        }
    }
    covered.truncate(out + 1);
}

/// Trims or drops earlier [`UploadCopy`] ranges that are overwritten by a later
/// upload to the same shadow buffer, ensuring every `CopyBufferRegion` recorded
/// into one upload prelude writes strictly disjoint destination bytes (last
/// write wins, matching the previous CPU `memcpy` order without extra barriers).
pub(crate) fn trim_overlapping_uploads(uploads: &mut Vec<UploadCopy>) {
    if uploads.len() <= 1 {
        return;
    }
    let mut covered: std::collections::HashMap<usize, Vec<(u64, u64)>> =
        std::collections::HashMap::new();
    let mut kept = Vec::with_capacity(uploads.len());
    for u in uploads.drain(..).rev() {
        if u.len == 0 {
            continue;
        }
        let dst_start = u.dst_offset;
        let dst_end = dst_start.saturating_add(u.len);
        let entry = covered.entry(Arc::as_ptr(&u.shadow) as usize).or_default();
        let pieces = subtract_covered(dst_start, dst_end, entry);
        for &(s, e) in pieces.iter().rev() {
            kept.push(UploadCopy {
                shadow: u.shadow.clone(),
                src: u.src.clone(),
                src_offset: u.src_offset + (s - dst_start),
                dst_offset: s,
                len: e - s,
            });
        }
        if !pieces.is_empty() {
            insert_covered(entry, dst_start, dst_end);
        }
    }
    kept.reverse();
    *uploads = kept;
}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    pub(crate) fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (
            from - start,
            (address.saturating_add(len)).clamp(from, end) - start,
        )
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: records that a submitted
    /// list's shaders may write `[address, address + len)` so later refreshes
    /// in the same submission do not overwrite it.
    pub(crate) fn mark_pending(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address.saturating_add(len)));
    }

    /// After the submission completes and readbacks are merged: clears all
    /// pending shader-write ranges.
    pub(crate) fn clear_pending(&self) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// After the submission ran: compares the readback snapshots taken before
    /// (`old`) and after (`after`) the submission for buffer offset range
    /// `[from, from + old.len())`, and copies only the bytes the GPU modified
    /// back into title memory (unmapped pages left alone). Preserves any
    /// concurrent CPU writes to other bytes on the same page; pages without
    /// concurrent CPU writes are marked current so our own write-back is not
    /// re-uploaded on the next submission. Returns the bytes written.
    pub(crate) fn copy_back_range(&self, from: usize, old: &[u8], after: &[u8]) -> usize {
        debug_assert_eq!(old.len(), after.len());
        let len = old.len().min(after.len());
        if len == 0 {
            return 0;
        }
        let start = self.start as usize;
        let to = from + len;
        let mut written = 0;
        // By 64 KiB piece: compare after vs before, touching title memory
        // only where the GPU actually wrote.
        const PIECE: usize = 64 << 10;
        let mut title = vec![0u8; PIECE];
        let mut synced = Vec::new();
        let mut at = from;
        while at < to {
            let n = (PIECE - (start + at) % PIECE).min(to - at);
            let rel = at - from;
            let host = &after[rel..rel + n];
            let prev = &old[rel..rel + n];
            let title = &mut title[..n];
            if host != prev && crate::host::read_into((start + at) as *const u8, title) {
                let mut changed = false;
                let mut cpu_diff = false;
                for ((t, &h), &o) in title.iter_mut().zip(host).zip(prev) {
                    if h != o {
                        if *t != h {
                            *t = h;
                            changed = true;
                        }
                    } else if *t != h {
                        cpu_diff = true;
                    }
                }
                if changed {
                    crate::watch::writing(start + at, n);
                    crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                    written += n;
                }
                // Only mark a whole page synced if title memory and VRAM now match
                // byte for byte; partial edge pieces or pages with concurrent CPU
                // writes stay dirty so the next refresh uploads them.
                if !cpu_diff && n == PAGE {
                    synced.push((start + at) / PAGE);
                }
            }
            at += n;
        }
        if !synced.is_empty() && self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            let base = start / PAGE;
            let mut i = 0;
            while i < synced.len() {
                let run_first = synced[i];
                let mut run_last = run_first + 1;
                while i + 1 < synced.len() && synced[i + 1] == run_last {
                    i += 1;
                    run_last += 1;
                }
                i += 1;
                if let Some((now, _)) =
                    crate::watch::writes(run_first * PAGE, run_last * PAGE, true)
                {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    for page in run_first..run_last {
                        copied[page - base] = now;
                    }
                }
            }
        }
        written
    }

    /// At submission: stages what changed of `[address, address + len)` since
    /// it was last copied (all of it if the memory is not write-watched) into
    /// upload memory allocated via `alloc`, pushing GPU copy descriptors onto
    /// `out`. Returns the bytes staged.
    pub(crate) fn update(
        self: &Arc<Self>,
        address: usize,
        len: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a, alloc, out);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to, alloc, out);
            }
            return bytes;
        }
        self.update_free(from, to, alloc, out)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written, alloc, out);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.stage(from, to, alloc, out).unwrap_or(0)
    }

    /// Stages the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        // Mark pages current only after their run's staging allocation succeeds;
        // if `alloc` returns `None`, leave `copied` unchanged so the next
        // submission retries them.
        let mut flush_run = |first: usize, end_page: usize, copied: &mut [u64]| -> usize {
            let Some(staged) = self.stage(
                (first * PAGE).max(from),
                (end_page * PAGE).min(to),
                alloc,
                out,
            ) else {
                return 0;
            };
            for p in first..end_page {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                if (p * PAGE).max(start) >= from && ((p + 1) * PAGE).min(end) <= to {
                    copied[p - base] = now;
                }
            }
            staged
        };
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = copied[page - base];
            if at == 0 || write > at {
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += flush_run(first, page, &mut copied);
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += flush_run(first, last + 1, &mut copied);
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into upload
    /// staging memory and records the GPU copy into this default-heap buffer.
    /// Returns `None` if staging allocation fails.
    fn stage(
        self: &Arc<Self>,
        from: usize,
        to: usize,
        alloc: &mut impl FnMut(usize) -> Option<(ID3D12Resource, u64, *mut u8)>,
        out: &mut Vec<UploadCopy>,
    ) -> Option<usize> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        let len = to.saturating_sub(from);
        if len == 0 {
            return Some(0);
        }
        let (src, src_offset, dst_ptr) = alloc(len)?;
        crate::host::copy_mapped(dst_ptr, from as *const u8, len, false);
        out.push(UploadCopy {
            shadow: self.clone(),
            src,
            src_offset,
            dst_offset: (from - start) as u64,
            len: len as u64,
        });
        Some(len)
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // GPU-local default heap so shaders read/write VRAM at full bandwidth.
        let buffer = host()?.default_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory. Such
/// memory gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let buffer = host()?.default_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::{D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    struct WatchedAlloc {
        ptr: *mut u8,
        len: usize,
        resource_id: usize,
    }

    impl WatchedAlloc {
        fn new(len: usize, resource_id: usize) -> Self {
            // SAFETY: fresh write-watched virtual allocation, freed in Drop.
            let ptr = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            }
            .cast::<u8>();
            assert!(!ptr.is_null());
            register(resource_id, ptr as u64, len as u64);
            Self {
                ptr,
                len,
                resource_id,
            }
        }

        fn addr(&self) -> u64 {
            self.ptr as u64
        }

        fn bytes_mut(&mut self) -> &mut [u8] {
            // SAFETY: exclusive access to our committed allocation.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    impl Drop for WatchedAlloc {
        fn drop(&mut self) {
            unregister(self.resource_id, self.addr());
            // Clear watch journal and guard entries while we still own the
            // virtual address range, before another thread can reuse it.
            crate::watch::forget_pages(
                self.ptr as usize,
                (self.ptr as usize).saturating_add(self.len),
            );
            // SAFETY: frees the allocation obtained from VirtualAlloc.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn merge_ranges_coalesces_overlapping_and_adjacent() {
        let mut empty = Vec::new();
        merge_ranges(&mut empty);
        assert!(empty.is_empty());

        let mut ranges = vec![
            (64, 64),
            (200, 100),
            (128, 256),
            (0, 64),
            (64, 96),
            (96, 128),
            (512, 600),
            (550, 700),
        ];
        merge_ranges(&mut ranges);
        assert_eq!(ranges, vec![(0, 256), (512, 700)]);
    }

    #[test]
    fn default_vram_shadow_uploads_only_dirty_pages() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut mem = WatchedAlloc::new(2 * PAGE, 0x5000_0001);
        // Touch every 4 KiB page of the allocation before initial submission so
        // Windows demand-zero page faults don't fire during the first read.
        mem.bytes_mut().fill(0);
        mem.bytes_mut()[0..4].copy_from_slice(&0x1122_3344u32.to_le_bytes());
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0x5566_7788u32.to_le_bytes());

        let shadow = find(mem.addr(), (2 * PAGE) as u64).expect("default shadow");

        // Verify the shadow buffer lives in GPU-local DEFAULT VRAM.
        let mut heap = D3D12_HEAP_PROPERTIES::default();
        // SAFETY: live D3D12 resource.
        unsafe { shadow.buffer.GetHeapProperties(Some(&mut heap), None) }.expect("heap props");
        assert_eq!(heap.Type, D3D12_HEAP_TYPE_DEFAULT);

        let (rb, rb_ptr) = host
            .readback_buffer((2 * PAGE) as u64)
            .expect("readback buffer");

        // Submission 1: initial upload copies both pages into DEFAULT VRAM.
        let list = host.new_list().expect("new list");
        shadow.refresh(&list, mem.addr(), (2 * PAGE) as u64);
        // SAFETY: list is open; buffers stay alive through execute.
        unsafe {
            list.list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, (2 * PAGE) as u64);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // SAFETY: fence completed; readback buffer is mapped for 2 * PAGE bytes.
        let got = unsafe { std::slice::from_raw_parts(rb_ptr, 2 * PAGE) };
        assert_eq!(&got[0..4], &0x1122_3344u32.to_le_bytes());
        assert_eq!(&got[PAGE..PAGE + 4], &0x5566_7788u32.to_le_bytes());

        // Unchanged check: zero dirty pages -> zero bytes staged.
        assert_eq!(
            shadow.watched.load(Ordering::Relaxed),
            1,
            "write-watched allocation should be detected as watched"
        );
        let mut uploads = Vec::new();
        let staged = shadow.update(
            mem.addr() as usize,
            2 * PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(
            staged,
            0,
            "copied={:?}",
            shadow.copied.lock().unwrap_or_else(|e| e.into_inner())
        );
        assert!(uploads.is_empty());

        // Dirty only page 1: only page 1 is staged at dst_offset == PAGE.
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0x99AA_BBCCu32.to_le_bytes());
        let mut uploads = Vec::new();
        let staged = shadow.update(
            mem.addr() as usize,
            2 * PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, PAGE);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].dst_offset, PAGE as u64);
        assert_eq!(uploads[0].len, PAGE as u64);

        // Submission 2: write page 1 again and verify GPU sees new page 1 + original page 0.
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        list.reset();
        shadow.refresh(&list, mem.addr(), (2 * PAGE) as u64);
        // SAFETY: list is open; buffers stay alive through execute.
        unsafe {
            list.list
                .CopyBufferRegion(&rb, 0, &shadow.buffer, 0, (2 * PAGE) as u64);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // SAFETY: fence completed; readback buffer is mapped for 2 * PAGE bytes.
        let got = unsafe { std::slice::from_raw_parts(rb_ptr, 2 * PAGE) };
        assert_eq!(&got[0..4], &0x1122_3344u32.to_le_bytes());
        assert_eq!(&got[PAGE..PAGE + 4], &0xDEAD_BEEFu32.to_le_bytes());
    }

    #[test]
    fn gpu_writes_readback_merges_and_preserves_concurrent_cpu_writes() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut mem = WatchedAlloc::new(PAGE, 0x5000_0002);
        mem.bytes_mut().fill(0);
        // Indirect draw record at offset 0: vertex count = u32::MAX (clamped to 1<<24 by GPU),
        // instances = 1, start_vertex = 0, start_instance = 0; sentinel at offset 16.
        let initial: [u32; 5] = [u32::MAX, 1, 0, 0, 0x1234_5678];
        for (i, word) in initial.iter().enumerate() {
            mem.bytes_mut()[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
        }

        let shadow = find(mem.addr(), PAGE as u64).expect("default shadow");
        let list = host.new_list().expect("new list");
        shadow.refresh(&list, mem.addr(), PAGE as u64);
        shadow.gpu_writes(&list, mem.addr(), 64);
        assert!(crate::indirect::bound(
            &list,
            &shadow.buffer,
            shadow.gpu(mem.addr()),
            (0, 16, 1),
        ));
        crate::indirect::done(&list, &shadow.buffer);
        list.close();
        host.execute(&[&list]);
        // SAFETY: plain device-removed check.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());

        // GPU clamped word 0 to 1<<24 and readback merged it into title memory;
        // word 4 (not written by GPU) stayed untouched.
        let clamped = u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap());
        let sentinel = u32::from_le_bytes(mem.bytes_mut()[16..20].try_into().unwrap());
        assert_eq!(clamped, 1 << 24);
        assert_eq!(sentinel, 0x1234_5678);

        // Flush write-watch state so the page starts clean.
        let mut uploads = Vec::new();
        shadow.update(
            mem.addr() as usize,
            PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        uploads.clear();
        assert_eq!(
            shadow.update(
                mem.addr() as usize,
                PAGE,
                &mut |n| {
                    let (b, p) = host.upload_buffer(n as u64)?;
                    Some((b, 0, p))
                },
                &mut uploads,
            ),
            0
        );

        // Case 1: GPU-only write back on a whole page marks the page synced so our
        // own write-back is not re-uploaded on the next submission.
        let old = mem.bytes_mut().to_vec();
        let mut after = old.clone();
        after[0..4].copy_from_slice(&0x1357_2468u32.to_le_bytes());
        assert_eq!(shadow.copy_back_range(0, &old, &after), PAGE);
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap()),
            0x1357_2468
        );
        uploads.clear();
        assert_eq!(
            shadow.update(
                mem.addr() as usize,
                PAGE,
                &mut |n| {
                    let (b, p) = host.upload_buffer(n as u64)?;
                    Some((b, 0, p))
                },
                &mut uploads,
            ),
            0
        );

        // Case 2: GPU write on [0..4) plus concurrent CPU write on [32..36) of the
        // same page merges GPU bytes, preserves CPU bytes, and leaves the page dirty
        // so the next upload pushes the merged page back to DEFAULT VRAM.
        let old = mem.bytes_mut().to_vec();
        let mut after = old.clone();
        after[0..4].copy_from_slice(&0xCAFE_BABEu32.to_le_bytes());
        mem.bytes_mut()[32..36].copy_from_slice(&0xFACE_FEEDu32.to_le_bytes());

        assert_eq!(shadow.copy_back_range(0, &old, &after), PAGE);
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[0..4].try_into().unwrap()),
            0xCAFE_BABE
        );
        assert_eq!(
            u32::from_le_bytes(mem.bytes_mut()[32..36].try_into().unwrap()),
            0xFACE_FEED
        );

        uploads.clear();
        let staged = shadow.update(
            mem.addr() as usize,
            PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, PAGE);
        assert_eq!(uploads.len(), 1);
    }

    #[test]
    fn failed_upload_alloc_leaves_pages_stale() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mut mem = WatchedAlloc::new(2 * PAGE, 0x5000_0003);
        mem.bytes_mut().fill(0);
        mem.bytes_mut()[0..4].copy_from_slice(&0xAAAA_BBBBu32.to_le_bytes());
        mem.bytes_mut()[PAGE..PAGE + 4].copy_from_slice(&0xCCCC_DDDDu32.to_le_bytes());

        let shadow = find(mem.addr(), (2 * PAGE) as u64).expect("default shadow");

        // Simulate staging allocation failure: nothing is staged and `copied` stays 0.
        let mut uploads = Vec::new();
        let staged = shadow.update(mem.addr() as usize, 2 * PAGE, &mut |_| None, &mut uploads);
        assert_eq!(staged, 0);
        assert!(uploads.is_empty());
        assert_eq!(
            *shadow.copied.lock().unwrap_or_else(|e| e.into_inner()),
            vec![0, 0]
        );

        // Next submission with working allocation still stages both pages.
        let staged = shadow.update(
            mem.addr() as usize,
            2 * PAGE,
            &mut |n| {
                let (b, p) = host.upload_buffer(n as u64)?;
                Some((b, 0, p))
            },
            &mut uploads,
        );
        assert_eq!(staged, 2 * PAGE);
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].dst_offset, 0);
        assert_eq!(uploads[0].len, (2 * PAGE) as u64);
    }

    #[test]
    fn trim_overlapping_uploads_keeps_latest_bytes_disjoint() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let mem_a = WatchedAlloc::new(PAGE, 0x5000_0004);
        let mem_b = WatchedAlloc::new(PAGE, 0x5000_0005);
        let shadow_a = find(mem_a.addr(), PAGE as u64).expect("shadow a");
        let shadow_b = find(mem_b.addr(), PAGE as u64).expect("shadow b");
        let (src0, _) = host.upload_buffer(256).expect("upload 0");
        let (src1, _) = host.upload_buffer(256).expect("upload 1");
        let (src2, _) = host.upload_buffer(256).expect("upload 2");

        // Earlier upload [0..128) on shadow_a, then middle overwrite [32..96) on shadow_a,
        // then full overwrite of [0..64) on shadow_a, plus independent [0..128) on shadow_b.
        let mut uploads = vec![
            UploadCopy {
                shadow: shadow_a.clone(),
                src: src0.clone(),
                src_offset: 10,
                dst_offset: 0,
                len: 128,
            },
            UploadCopy {
                shadow: shadow_b.clone(),
                src: src0.clone(),
                src_offset: 0,
                dst_offset: 0,
                len: 128,
            },
            UploadCopy {
                shadow: shadow_a.clone(),
                src: src1.clone(),
                src_offset: 20,
                dst_offset: 32,
                len: 64,
            },
            UploadCopy {
                shadow: shadow_a.clone(),
                src: src2.clone(),
                src_offset: 30,
                dst_offset: 0,
                len: 64,
            },
        ];

        trim_overlapping_uploads(&mut uploads);

        // shadow_a should have strictly disjoint ranges covering [0..128):
        //   [96..128) from src0 (src_offset 10 + 96 = 106),
        //   [64..96) from src1 (src_offset 20 + 32 = 52),
        //   [0..64) from src2 (src_offset 30).
        // shadow_b keeps its full [0..128) untouched.
        let a_ranges: Vec<(u64, u64, u64)> = uploads
            .iter()
            .filter(|u| Arc::ptr_eq(&u.shadow, &shadow_a))
            .map(|u| (u.dst_offset, u.len, u.src_offset))
            .collect();
        assert_eq!(a_ranges, vec![(96, 32, 106), (64, 32, 52), (0, 64, 30)]);

        let b_ranges: Vec<(u64, u64, u64)> = uploads
            .iter()
            .filter(|u| Arc::ptr_eq(&u.shadow, &shadow_b))
            .map(|u| (u.dst_offset, u.len, u.src_offset))
            .collect();
        assert_eq!(b_ranges, vec![(0, 128, 0)]);
    }

    #[test]
    fn default_vram_multi_submission_compute_uav_and_overlap_roundtrip() {
        use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_MESSAGE_CALLBACK_FLAG_NONE,
            D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_ID,
            D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_INVALIDLIBRARYBLOB, D3D12_MESSAGE_SEVERITY,
            D3D12_MESSAGE_SEVERITY_ERROR, D3D12_SHADER_BYTECODE, ID3D12InfoQueue1,
            ID3D12PipelineState, ID3D12RootSignature,
        };
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        use windows::core::{Interface, PCSTR, s};

        static DEBUG_ERRORS: AtomicU32 = AtomicU32::new(0);
        unsafe extern "system" fn on_debug_message(
            _: D3D12_MESSAGE_CATEGORY,
            severity: D3D12_MESSAGE_SEVERITY,
            id: D3D12_MESSAGE_ID,
            desc: PCSTR,
            _: *mut std::ffi::c_void,
        ) {
            // `pipeline::library::tests::a_rejected_blob_gives_an_empty_library`
            // intentionally feeds an invalid blob on the shared device in parallel.
            if severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0
                && id != D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_INVALIDLIBRARYBLOB
            {
                DEBUG_ERRORS.fetch_add(1, AtomicOrdering::Relaxed);
                // SAFETY: D3D12 passes a valid NUL-terminated description for the callback.
                let msg = unsafe { desc.to_string() }.unwrap_or_default();
                eprintln!("D3D12 debug error (id={}): {msg}", id.0);
            }
        }

        struct DebugCallback(Option<(ID3D12InfoQueue1, u32)>);
        impl Drop for DebugCallback {
            fn drop(&mut self) {
                if let Some((ref q, cookie)) = self.0 {
                    // SAFETY: unregisters the cookie registered on this queue.
                    let _ = unsafe { q.UnregisterMessageCallback(cookie) };
                }
            }
        }

        let Some(host) = crate::host::host() else {
            return;
        };
        DEBUG_ERRORS.store(0, AtomicOrdering::Relaxed);
        let _debug_cb = DebugCallback(host.device.cast::<ID3D12InfoQueue1>().ok().and_then(|q| {
            let mut cookie = 0;
            // SAFETY: static callback function, unregistered on drop.
            unsafe {
                q.RegisterMessageCallback(
                    Some(on_debug_message),
                    D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                    std::ptr::null_mut(),
                    &mut cookie,
                )
            }
            .ok()
            .map(|()| (q, cookie))
        }));

        // SAFETY: plain DXGI adapter lookup for diagnostic logging.
        if let Ok(desc) = unsafe {
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
                .and_then(|a| a.GetDesc1())
        } {
            eprintln!(
                "shadow roundtrip adapter: {} ({:04x}:{:04x}, flags={:#x})",
                String::from_utf16_lossy(&desc.Description).trim_end_matches('\0'),
                desc.VendorId,
                desc.DeviceId,
                desc.Flags
            );
        }

        const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer Mode : register(b0) { uint pass_idx; uint page_bytes; };
ByteAddressBuffer src_buf : register(t0);
RWByteAddressBuffer dst_buf : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    if (pass_idx == 0) {
        uint a = src_buf.Load(0);
        uint b = src_buf.Load(64);
        dst_buf.Store(0, a ^ 0x0F0F0F0Fu);
        dst_buf.Store(page_bytes, b + 0x11111111u);
    } else {
        uint gpu_prev = dst_buf.Load(0);
        uint cpu_concurrent = dst_buf.Load(32);
        uint page1_prev = dst_buf.Load(page_bytes);
        dst_buf.Store(16, (gpu_prev ^ cpu_concurrent) + page1_prev);
    }
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: static HLSL source and valid output blob pointer.
        unsafe {
            D3DCompile(
                SHADER.as_ptr().cast(),
                SHADER.len(),
                s!("shadow_roundtrip"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("compile synthetic compute shader");
        let code = code.expect("shader blob");
        // SAFETY: valid compiled DXBC bytecode with embedded root signature.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
            let pso = host
                .device
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
        };

        let mut src_mem = WatchedAlloc::new(PAGE, 0x5000_0010);
        let mut staging_patch = WatchedAlloc::new(PAGE, 0x5000_0011);
        let mut dst_mem = WatchedAlloc::new(2 * PAGE, 0x5000_0012);
        src_mem.bytes_mut().fill(0);
        staging_patch.bytes_mut().fill(0);
        dst_mem.bytes_mut().fill(0);

        // Initial values in src_mem, plus a submission-time Copy overwrite at [0..4)
        // so one submission exercises overlapping UploadCopy trimming on the GPU.
        src_mem.bytes_mut()[0..4].copy_from_slice(&0xAAAA_0000u32.to_le_bytes());
        src_mem.bytes_mut()[64..68].copy_from_slice(&0x2222_3333u32.to_le_bytes());
        staging_patch.bytes_mut()[0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        dst_mem.bytes_mut()[12..16].copy_from_slice(&0x7788_99AAu32.to_le_bytes());

        let src_shadow = find(src_mem.addr(), PAGE as u64).expect("src shadow");
        let dst_shadow = find(dst_mem.addr(), (2 * PAGE) as u64).expect("dst shadow");

        // Submission 1:
        // - Refresh src_shadow, overwrite src_mem[0..4) via defer_copy, refresh again
        //   (overlapping upload in one prelude; GPU must see 0x1234_5678, not 0xAAAA_0000).
        // - Refresh dst_shadow (2 pages) and mark all 2 pages as gpu_writes.
        // - Simulate a concurrent CPU write to dst_mem[32..36) on page 0 while page 1
        //   only receives a GPU write.
        let list = host.new_list().expect("new list");
        list.keep(&root);
        list.keep(&pso);
        src_shadow.refresh(&list, src_mem.addr(), PAGE as u64);
        list.defer_copy(
            src_mem.addr() as *mut u8,
            staging_patch.addr() as *const u8,
            4,
        );
        src_shadow.refresh(&list, src_mem.addr(), 64);
        dst_shadow.refresh(&list, dst_mem.addr(), (2 * PAGE) as u64);
        dst_shadow.gpu_writes(&list, dst_mem.addr(), (2 * PAGE) as u64);
        let concurrent_dst = (dst_mem.addr() as usize) + 32;
        list.defer_host(move || {
            // SAFETY: 4-byte aligned slot inside `dst_mem`, disjoint from GPU writes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    0xCAFE_F00Du32.to_le_bytes().as_ptr(),
                    concurrent_dst as *mut u8,
                    4,
                );
            }
        });
        let pass0 = [0u32, PAGE as u32];
        // SAFETY: open command list with live pipeline and default-heap shadow buffers.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRoot32BitConstants(0, 2, pass0.as_ptr().cast(), 0);
            list.list
                .SetComputeRootShaderResourceView(1, src_shadow.gpu(src_mem.addr()));
            list.list
                .SetComputeRootUnorderedAccessView(2, dst_shadow.gpu(dst_mem.addr()));
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);

        let word =
            |buf: &[u8], off: usize| u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let expected_p0 = 0x1234_5678u32 ^ 0x0F0F_0F0Fu32;
        let expected_p1 = 0x2222_3333u32.wrapping_add(0x1111_1111u32);
        assert_eq!(word(dst_mem.bytes_mut(), 0), expected_p0);
        assert_eq!(word(dst_mem.bytes_mut(), 12), 0x7788_99AA);
        assert_eq!(word(dst_mem.bytes_mut(), 32), 0xCAFE_F00D);
        assert_eq!(word(dst_mem.bytes_mut(), PAGE), expected_p1);

        // Page 1 had only GPU writes so copy_back_range marked it synced; page 0 had a
        // concurrent CPU write at [32..36) so page 0 stayed dirty for the next refresh.
        let copied_after_pass0 = dst_shadow
            .copied
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        assert!(
            copied_after_pass0[1] > copied_after_pass0[0],
            "pure-GPU page 1 should advance copied while CPU+GPU page 0 stays dirty: {copied_after_pass0:?}"
        );

        // Submission 2: refresh dst_shadow (uploads merged page 0 into DEFAULT VRAM)
        // and run pass 1 on the GPU, which combines page 0's GPU output, page 0's
        // concurrent CPU write, and page 1's GPU output from DEFAULT VRAM.
        list.reset();
        list.keep(&root);
        list.keep(&pso);
        dst_shadow.refresh(&list, dst_mem.addr(), (2 * PAGE) as u64);
        dst_shadow.gpu_writes(&list, dst_mem.addr(), 64);
        let pass1 = [1u32, PAGE as u32];
        // SAFETY: open command list with live pipeline and default-heap shadow buffers.
        unsafe {
            list.list.SetComputeRootSignature(&root);
            list.list.SetPipelineState(&pso);
            list.list
                .SetComputeRoot32BitConstants(0, 2, pass1.as_ptr().cast(), 0);
            list.list
                .SetComputeRootShaderResourceView(1, src_shadow.gpu(src_mem.addr()));
            list.list
                .SetComputeRootUnorderedAccessView(2, dst_shadow.gpu(dst_mem.addr()));
            list.list.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);

        let expected_combined = (expected_p0 ^ 0xCAFE_F00Du32).wrapping_add(expected_p1);
        assert_eq!(word(dst_mem.bytes_mut(), 16), expected_combined);
        assert_eq!(DEBUG_ERRORS.load(AtomicOrdering::Relaxed), 0);
    }
}
