//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved
/// too, and their draws traced at debug level.
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list = std::env::var("VICTORY_CAPTURE_FRAMES").unwrap_or_default();
        list.split(',')
            .filter_map(|f| f.trim().parse().ok())
            .collect()
    })
    .contains(&n)
}

/// A plane selected for this frame, with the same resource ID as its target dump.
pub(crate) fn present_source(
    frame: u64,
    plane: usize,
    composition_index: usize,
    resource: crate::object::This,
    desc: &D3D12_RESOURCE_DESC,
) {
    tracing::info!(
        frame,
        plane,
        composition_index,
        ?resource,
        width = desc.Width,
        height = desc.Height,
        format = desc.Format.0,
        "present source"
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriteKind {
    ClearRtv,
    ClearDsv,
    Draw,
    DrawIndexed,
    DispatchMesh,
    Dispatch,
    IndirectDraw,
    IndirectIndexed,
    IndirectMesh,
    IndirectDispatch,
    CopyResource,
    CopyTextureRegion,
    CopyBufferRegion,
    ResolveSubresource,
    TextureUpload,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WriteRecord {
    pub(crate) target: usize,
    pub(crate) source: Option<usize>,
    pub(crate) kind: WriteKind,
    pub(crate) slot: Option<u32>,
    pub(crate) subresource: u32,
    pub(crate) width: u64,
    pub(crate) height: u32,
    pub(crate) format: u32,
    pub(crate) pipeline: String,
    pub(crate) submission: u64,
    pub(crate) list_index: usize,
    pub(crate) op_index: u32,
    pub(crate) frame: u64,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PresentPlane {
    pub(crate) plane: usize,
    pub(crate) composition_index: usize,
    pub(crate) resource: usize,
    pub(crate) width: u64,
    pub(crate) height: u32,
    pub(crate) format: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InputKind {
    Srv,
    Cbv,
    Uav,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BoundInput {
    pub(crate) kind: InputKind,
    pub(crate) stage: &'static str,
    pub(crate) slot: u32,
    pub(crate) resource: usize,
    pub(crate) is_buffer: bool,
    pub(crate) size: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DroppedCompute {
    pub(crate) hash: String,
    pub(crate) stage_or_kind: String,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DroppedDispatch {
    pub(crate) pipeline: String,
    pub(crate) reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InputTrace {
    pub(crate) pipeline: String,
    pub(crate) targets: Vec<usize>,
    pub(crate) inputs: Vec<BoundInput>,
    pub(crate) submission: u64,
    pub(crate) list_index: usize,
    pub(crate) op_index: u32,
    pub(crate) frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InputProvenance {
    pub(crate) input: BoundInput,
    pub(crate) writer: Option<WriteRecord>,
    pub(crate) chain: Vec<WriteRecord>,
    pub(crate) status: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DrawInputReport {
    pub(crate) frame: u64,
    pub(crate) pipeline: String,
    pub(crate) targets: Vec<usize>,
    pub(crate) submission: u64,
    pub(crate) list_index: usize,
    pub(crate) op_index: u32,
    pub(crate) inputs: Vec<InputProvenance>,
}

const MAX_TARGET_WRITES_PER_RESOURCE: usize = 64;
const MAX_FRAME_TARGETS: usize = 16_384;
const MAX_FRAME_BUFFER_WRITES: usize = 16_384;
const MAX_LAST_BUFFERS: usize = 32_768;
const MAX_INPUT_REPORTS: usize = 4_096;
const MAX_DROPPED_RECORDS: usize = 256;

#[derive(Default)]
struct WriteState {
    last_targets: std::collections::HashMap<usize, WriteRecord>,
    last_buffers: std::collections::BTreeMap<usize, WriteRecord>,
    last_buffer_max_width: u64,
    frame_targets: std::collections::HashMap<usize, Vec<WriteRecord>>,
    frame_buffers: std::collections::BTreeMap<usize, Vec<WriteRecord>>,
    frame_buffer_max_width: u64,
}

static WRITE_STATE: std::sync::Mutex<Option<WriteState>> = std::sync::Mutex::new(None);
static DROPPED_COMPUTES: std::sync::Mutex<std::collections::VecDeque<DroppedCompute>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
static DROPPED_DISPATCHES: std::sync::Mutex<std::collections::VecDeque<DroppedDispatch>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
static INPUT_REPORTS: std::sync::Mutex<Vec<DrawInputReport>> = std::sync::Mutex::new(Vec::new());

pub(crate) fn present_source_trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PRESENT_SOURCE_TRACE").is_some())
}

fn parse_filter_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| {
            let lower = s.trim().to_ascii_lowercase();
            lower.strip_prefix("0x").unwrap_or(&lower).to_owned()
        })
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_env_filters(var: &str) -> Vec<String> {
    parse_filter_list(&std::env::var(var).unwrap_or_default())
}

fn trace_inputs_pso_filters() -> &'static [String] {
    static FILTERS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    FILTERS.get_or_init(|| parse_env_filters("VICTORY_TRACE_INPUTS_PSO"))
}

fn trace_inputs_vs_filters() -> &'static [String] {
    static FILTERS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    FILTERS.get_or_init(|| parse_env_filters("VICTORY_TRACE_INPUTS_VS"))
}

fn should_trace_inputs_with(
    present_trace: bool,
    pso_filters: &[String],
    vs_filters: &[String],
    pipeline_name: &str,
    vs_hashes: &[&str],
) -> bool {
    if pso_filters.is_empty() && vs_filters.is_empty() {
        return present_trace;
    }
    if !pso_filters.is_empty() {
        let lower = pipeline_name.to_ascii_lowercase();
        if pso_filters.iter().any(|f| lower.contains(f)) {
            return true;
        }
    }
    if !vs_filters.is_empty() {
        for hash in vs_hashes {
            let lower = hash.to_ascii_lowercase();
            if vs_filters.iter().any(|f| lower.contains(f)) {
                return true;
            }
        }
    }
    false
}

pub(crate) fn should_trace_inputs(pipeline_name: &str, vs_hashes: &[&str]) -> bool {
    should_trace_inputs_with(
        present_source_trace_on(),
        trace_inputs_pso_filters(),
        trace_inputs_vs_filters(),
        pipeline_name,
        vs_hashes,
    )
}

fn push_dropped_compute(
    dropped: &mut std::collections::VecDeque<DroppedCompute>,
    hash: &str,
    stage_or_kind: &str,
    reason: &str,
) {
    if let Some(idx) = dropped
        .iter()
        .position(|d| d.hash == hash && d.stage_or_kind == stage_or_kind)
    {
        if let Some(mut item) = dropped.remove(idx) {
            if item.reason == "shader translation failed" && reason != "shader translation failed" {
                item.reason = reason.to_owned();
            }
            dropped.push_back(item);
        }
        return;
    }
    if dropped.len() >= MAX_DROPPED_RECORDS {
        dropped.pop_front();
    }
    dropped.push_back(DroppedCompute {
        hash: hash.to_owned(),
        stage_or_kind: stage_or_kind.to_owned(),
        reason: reason.to_owned(),
    });
}

fn push_dropped_dispatch(
    dropped: &mut std::collections::VecDeque<DroppedDispatch>,
    pipeline: &str,
    reason: &str,
) {
    if let Some(idx) = dropped
        .iter()
        .position(|d| d.pipeline == pipeline && d.reason == reason)
    {
        if let Some(item) = dropped.remove(idx) {
            dropped.push_back(item);
        }
        return;
    }
    if dropped.len() >= MAX_DROPPED_RECORDS {
        dropped.pop_front();
    }
    dropped.push_back(DroppedDispatch {
        pipeline: pipeline.to_owned(),
        reason: reason.to_owned(),
    });
}

pub(crate) fn record_dropped_compute(hash: &str, stage_or_kind: &str, reason: &str) {
    let mut dropped = DROPPED_COMPUTES.lock().unwrap_or_else(|e| e.into_inner());
    push_dropped_compute(&mut dropped, hash, stage_or_kind, reason);
}

pub(crate) fn record_dropped_dispatch(pipeline: &str, reason: &str) {
    let mut dropped = DROPPED_DISPATCHES.lock().unwrap_or_else(|e| e.into_inner());
    push_dropped_dispatch(&mut dropped, pipeline, reason);
}

pub(crate) fn resource_dims(resource: crate::object::This) -> (u64, u32, u32) {
    let Some(tex) = crate::resource::host_resource(resource) else {
        return (0, 0, 0);
    };
    // SAFETY: a live host resource.
    let d = unsafe { tex.GetDesc() };
    (d.Width, d.Height, d.Format.0 as u32)
}

fn is_buffer_write(w: &WriteRecord) -> bool {
    w.height == 1 && w.format == 0
}

fn prune_oldest_target_keys(
    map: &mut std::collections::HashMap<usize, Vec<WriteRecord>>,
    keep: usize,
) {
    let mut latest: Vec<(usize, (u64, usize, u32))> = map
        .iter()
        .filter_map(|(&k, v)| {
            v.last()
                .map(|w| (k, (w.submission, w.list_index, w.op_index)))
        })
        .collect();
    latest.sort_unstable_by_key(|&(_, order)| order);
    let remove_count = latest.len().saturating_sub(keep);
    for (k, _) in latest.into_iter().take(remove_count) {
        map.remove(&k);
    }
}

fn prune_oldest_buffer_lists(
    map: &mut std::collections::BTreeMap<usize, Vec<WriteRecord>>,
    keep: usize,
) {
    let mut latest: Vec<(usize, (u64, usize, u32))> = map
        .iter()
        .filter_map(|(&k, v)| {
            v.last()
                .map(|w| (k, (w.submission, w.list_index, w.op_index)))
        })
        .collect();
    latest.sort_unstable_by_key(|&(_, order)| order);
    let remove_count = latest.len().saturating_sub(keep);
    for (k, _) in latest.into_iter().take(remove_count) {
        map.remove(&k);
    }
}

impl WriteState {
    fn commit(&mut self, frame: u64, submission: u64, list_index: usize, writes: &[WriteRecord]) {
        if writes.is_empty() {
            return;
        }
        let mut ordered = writes.to_vec();
        ordered.sort_by_key(|w| w.op_index);
        for mut record in ordered {
            record.frame = frame;
            record.submission = submission;
            record.list_index = list_index;
            if is_buffer_write(&record) {
                self.last_buffer_max_width = self.last_buffer_max_width.max(record.width);
                self.frame_buffer_max_width = self.frame_buffer_max_width.max(record.width);
                self.last_buffers.insert(record.target, record.clone());
                let list = self.frame_buffers.entry(record.target).or_default();
                list.push(record);
                if list.len() > MAX_TARGET_WRITES_PER_RESOURCE {
                    let excess = list.len() - MAX_TARGET_WRITES_PER_RESOURCE;
                    list.drain(..excess);
                }
            } else {
                self.last_targets.insert(record.target, record.clone());
                let list = self.frame_targets.entry(record.target).or_default();
                list.push(record);
                if list.len() > MAX_TARGET_WRITES_PER_RESOURCE {
                    let excess = list.len() - MAX_TARGET_WRITES_PER_RESOURCE;
                    list.drain(..excess);
                }
            }
        }
        if self.frame_targets.len() > MAX_FRAME_TARGETS {
            prune_oldest_target_keys(&mut self.frame_targets, MAX_FRAME_TARGETS / 2);
        }
        if self.frame_buffers.len() > MAX_FRAME_BUFFER_WRITES {
            prune_oldest_buffer_lists(&mut self.frame_buffers, MAX_FRAME_BUFFER_WRITES / 2);
        }
        if self.last_buffers.len() > MAX_LAST_BUFFERS {
            let mut by_age: Vec<(usize, (u64, usize, u32))> = self
                .last_buffers
                .iter()
                .map(|(&k, w)| (k, (w.submission, w.list_index, w.op_index)))
                .collect();
            by_age.sort_unstable_by_key(|&(_, order)| order);
            let remove_count = by_age.len().saturating_sub(MAX_LAST_BUFFERS / 2);
            for (k, _) in by_age.into_iter().take(remove_count) {
                self.last_buffers.remove(&k);
            }
        }
    }

    fn forget(&mut self, resource: usize, buffer_range: Option<(usize, u64)>) {
        self.last_targets.remove(&resource);
        self.frame_targets.remove(&resource);
        if buffer_range.is_some() {
            self.last_buffers
                .retain(|_, w| !record_matches_forget(w, resource, buffer_range));
            self.frame_buffers.retain(|_, list| {
                list.retain(|w| !record_matches_forget(w, resource, buffer_range));
                !list.is_empty()
            });
            if self.last_buffers.is_empty() {
                self.last_buffer_max_width = 0;
            }
            if self.frame_buffers.is_empty() {
                self.frame_buffer_max_width = 0;
            }
        } else {
            self.last_buffers.remove(&resource);
            self.frame_buffers.remove(&resource);
        }
    }

    fn end_frame(&mut self, frame: u64) {
        self.frame_targets.retain(|_, list| {
            list.retain(|w| w.frame > frame);
            !list.is_empty()
        });
        self.frame_buffers.retain(|_, list| {
            list.retain(|w| w.frame > frame);
            !list.is_empty()
        });
        if self.frame_buffers.is_empty() {
            self.frame_buffer_max_width = 0;
        }
    }
}

fn record_matches_forget(
    w: &WriteRecord,
    resource: usize,
    buffer_range: Option<(usize, u64)>,
) -> bool {
    if w.target == resource {
        return true;
    }
    let Some((base, len)) = buffer_range else {
        return false;
    };
    if len == 0 || !is_buffer_write(w) {
        return false;
    }
    let b_start = base as u128;
    let b_end = b_start + u128::from(len);
    let w_start = w.target as u128;
    if w.width > 0 {
        let w_end = w_start + u128::from(w.width);
        w_start < b_end && b_start < w_end
    } else {
        w_start >= b_start && w_start < b_end
    }
}

pub(crate) fn forget_resource(resource: usize, buffer_range: Option<(usize, u64)>) {
    let mut guard = WRITE_STATE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(state) = guard.as_mut() {
        state.forget(resource, buffer_range);
    }
}

/// Commits writes recorded on a host command list after GPU execution succeeds.
pub(crate) fn commit_writes(
    frame: u64,
    submission: u64,
    list_index: usize,
    writes: &[WriteRecord],
) {
    if writes.is_empty() {
        return;
    }
    let mut guard = WRITE_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = guard.get_or_insert_with(WriteState::default);
    state.commit(frame, submission, list_index, writes);
}

#[cfg(test)]
pub(crate) fn last_writer(resource: usize) -> Option<WriteRecord> {
    let guard = WRITE_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = guard.as_ref()?;
    state
        .last_targets
        .get(&resource)
        .or_else(|| state.last_buffers.get(&resource))
        .cloned()
}

fn write_matches_input(w: &WriteRecord, resource: usize, is_buffer: bool, size: u64) -> bool {
    if !is_buffer || !is_buffer_write(w) || w.width == 0 {
        return w.target == resource;
    }
    let w_start = w.target as u128;
    let w_end = w_start + u128::from(w.width);
    let r_start = resource as u128;
    if size > 0 {
        let r_end = r_start + u128::from(size);
        w_start < r_end && r_start < w_end
    } else {
        r_start >= w_start && r_start < w_end
    }
}

fn buffer_range_bounds(resource: usize, size: u64, max_width: u64) -> (usize, Option<usize>) {
    let r_start = resource as u128;
    let r_end = if size > 0 {
        r_start + u128::from(size)
    } else {
        r_start + 1
    };
    let min_target = r_start.saturating_sub(u128::from(max_width));
    let min_key = usize::try_from(min_target).unwrap_or(usize::MAX);
    let end_key = usize::try_from(r_end).ok();
    (min_key, end_key)
}

fn prior_write_from_state(
    state: &WriteState,
    resource: usize,
    is_buffer: bool,
    size: u64,
    before: (u64, usize, u32),
) -> Option<WriteRecord> {
    if !is_buffer {
        return state
            .frame_targets
            .get(&resource)
            .and_then(|list| {
                list.iter()
                    .filter(|w| (w.submission, w.list_index, w.op_index) < before)
                    .max_by_key(|w| (w.submission, w.list_index, w.op_index))
                    .cloned()
            })
            .or_else(|| {
                state
                    .last_targets
                    .get(&resource)
                    .filter(|w| (w.submission, w.list_index, w.op_index) < before)
                    .cloned()
            });
    }

    let (min_key, end_key) = buffer_range_bounds(resource, size, state.frame_buffer_max_width);
    let frame_hit = match end_key {
        Some(end) => state
            .frame_buffers
            .range(min_key..end)
            .flat_map(|(_, list)| list.iter())
            .filter(|w| {
                (w.submission, w.list_index, w.op_index) < before
                    && write_matches_input(w, resource, true, size)
            })
            .max_by_key(|w| (w.submission, w.list_index, w.op_index))
            .cloned(),
        None => state
            .frame_buffers
            .range(min_key..)
            .flat_map(|(_, list)| list.iter())
            .filter(|w| {
                (w.submission, w.list_index, w.op_index) < before
                    && write_matches_input(w, resource, true, size)
            })
            .max_by_key(|w| (w.submission, w.list_index, w.op_index))
            .cloned(),
    };
    if frame_hit.is_some() {
        return frame_hit;
    }

    let (min_key, end_key) = buffer_range_bounds(resource, size, state.last_buffer_max_width);
    match end_key {
        Some(end) => state
            .last_buffers
            .range(min_key..end)
            .filter_map(|(_, w)| {
                ((w.submission, w.list_index, w.op_index) < before
                    && write_matches_input(w, resource, true, size))
                .then_some(w)
            })
            .max_by_key(|w| (w.submission, w.list_index, w.op_index))
            .cloned(),
        None => state
            .last_buffers
            .range(min_key..)
            .filter_map(|(_, w)| {
                ((w.submission, w.list_index, w.op_index) < before
                    && write_matches_input(w, resource, true, size))
                .then_some(w)
            })
            .max_by_key(|w| (w.submission, w.list_index, w.op_index))
            .cloned(),
    }
}

fn trace_chain_from(first: WriteRecord, max_depth: usize, state: &WriteState) -> Vec<WriteRecord> {
    if max_depth == 0 {
        return Vec::new();
    }
    let mut chain = Vec::with_capacity(max_depth.min(16));
    chain.push(first);
    while chain.len() < max_depth {
        let current = &chain[chain.len() - 1];
        let Some(src) = current.source else {
            break;
        };
        let is_buffer = is_buffer_write(current);
        let size = if is_buffer { current.width } else { 0 };
        let cur_order = (current.submission, current.list_index, current.op_index);
        match prior_write_from_state(state, src, is_buffer, size, cur_order) {
            Some(record) => chain.push(record),
            None => break,
        }
    }
    chain
}

/// Walks committed `source` links backward with strictly decreasing
/// `(submission, list_index, op_index)` order and a depth cap so ping-pong
/// double-buffer chains (`A -> B -> A`) resolve without looping.
pub(crate) fn writer_chain(resource: usize, max_depth: usize) -> Vec<WriteRecord> {
    if max_depth == 0 {
        return Vec::new();
    }
    let guard = WRITE_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let Some(state) = guard.as_ref() else {
        return Vec::new();
    };
    let Some(first) = state
        .last_targets
        .get(&resource)
        .or_else(|| state.last_buffers.get(&resource))
        .cloned()
    else {
        return Vec::new();
    };
    trace_chain_from(first, max_depth, state)
}

fn format_dropped_upstream_tags(
    computes: &std::collections::VecDeque<DroppedCompute>,
    dispatches: &std::collections::VecDeque<DroppedDispatch>,
) -> String {
    let mut status = String::from("never host-written");
    if !computes.is_empty() {
        let list: Vec<String> = computes
            .iter()
            .rev()
            .take(8)
            .map(|c| format!("{} {} ({})", c.stage_or_kind, c.hash, c.reason))
            .collect();
        status.push_str("; dropped upstream CS translation: ");
        status.push_str(&list.join(", "));
    }
    if !dispatches.is_empty() {
        let list: Vec<String> = dispatches
            .iter()
            .rev()
            .take(8)
            .map(|d| format!("{} ({})", d.pipeline, d.reason))
            .collect();
        status.push_str("; dropped upstream dispatch: ");
        status.push_str(&list.join(", "));
    }
    status
}

/// Correlates bound SRV/CBV/UAV inputs recorded for draws/dispatches against
/// strictly earlier committed `WriteRecord`s and upstream dropped CS/dispatches.
pub(crate) fn commit_input_traces(
    frame: u64,
    submission: u64,
    list_index: usize,
    traces: &[InputTrace],
) {
    if traces.is_empty() {
        return;
    }
    let dropped_computes = DROPPED_COMPUTES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let dropped_dispatches = DROPPED_DISPATCHES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();

    let mut new_reports = Vec::with_capacity(traces.len());
    {
        let guard = WRITE_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let empty = WriteState::default();
        let state = guard.as_ref().unwrap_or(&empty);

        for trace in traces {
            let order = (submission, list_index, trace.op_index);
            let mut provenances = Vec::with_capacity(trace.inputs.len());
            for input in &trace.inputs {
                let writer = prior_write_from_state(
                    state,
                    input.resource,
                    input.is_buffer,
                    input.size,
                    order,
                );
                let (chain, status) = match &writer {
                    Some(w) => {
                        let chain = trace_chain_from(w.clone(), 8, state);
                        let status = format!("host-written by {:?} ({})", w.kind, w.pipeline);
                        tracing::info!(
                            frame,
                            pipeline = %trace.pipeline,
                            stage = input.stage,
                            input_kind = ?input.kind,
                            slot = input.slot,
                            resource = format_args!("{:#x}", input.resource),
                            is_buffer = input.is_buffer,
                            writer_kind = ?w.kind,
                            writer_pipeline = %w.pipeline,
                            chain_len = chain.len(),
                            status = %status,
                            "bound input provenance"
                        );
                        (chain, status)
                    }
                    None => {
                        let status =
                            format_dropped_upstream_tags(&dropped_computes, &dropped_dispatches);
                        tracing::info!(
                            frame,
                            pipeline = %trace.pipeline,
                            stage = input.stage,
                            input_kind = ?input.kind,
                            slot = input.slot,
                            resource = format_args!("{:#x}", input.resource),
                            is_buffer = input.is_buffer,
                            status = %status,
                            "bound input provenance"
                        );
                        (Vec::new(), status)
                    }
                };
                provenances.push(InputProvenance {
                    input: input.clone(),
                    writer,
                    chain,
                    status,
                });
            }
            new_reports.push(DrawInputReport {
                frame,
                pipeline: trace.pipeline.clone(),
                targets: trace.targets.clone(),
                submission,
                list_index,
                op_index: trace.op_index,
                inputs: provenances,
            });
        }
    }

    let mut reports = INPUT_REPORTS.lock().unwrap_or_else(|e| e.into_inner());
    reports.extend(new_reports);
    if reports.len() > MAX_INPUT_REPORTS {
        let excess = reports.len() - MAX_INPUT_REPORTS;
        reports.drain(..excess);
    }
}

#[cfg(test)]
pub(crate) fn input_reports(frame: u64) -> Vec<DrawInputReport> {
    INPUT_REPORTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|r| r.frame == frame)
        .cloned()
        .collect()
}

pub(crate) fn correlate_present(frame: u64, planes: &[PresentPlane]) {
    let dropped_computes = DROPPED_COMPUTES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let dropped_dispatches = DROPPED_DISPATCHES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let reports: Vec<DrawInputReport> = INPUT_REPORTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|r| r.frame == frame)
        .cloned()
        .collect();

    for p in planes {
        let chain = writer_chain(p.resource, 8);
        if chain.is_empty() {
            let status = format_dropped_upstream_tags(&dropped_computes, &dropped_dispatches);
            tracing::info!(
                frame,
                plane = p.plane,
                composition_index = p.composition_index,
                resource = format_args!("{:#x}", p.resource),
                width = p.width,
                height = p.height,
                format = p.format,
                status = %status,
                "present source has no recorded writer"
            );
        } else {
            let steps: Vec<String> = chain
                .iter()
                .map(|w| {
                    format!(
                        "{:?}({:#x}<-{:?} {}x{} f{} slot={:?} sub={} sub#={} list#={} op#={} frame={} {})",
                        w.kind,
                        w.target,
                        w.source.map(|s| format!("{s:#x}")),
                        w.width,
                        w.height,
                        w.format,
                        w.slot,
                        w.subresource,
                        w.submission,
                        w.list_index,
                        w.op_index,
                        w.frame,
                        w.pipeline
                    )
                })
                .collect();
            tracing::info!(
                frame,
                plane = p.plane,
                composition_index = p.composition_index,
                resource = format_args!("{:#x}", p.resource),
                width = p.width,
                height = p.height,
                format = p.format,
                chain = ?steps,
                "present source writer chain"
            );
            for w in &chain {
                for r in reports
                    .iter()
                    .filter(|r| r.targets.contains(&w.target) && r.submission == w.submission)
                {
                    let summary: Vec<String> = r
                        .inputs
                        .iter()
                        .map(|ip| {
                            format!(
                                "{:?}:{}@{:#x}=>{}",
                                ip.input.kind, ip.input.slot, ip.input.resource, ip.status
                            )
                        })
                        .collect();
                    tracing::info!(
                        frame,
                        target = format_args!("{:#x}", w.target),
                        pipeline = %r.pipeline,
                        inputs = ?summary,
                        "present source upstream input provenance"
                    );
                }
            }
        }
    }
    let candidates: Vec<WriteRecord> = {
        let guard = WRITE_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let mut by_target = Vec::new();
        if let Some(state) = guard.as_ref() {
            for list in state.frame_targets.values() {
                if let Some(w) = list.iter().rev().find(|w| w.frame == frame) {
                    by_target.push(w.clone());
                }
            }
            for list in state.frame_buffers.values() {
                if let Some(w) = list.iter().rev().find(|w| w.frame == frame) {
                    by_target.push(w.clone());
                }
            }
            by_target.sort_unstable_by_key(|w| (w.submission, w.list_index, w.op_index));
        }
        by_target
    };
    for c in &candidates {
        tracing::info!(
            frame,
            target = format_args!("{:#x}", c.target),
            source = ?c.source.map(|s| format!("{s:#x}")),
            kind = ?c.kind,
            width = c.width,
            height = c.height,
            format = c.format,
            slot = ?c.slot,
            subresource = c.subresource,
            submission = c.submission,
            list_index = c.list_index,
            op_index = c.op_index,
            pipeline = %c.pipeline,
            "present frame candidate target"
        );
    }
}

pub(crate) fn end_frame(frame: u64) {
    if let Some(state) = WRITE_STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_mut()
    {
        state.end_frame(frame);
    }
    INPUT_REPORTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|r| r.frame > frame);
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    folder().is_some() && (n.is_power_of_two() || listed(n))
}

/// Saves 8-bit RGBA or BGRA rows (`pitch` bytes apart) as frame `n`.
pub(crate) fn save(n: u64, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_as(&format!("frame-{n}"), width, height, pitch, bgra, rows);
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3),
    );
    chunk(&mut png, b"IEND", &[]);
    let path = folder.join(format!("{name}.png"));
    match std::fs::write(&path, png) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
static TARGETS: std::sync::Mutex<Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note_target(resource: crate::object::This) {
    if resource.is_null() || !dump_on() || !listed(crate::queue::presented() + 1) {
        return;
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|&(r, _)| r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy));
    }
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        dump_one(host, texture, state, frame, i, *resource as crate::object::This, &d);
    }
}

fn dump_one(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) {
    let Ok(list) = host.new_list() else { return };
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back(&host.device, &list.list, texture);
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        read
    };
    list.close();
    host.execute(&[&list]);
    let Some((buffer, footprint)) = read else { return };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
        if let Some(rgba) = to_rgba(f.Format.0 as u32, f.Width as usize, f.Height as usize, pitch, rows) {
            let name = format!(
                "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, &rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        buffer.Unmap(0, None);
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Rows of a host format as 8-bit RGBA: 8-bit color as is, everything else
/// as channel values scaled so the largest finite one is white (single
/// channels: lowest black, highest white, as grey). `None` for formats not
/// handled.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        11..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    tracing::info!(format, low, high, "target values");
    for v in values {
        let c = v.map(|c| (((c - low) / range).clamp(0.0, 1.0) * 255.0) as u8);
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_write(
        target: usize,
        source: Option<usize>,
        kind: WriteKind,
        slot: Option<u32>,
        op_index: u32,
        pipeline: &str,
    ) -> WriteRecord {
        WriteRecord {
            target,
            source,
            kind,
            slot,
            subresource: 0,
            width: 64,
            height: 64,
            format: 28,
            pipeline: pipeline.to_owned(),
            submission: 0,
            list_index: 0,
            op_index,
            frame: 0,
        }
    }

    fn synthetic_buffer_write(
        target: usize,
        source: Option<usize>,
        kind: WriteKind,
        width: u64,
        op_index: u32,
        pipeline: &str,
    ) -> WriteRecord {
        WriteRecord {
            target,
            source,
            kind,
            slot: None,
            subresource: 0,
            width,
            height: 1,
            format: 0,
            pipeline: pipeline.to_owned(),
            submission: 0,
            list_index: 0,
            op_index,
            frame: 0,
        }
    }

    fn frame_writes_for_target(target: usize) -> Vec<WriteRecord> {
        let guard = WRITE_STATE.lock().unwrap_or_else(|e| e.into_inner());
        let Some(state) = guard.as_ref() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        if let Some(list) = state.frame_targets.get(&target) {
            out.extend(list.iter().cloned());
        }
        if let Some(list) = state.frame_buffers.get(&target) {
            out.extend(list.iter().cloned());
        }
        out.sort_by_key(|w| (w.submission, w.list_index, w.op_index));
        out
    }

    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 127, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }

    #[test]
    fn commit_writes_orders_deterministically_across_lists() {
        let target = 0x1000_0001usize;
        // Out-of-order op_index inside list 0, followed by list 1.
        let list0 = vec![
            synthetic_write(target, None, WriteKind::Draw, Some(0), 2, "draw_late"),
            synthetic_write(target, None, WriteKind::ClearRtv, None, 0, ""),
        ];
        let list1 = vec![
            synthetic_write(
                target,
                Some(0x1000_0002),
                WriteKind::CopyResource,
                None,
                0,
                "",
            ),
            synthetic_write(
                target,
                Some(0x1000_0003),
                WriteKind::ResolveSubresource,
                None,
                1,
                "",
            ),
        ];
        commit_writes(7, 10, 0, &list0);
        commit_writes(7, 10, 1, &list1);

        let last = last_writer(target).expect("target must have a committed last writer");
        assert_eq!(last.kind, WriteKind::ResolveSubresource);
        assert_eq!(last.source, Some(0x1000_0003));
        assert_eq!(
            (last.frame, last.submission, last.list_index, last.op_index),
            (7, 10, 1, 1)
        );

        let committed = frame_writes_for_target(target);
        let order: Vec<(u64, usize, u32, WriteKind)> = committed
            .iter()
            .map(|w| (w.submission, w.list_index, w.op_index, w.kind))
            .collect();
        assert_eq!(
            order,
            vec![
                (10, 0, 0, WriteKind::ClearRtv),
                (10, 0, 2, WriteKind::Draw),
                (10, 1, 0, WriteKind::CopyResource),
                (10, 1, 1, WriteKind::ResolveSubresource),
            ]
        );
        end_frame(7);
    }

    #[test]
    fn writer_chain_traces_draw_dispatch_copy_and_resolve() {
        let rtv = 0x2000_0001usize;
        let uav = 0x2000_0002usize;
        let copied = 0x2000_0003usize;
        let msaa = 0x2000_0004usize;
        let resolved = 0x2000_0005usize;

        commit_writes(
            8,
            20,
            0,
            &[
                synthetic_write(rtv, None, WriteKind::Draw, Some(0), 0, "gbuffer_vs_ps"),
                synthetic_write(uav, Some(rtv), WriteKind::Dispatch, Some(0), 1, "post_cs"),
                synthetic_write(copied, Some(uav), WriteKind::CopyTextureRegion, None, 2, ""),
                synthetic_write(
                    msaa,
                    Some(copied),
                    WriteKind::DrawIndexed,
                    Some(0),
                    3,
                    "msaa_ps",
                ),
                synthetic_write(
                    resolved,
                    Some(msaa),
                    WriteKind::ResolveSubresource,
                    None,
                    4,
                    "",
                ),
            ],
        );

        let chain = writer_chain(resolved, 8);
        let kinds: Vec<(usize, WriteKind, Option<usize>)> =
            chain.iter().map(|w| (w.target, w.kind, w.source)).collect();
        assert_eq!(
            kinds,
            vec![
                (resolved, WriteKind::ResolveSubresource, Some(msaa)),
                (msaa, WriteKind::DrawIndexed, Some(copied)),
                (copied, WriteKind::CopyTextureRegion, Some(uav)),
                (uav, WriteKind::Dispatch, Some(rtv)),
                (rtv, WriteKind::Draw, None),
            ]
        );
        assert_eq!(writer_chain(resolved, 2).len(), 2);
        assert!(writer_chain(resolved, 0).is_empty());
        end_frame(8);
    }

    #[test]
    fn writer_chain_ping_pong_double_buffer_terminates_monotonically() {
        let buf_a = 0x3000_0001usize;
        let buf_b = 0x3000_0002usize;

        // Frame 9: Draw -> A, then ping-pong A -> B -> A -> B -> A.
        commit_writes(
            9,
            30,
            0,
            &[
                synthetic_write(buf_a, None, WriteKind::Draw, Some(0), 0, "base_pass"),
                synthetic_write(buf_b, Some(buf_a), WriteKind::Dispatch, Some(0), 1, "taa_0"),
                synthetic_write(buf_a, Some(buf_b), WriteKind::CopyResource, None, 2, ""),
            ],
        );
        commit_writes(
            9,
            30,
            1,
            &[
                synthetic_write(buf_b, Some(buf_a), WriteKind::Dispatch, Some(0), 0, "taa_1"),
                synthetic_write(
                    buf_a,
                    Some(buf_b),
                    WriteKind::CopyTextureRegion,
                    None,
                    1,
                    "",
                ),
            ],
        );

        let chain = writer_chain(buf_a, 8);
        let trace: Vec<(usize, WriteKind, usize, u32, Option<usize>)> = chain
            .iter()
            .map(|w| (w.target, w.kind, w.list_index, w.op_index, w.source))
            .collect();
        assert_eq!(
            trace,
            vec![
                (buf_a, WriteKind::CopyTextureRegion, 1, 1, Some(buf_b)),
                (buf_b, WriteKind::Dispatch, 1, 0, Some(buf_a)),
                (buf_a, WriteKind::CopyResource, 0, 2, Some(buf_b)),
                (buf_b, WriteKind::Dispatch, 0, 1, Some(buf_a)),
                (buf_a, WriteKind::Draw, 0, 0, None),
            ]
        );

        // Enforce depth cap even inside a ping-pong sequence.
        assert_eq!(writer_chain(buf_a, 3).len(), 3);
        end_frame(9);

        // Same-frame monotonic guard: a newer write to A must never be pulled
        // in as the historical source of an earlier write to B, while an
        // earlier write in the frame IS found.
        commit_writes(
            10,
            31,
            0,
            &[
                synthetic_write(buf_a, None, WriteKind::Draw, Some(0), 0, "earlier"),
                synthetic_write(buf_b, Some(buf_a), WriteKind::CopyResource, None, 1, ""),
                synthetic_write(buf_a, None, WriteKind::ClearRtv, None, 2, ""),
            ],
        );
        let b_chain = writer_chain(buf_b, 8);
        assert_eq!(b_chain.len(), 2);
        assert_eq!((b_chain[0].submission, b_chain[0].op_index), (31, 1));
        assert_eq!(
            (b_chain[1].submission, b_chain[1].op_index, b_chain[1].kind),
            (31, 0, WriteKind::Draw)
        );
        end_frame(10);

        // Cross-frame guard: if A was overwritten later in frame 11 after
        // B copied from A, and frame 10 was already ended, LAST_WRITERS[A]
        // is newer than B's copy and must be rejected rather than misattributed.
        let buf_c = 0x3000_0003usize;
        let buf_d = 0x3000_0004usize;
        commit_writes(
            11,
            32,
            0,
            &[
                synthetic_write(buf_d, Some(buf_c), WriteKind::CopyResource, None, 0, ""),
                synthetic_write(buf_c, None, WriteKind::ClearRtv, None, 1, ""),
            ],
        );
        assert_eq!(writer_chain(buf_d, 8).len(), 1);
        end_frame(11);
    }

    #[test]
    fn correlate_present_and_end_frame_clear_current_frame_writes() {
        let tex_11 = 0x4000_0001usize;
        let tex_12 = 0x4000_0002usize;
        commit_writes(
            11,
            40,
            0,
            &[synthetic_write(
                tex_11,
                None,
                WriteKind::Draw,
                Some(0),
                0,
                "ui",
            )],
        );
        commit_writes(
            12,
            41,
            0,
            &[synthetic_write(
                tex_12,
                None,
                WriteKind::ClearRtv,
                None,
                0,
                "",
            )],
        );
        correlate_present(
            11,
            &[PresentPlane {
                plane: 0,
                composition_index: 0,
                resource: tex_11,
                width: 64,
                height: 64,
                format: 28,
            }],
        );
        end_frame(11);
        let remaining: Vec<u64> = frame_writes_for_target(tex_11)
            .into_iter()
            .chain(frame_writes_for_target(tex_12))
            .map(|w| w.frame)
            .collect();
        assert_eq!(remaining, vec![12]);
        end_frame(12);
    }

    #[test]
    fn bound_input_provenance_enforces_monotonic_order_and_tags_dropped_cs() {
        let written_tex = 0x5000_0001usize;
        let unwritten_buf = 0x5000_0010usize;
        let future_tex = 0x5000_0020usize;
        let out_rtv = 0x5000_0030usize;

        record_dropped_compute("e18931c72af42e46", "compute", "unsupported opcode");
        record_dropped_dispatch(
            "compute e18931c72af42e46",
            "compute shader translation failed",
        );

        commit_writes(
            13,
            50,
            0,
            &[
                synthetic_write(written_tex, None, WriteKind::Draw, Some(0), 0, "gbuffer"),
                synthetic_write(out_rtv, None, WriteKind::Draw, Some(0), 1, "lighting"),
                synthetic_write(future_tex, None, WriteKind::ClearRtv, None, 2, "later"),
            ],
        );
        commit_input_traces(
            13,
            50,
            0,
            &[InputTrace {
                pipeline: "lighting".to_owned(),
                targets: vec![out_rtv],
                inputs: vec![
                    BoundInput {
                        kind: InputKind::Srv,
                        stage: "pixel",
                        slot: 0,
                        resource: written_tex,
                        is_buffer: false,
                        size: 0,
                    },
                    BoundInput {
                        kind: InputKind::Cbv,
                        stage: "pixel",
                        slot: 1,
                        resource: unwritten_buf,
                        is_buffer: true,
                        size: 256,
                    },
                    BoundInput {
                        kind: InputKind::Srv,
                        stage: "pixel",
                        slot: 2,
                        resource: future_tex,
                        is_buffer: false,
                        size: 0,
                    },
                ],
                submission: 0,
                list_index: 0,
                op_index: 1,
                frame: 0,
            }],
        );

        let reports = input_reports(13);
        let report = reports
            .iter()
            .find(|r| r.pipeline == "lighting" && r.targets == vec![out_rtv])
            .expect("lighting input report must be recorded");
        assert_eq!(report.inputs.len(), 3);
        assert_eq!(
            report.inputs[0].writer.as_ref().map(|w| w.kind),
            Some(WriteKind::Draw)
        );
        assert!(report.inputs[1].writer.is_none());
        assert!(report.inputs[1].status.contains("never host-written"));
        assert!(
            report.inputs[1]
                .status
                .contains("dropped upstream CS translation")
        );
        assert!(report.inputs[1].status.contains("e18931c72af42e46"));
        assert!(
            report.inputs[1]
                .status
                .contains("dropped upstream dispatch")
        );
        // Future write at op_index=2 must NOT be attributed to op_index=1.
        assert!(report.inputs[2].writer.is_none());
        assert!(report.inputs[2].status.contains("never host-written"));
        end_frame(13);
    }

    #[test]
    fn buffer_subrange_overlap_and_offset_chain_match_accurately() {
        let mut state = WriteState::default();
        let buf_a = 0x6000_0000usize;
        let buf_b = 0x6001_0000usize;

        state.commit(
            1,
            10,
            0,
            &[
                synthetic_buffer_write(buf_a + 64, None, WriteKind::Dispatch, 64, 0, "cs_subrange"),
                synthetic_buffer_write(
                    buf_b + 256,
                    Some(buf_a + 96),
                    WriteKind::CopyBufferRegion,
                    64,
                    1,
                    "",
                ),
            ],
        );

        // Disjoint +1024B read must miss.
        assert!(
            prior_write_from_state(&state, buf_a + 1024, true, 64, (10, 0, 5)).is_none(),
            "disjoint +1024B subrange must not match +64B write"
        );
        // Adjacent +128B read must miss half-open [64, 128).
        assert!(prior_write_from_state(&state, buf_a + 128, true, 64, (10, 0, 5)).is_none());
        // Partial overlap: [0, 256) SRV contains [64, 128) UAV write.
        let hit = prior_write_from_state(&state, buf_a, true, 256, (10, 0, 5))
            .expect("partial overlap [0,256) vs [64,128) must hit");
        assert_eq!(hit.kind, WriteKind::Dispatch);
        assert_eq!(hit.target, buf_a + 64);

        // Matching +256B read on buf_b hits CopyBufferRegion and trace_chain_from
        // infers is_buffer = true to follow buf_a + 96 back into [buf_a + 64, buf_a + 128).
        let copy_hit = prior_write_from_state(&state, buf_b + 256, true, 64, (10, 0, 5))
            .expect("matching +256B subrange must hit");
        let chain = trace_chain_from(copy_hit, 8, &state);
        assert_eq!(chain.len(), 2, "offset CopyBufferRegion chain: {chain:?}");
        assert_eq!(chain[0].kind, WriteKind::CopyBufferRegion);
        assert_eq!(chain[1].kind, WriteKind::Dispatch);
        assert_eq!(chain[1].target, buf_a + 64);

        // Point containment (size == 0) and overflow-safe boundary.
        assert!(prior_write_from_state(&state, buf_a + 96, true, 0, (10, 0, 5)).is_some());
        assert!(prior_write_from_state(&state, buf_a + 128, true, 0, (10, 0, 5)).is_none());
        let near_max = WriteRecord {
            target: usize::MAX - 32,
            ..synthetic_buffer_write(
                usize::MAX - 32,
                None,
                WriteKind::CopyBufferRegion,
                64,
                2,
                "",
            )
        };
        assert!(write_matches_input(&near_max, usize::MAX - 16, true, 64));
        assert!(!write_matches_input(&near_max, 0x1000, true, 64));
    }

    #[test]
    fn forget_resource_purges_last_and_frame_writes_across_pointer_reuse() {
        let mut state = WriteState::default();
        let tex = 0x7000_0001usize;
        let buf_base = 0x7001_0000usize;

        state.commit(
            2,
            20,
            0,
            &[
                synthetic_write(tex, None, WriteKind::ClearRtv, None, 0, ""),
                synthetic_buffer_write(
                    buf_base + 256,
                    None,
                    WriteKind::CopyBufferRegion,
                    64,
                    1,
                    "",
                ),
            ],
        );
        // End frame 1 only so frame 2 stays in frame_targets/frame_buffers too.
        state.end_frame(1);
        assert!(prior_write_from_state(&state, tex, false, 0, (21, 0, 0)).is_some());
        assert!(prior_write_from_state(&state, buf_base + 256, true, 64, (21, 0, 0)).is_some());

        // Drop texture: both last_targets and frame_targets must be cleared.
        state.forget(tex, None);
        assert!(!state.last_targets.contains_key(&tex));
        assert!(!state.frame_targets.contains_key(&tex));
        assert!(prior_write_from_state(&state, tex, false, 0, (21, 0, 0)).is_none());

        // Drop buffer [buf_base, buf_base + 4096): subrange write at +256 must be cleared.
        state.forget(0x7000_9999, Some((buf_base, 4096)));
        assert!(state.last_buffers.is_empty());
        assert!(state.frame_buffers.is_empty());
        assert!(prior_write_from_state(&state, buf_base + 256, true, 64, (21, 0, 0)).is_none());
    }

    #[test]
    fn dropped_compute_ring_evicts_oldest_and_surfaces_newest_first() {
        let mut computes = std::collections::VecDeque::new();
        let mut dispatches = std::collections::VecDeque::new();

        for i in 0..260u32 {
            let hash = format!("{i:016x}");
            push_dropped_compute(&mut computes, &hash, "compute", "unsupported");
            push_dropped_dispatch(
                &mut dispatches,
                &format!("compute {hash}"),
                "shader translation failed",
            );
        }
        assert_eq!(computes.len(), MAX_DROPPED_RECORDS);
        assert_eq!(dispatches.len(), MAX_DROPPED_RECORDS);
        assert_eq!(computes.front().unwrap().hash, format!("{:016x}", 4));
        assert_eq!(computes.back().unwrap().hash, format!("{:016x}", 259));

        let status = format_dropped_upstream_tags(&computes, &dispatches);
        assert!(
            status.contains(&format!("{:016x}", 259)),
            "newest compute failure missing from status: {status}"
        );
        assert!(
            !status.contains(&format!("{:016x}", 0)),
            "oldest compute failure should have been evicted: {status}"
        );
        let pos_259 = status.find(&format!("{:016x}", 259)).unwrap();
        let pos_258 = status.find(&format!("{:016x}", 258)).unwrap();
        assert!(pos_259 < pos_258, "most recent failure must appear first");
    }

    #[test]
    fn env_filter_parsing_strips_0x_and_honors_filters_with_present_trace() {
        assert_eq!(
            parse_filter_list(" 0x4A1348B96D845ED8 , 0Xabcd , ef01 , 0x "),
            vec!["4a1348b96d845ed8", "abcd", "ef01"]
        );
        let pso = parse_filter_list("0x1234abcd");
        let vs = parse_filter_list("0Xdeadbeef");
        assert!(!should_trace_inputs_with(
            true,
            &pso,
            &vs,
            "vertex 1111 + pixel deadbeef",
            &["1111"]
        ));
        assert!(should_trace_inputs_with(
            true,
            &pso,
            &vs,
            "vertex 1111 + pixel 2222 (1234ABCD)",
            &["1111"]
        ));
        assert!(should_trace_inputs_with(
            true,
            &pso,
            &vs,
            "vertex deadbeef + pixel 2222",
            &["deadbeef"]
        ));
        assert!(should_trace_inputs_with(true, &[], &[], "any_pso", &[]));
    }

    #[test]
    fn write_state_bounds_long_headless_runs_without_end_frame() {
        let mut state = WriteState::default();
        let tex = 0x8000_0001usize;
        let writes: Vec<WriteRecord> = (0..100u32)
            .map(|op| synthetic_write(tex, None, WriteKind::Draw, Some(0), op, "headless"))
            .collect();
        state.commit(1, 1, 0, &writes);
        assert_eq!(
            state.frame_targets.get(&tex).map(Vec::len),
            Some(MAX_TARGET_WRITES_PER_RESOURCE)
        );
        assert_eq!(state.last_targets.get(&tex).map(|w| w.op_index), Some(99));
    }
}
