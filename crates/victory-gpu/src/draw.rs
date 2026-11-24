//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, what it holds) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u64), u32>,
    /// The dynamic slots the list holds, with their cache keys so a rejected
    /// operation gives back only the tables it reserved.
    dynamic: Vec<FreshTable>,
}

struct FreshTable {
    key: (u64, u64),
    heap: usize,
    first: u32,
    count: u32,
}

impl Drop for FreshTable {
    fn drop(&mut self) {
        crate::descriptor::release_dynamic(self.heap, self.first, self.count);
    }
}

struct StageTables<'a> {
    fresh: Vec<u8>,
    made: Vec<(u8, u32)>,
    plans: Vec<FreshPlan<'a>>,
}

enum FreshPlan<'a> {
    Cached {
        table: u64,
        entries: Vec<&'a UsedResource>,
    },
    New {
        slot: u8,
        table: u64,
        heap: &'static crate::descriptor::Heap,
        first: u32,
        count: u32,
        entries: Vec<&'a UsedResource>,
    },
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as 1 MiB (both cut at their buffer's end).
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { 1 << 20 } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(texture, record.desc().as_ref()));
        }
        self.depth = 0;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        let index = if indexed {
            let v = self.index;
            if v.address == 0 || v.size == 0 {
                return false;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                return false;
            };
            Some((shadow, v))
        } else {
            None
        };
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if let Some((shadow, v)) = index {
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        const DEPTH_WRITE: u32 = 0x10;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            crate::list::ensure_state(list, self.depth as This, DEPTH_WRITE);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a table it reads could
    /// not be made.
    fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let mark = self.dynamic.len();
        let mut reserved = Vec::with_capacity(p.stages.len());
        for stage in &p.stages {
            let Some(tables) = self.reserve_stage(&stage.translated().used, compute) else {
                self.rollback_tables(mark);
                static DROPPED_TABLES: AtomicU64 = AtomicU64::new(0);
                let n = DROPPED_TABLES.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::warn!(
                        dropped = n,
                        "a draw or dispatch dropped: no dynamic slots for its tables"
                    );
                }
                return false;
            };
            reserved.push(tables);
        }
        let l = &list.list;
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for (stage, tables) in p.stages.iter().zip(&reserved) {
            let t = stage.translated();
            self.populate_stage(list, &t.used, tables, compute);
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    let fresh = tables
                        .made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        continue;
                    };
                    let Some(gpu) = root_view(list, *address, srv) else {
                        continue;
                    };
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
        }
        true
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = self.user_value(compute, u.slot, u.root, u.dword);
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        text
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = list
            .list
            .cast::<ID3D12GraphicsCommandList6>()
            .ok()
            .filter(|_| self.prepare_for(list, false, true))
            .is_some_and(|l6| {
                #[cfg(feature = "diagnostics")]
                list.note_op(|| self.op_text(format!("mesh {x}"), false));
                // SAFETY: an open list with everything set.
                unsafe { l6.DispatchMesh(x, 1, 1) };
                DRAWN.fetch_add(1, Ordering::Relaxed);
                true
            });
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u, &view);
        }
    }

    /// Reserves dynamic slots for the tables a stage needs made afresh:
    /// tables holding a constant buffer (the host wants 256-byte alignment,
    /// the console only 16) and tables reaching past a capped host heap.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once. Returns `None` without touching `list` if any fresh table has no
    /// room in the dynamic range; the caller rolls back earlier stages'
    /// reservations from this bind.
    fn reserve_stage<'a>(
        &mut self,
        used: &'a [UsedResource],
        compute: bool,
    ) -> Option<StageTables<'a>> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        let mut made = Vec::new();
        let mut plans = Vec::new();
        for &slot in &fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let Some(host) = heap.host() else {
                continue;
            };
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8 + 1)
                .max()
                .unwrap_or(1);
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    let words = if heap.index(at).is_some() {
                        // SAFETY: a descriptor slot inside `heap`.
                        unsafe { (at as *const [u32; 3]).read_unaligned() }
                    } else {
                        [0; 3]
                    };
                    words.hash(&mut h);
                }
                (table, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                made.push((slot, first * 32));
                plans.push(FreshPlan::Cached { table, entries });
                continue;
            }
            // Without its fresh table the shader would read the title's
            // slots, which for constant buffer tables hold no constant view:
            // a texture or null descriptor read as a buffer gives a wild
            // address on AMD and the device is removed (#59).
            let first = host.dynamic(n)?;
            self.fresh_tables.insert(key, first);
            self.dynamic.push(FreshTable {
                key,
                heap: std::ptr::from_ref(heap) as usize,
                first,
                count: n,
            });
            made.push((slot, first * 32));
            plans.push(FreshPlan::New {
                slot,
                table,
                heap,
                first,
                count: n,
                entries,
            });
        }
        Some(StageTables { fresh, made, plans })
    }

    /// Writes the host descriptors and prepares the resources of a stage
    /// whose fresh tables [`State::reserve_stage`] already reserved.
    fn populate_stage(
        &self,
        list: &HostList,
        used: &[UsedResource],
        tables: &StageTables<'_>,
        compute: bool,
    ) {
        for u in used.iter().filter(|u| u.root == RootType::Table) {
            if !tables.fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        for plan in &tables.plans {
            match plan {
                FreshPlan::Cached { table, entries } => {
                    let heap = self.heap_of(*table);
                    for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        if heap.is_some_and(|h| h.index(handle).is_some())
                            && let Some(view) = crate::descriptor::view(handle)
                        {
                            readable(list, u, &view);
                        }
                    }
                }
                &FreshPlan::New {
                    slot,
                    table,
                    heap,
                    first,
                    count,
                    ref entries,
                } => {
                    let Some(host) = heap.host() else {
                        continue;
                    };
                    // The whole range: a shader may index past the entries its
                    // used-resource list names (descriptor arrays).
                    for k in 0..count {
                        let index = first + k;
                        let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                        let at = table + u64::from(k) * 32;
                        match heap.index(at).and_then(|_| crate::descriptor::view(at)) {
                            Some(View::Constant(address, size)) => {
                                let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                                    host.make_null(index);
                                    continue;
                                };
                                list.defer_fill(to, address as *const u8, size as usize);
                                host.constant_at(index, gpu, size);
                            }
                            Some(view) => {
                                host.make_at(index, &view);
                                if let Some(u) = used {
                                    readable(list, u, &view);
                                }
                            }
                            None => {
                                host.make_null(index);
                                if used.is_some() {
                                    once!(slot, "a table entry without a view");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn rollback_tables(&mut self, mark: usize) {
        for table in self.dynamic.drain(mark..) {
            self.fresh_tables.remove(&table.key);
        }
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_resource(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let rects = self.topology == RECTANGLES;
        let mark = self.dynamic.len();
        let ran = (!rects || self.rectangles_ready(vertices, first))
            && self.prepare(list, false)
            && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            self.rollback_tables(mark);
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    fn rectangles_ready(&self, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        let end = u64::from(first) + u64::from(count);
        for slot in 0..32 {
            if !p.inputs.iter().any(|e| e.slot == slot && !e.per_instance) {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || end * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
        }
        true
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        let end = u64::from(first) + u64::from(count);
        let mut expanded = Vec::new();
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || end * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            expanded.push((slot, elements, v.stride, from, gpu, to));
        }
        for (slot, elements, raw_stride, from, gpu, to) in expanded {
            let stride = raw_stride as usize;
            let (from, to, n) = (from as usize, to as usize, rects as usize);
            list.defer_host(move || {
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: raw_stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        if self.topology == RECTANGLES {
            once!("indexed rectangles not drawn");
        }
        let ran = self.topology != RECTANGLES && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        args.refresh(list, arguments, len);
        if let Some((s, at)) = &counter {
            s.refresh(list, *at, 4);
        }
        // Bounded first: the clamp binds a compute pipeline of its own.
        if !crate::indirect::bound(list, &args.buffer, args.gpu(arguments), (kind, stride, max)) {
            why("not bounded (no clamp, or arguments not 4-byte aligned)");
            return false;
        }
        self.host_pipeline = 0;
        self.host_roots[1] = 0;
        let ran = match kind {
            DRAW | INDEXED => self.topology != RECTANGLES && self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => match crate::pipeline::host_pipeline(self.pipeline as This) {
                Some(p) if p.compute => self.bind(list, p, true),
                _ => false,
            },
            _ => false,
        };
        if !ran {
            crate::indirect::done(list, &args.buffer);
            #[cfg(feature = "diagnostics")]
            match crate::pipeline::host_pipeline(self.pipeline as This) {
                None => crate::pipeline::note_drop(self.pipeline as This),
                Some(p) => why(&format!("prepare failed (compute {}, mesh {})", p.compute, p.mesh)),
            }
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        // SAFETY: an open list with everything set; the buffers are host
        // copies the list keeps alive (refresh), the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &args.buffer,
                args.offset(arguments),
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        crate::indirect::done(list, &args.buffer);
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn readable(list: &HostList, u: &UsedResource, view: &View) {
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((address, len)) = crate::descriptor::host::buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        shadow.refresh(list, address, len);
        if u.kind == ResourceType::Uav {
            shadow.gpu_writes(list, address, len);
        }
    }
    match (view, u.kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            if let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            crate::list::ensure_state(list, *resource as This, SHADER_RESOURCE);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
        }
        _ => {}
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_resource(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        // A buffer has no host texture: its title range, as for a shader view.
        Some(v @ View::Unordered(r, ..)) => texture(*r as This).map_or_else(
            || format!("uav buffer {:?}", crate::descriptor::host::buffer_range(v)),
            |t| format!("uav {t}"),
        ),
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{create_cbv, create_uav, heap, test_sampler_heap, test_view_heap};
    use crate::object::slot;
    use crate::pipeline::host::GraphicsState;
    use crate::pipeline::{Shader, Translated, test_pipeline};
    use victory_shader::emit::HostShader;
    use victory_shader::native::{ComputeIo, NativeProgram, RootParam, StageIo};
    use victory_shader::{Container, HwStage};
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
    use windows::core::s;

    fn compile_hlsl(source: &[u8], target: windows::core::PCSTR) -> Vec<u8> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source slice and output pointer are valid for the call.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("test"),
                None,
                None,
                s!("main"),
                target,
                0,
                0,
                &mut code,
                None,
            )
        }
        .unwrap();
        let code = code.unwrap();
        // SAFETY: the blob owns its buffer.
        unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast(), code.GetBufferSize()) }
            .to_vec()
    }

    fn cbv_table_entries(sizes: &[(u8, u32)]) -> Vec<UsedResource> {
        sizes
            .iter()
            .map(|&(slot, count)| UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Table,
                table_offset_dwords: (count - 1) * 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot,
            })
            .collect()
    }

    fn empty_host_shader(stage: HwStage, dxil: Vec<u8>) -> HostShader {
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

    fn release_obj(this: This) {
        slot::<extern "system" fn(This) -> u32>(this, 0x10)(this);
    }

    /// Reproduces the partial-reservation failure mode (#911) and verifies
    /// that after rejecting a 4+3-slot bind under 6 free slots, the accepted
    /// 2-slot table remains intact and a subsequent 6-slot contiguous fresh
    /// table succeeds immediately.
    #[test]
    fn rejected_bind_releases_partial_tables_and_leaves_contiguous_span() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let cs = compile_hlsl(b"[numthreads(1,1,1)] void main() {}", s!("cs_5_1"));
        let p_4_3 = HostPipeline::new(
            &[Shader::for_test(
                "compute",
                Translated {
                    host: empty_host_shader(HwStage::Cs, cs.clone()),
                    used: cbv_table_entries(&[(0, 2), (1, 4), (2, 3)]),
                },
            )],
            None,
        )
        .unwrap();
        let p_6 = HostPipeline::new(
            &[Shader::for_test(
                "compute",
                Translated {
                    host: empty_host_shader(HwStage::Cs, cs),
                    used: cbv_table_entries(&[(0, 2), (1, 6)]),
                },
            )],
            None,
        )
        .unwrap();

        let title_heap = test_view_heap(16);
        let sampler_heap = test_sampler_heap(4);
        let guest = heap(title_heap);
        let view_heap = guest.host().unwrap();
        // SAFETY: a live descriptor heap.
        let dynamic_cap = unsafe { view_heap.heap.GetDesc() }.NumDescriptors - view_heap.count();

        let mut state = State::default();
        state.set_heaps(&[title_heap, sampler_heap]);
        state.set_table(true, 0, guest.start);
        assert!(state.bind(&list, &p_4_3, true));
        assert_eq!(state.dynamic.len(), 1);
        let (&accepted_key, &accepted_first) = state.fresh_tables.iter().next().unwrap();

        // Leave 6 dynamic slots free after the accepted 2-slot table.
        let blocker = view_heap.dynamic(dynamic_cap - 8).unwrap();
        state.set_table(true, 1, guest.start + 2 * 32);
        state.set_table(true, 2, guest.start + 6 * 32);
        assert!(!state.bind(&list, &p_4_3, true));
        assert_eq!(state.dynamic.len(), 1);
        assert_eq!(state.fresh_tables.len(), 1);
        assert_eq!(state.fresh_tables[&accepted_key], accepted_first);

        // All 6 contiguous slots remain allocatable as one fresh table, even
        // though the failed attempt stepped the allocator's cursor to +6.
        assert!(state.bind(&list, &p_6, true));
        assert_eq!(state.dynamic.len(), 2);
        assert_eq!(state.fresh_tables[&accepted_key], accepted_first);
        assert!(view_heap.dynamic(1).is_none());

        view_heap.release(blocker, dynamic_cap - 8);
        drop(state);
        release_obj(sampler_heap);
        release_obj(title_heap);
    }

    /// A graphics pipeline whose vertex stage reserves a fresh table before
    /// its pixel stage runs out of dynamic slots rolls back the vertex table
    /// too.
    #[test]
    fn multi_stage_graphics_rejection_rolls_back_earlier_stage_tables() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let vs_code = compile_hlsl(
            b"float4 main() : SV_Position { return float4(0, 0, 0, 1); }",
            s!("vs_5_0"),
        );
        let ps_code = compile_hlsl(
            b"float4 main() : SV_Target { return float4(1, 0, 0, 1); }",
            s!("ps_5_0"),
        );
        let vs = Shader::for_test(
            "vertex",
            Translated {
                host: empty_host_shader(HwStage::Vs, vs_code),
                used: cbv_table_entries(&[(0, 3)]),
            },
        );
        let ps = Shader::for_test(
            "pixel",
            Translated {
                host: empty_host_shader(HwStage::Ps, ps_code),
                used: cbv_table_entries(&[(1, 4)]),
            },
        );
        let mut g = GraphicsState::default();
        g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let Some(pipeline) = HostPipeline::new(&[vs, ps], Some(&g)) else {
            return;
        };

        let title_heap = test_view_heap(16);
        let sampler_heap = test_sampler_heap(4);
        let guest = heap(title_heap);
        let view_heap = guest.host().unwrap();
        // SAFETY: a live descriptor heap.
        let dynamic_cap = unsafe { view_heap.heap.GetDesc() }.NumDescriptors - view_heap.count();
        let blocker = view_heap.dynamic(dynamic_cap - 5).unwrap();

        let mut state = State::default();
        state.set_heaps(&[title_heap, sampler_heap]);
        state.set_table(false, 0, guest.start);
        state.set_table(false, 1, guest.start + 4 * 32);
        assert!(!state.bind(&list, &pipeline, false));
        assert!(state.dynamic.is_empty());
        assert!(state.fresh_tables.is_empty());

        let five = view_heap.dynamic(5).expect("all five slots remain free");
        view_heap.release(five, 5);
        view_heap.release(blocker, dynamic_cap - 5);
        drop(state);
        release_obj(sampler_heap);
        release_obj(title_heap);
    }

    /// Indexed draws without a shadowed index buffer and rectangle-list draws
    /// with invalid vertex buffers fail without retaining fresh tables.
    #[test]
    fn rejected_indexed_and_rectangle_draws_retain_no_fresh_tables() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let vs_code = compile_hlsl(
            b"float4 main(float4 p : POSITION) : SV_Position { return p; }",
            s!("vs_5_0"),
        );
        let ps_code = compile_hlsl(
            b"float4 main() : SV_Target { return float4(0, 1, 0, 1); }",
            s!("ps_5_0"),
        );
        let make_pair = || {
            vec![
                Shader::for_test(
                    "vertex",
                    Translated {
                        host: empty_host_shader(HwStage::Vs, vs_code.clone()),
                        used: cbv_table_entries(&[(0, 4)]),
                    },
                ),
                Shader::for_test(
                    "pixel",
                    Translated {
                        host: empty_host_shader(HwStage::Ps, ps_code.clone()),
                        used: Vec::new(),
                    },
                ),
            ]
        };

        let mut tri_g = Box::new(GraphicsState::default());
        tri_g.with_vec4_input();
        tri_g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        tri_g.targets.NumRenderTargets = 1;
        tri_g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let tri_pso = test_pipeline(make_pair(), Some(tri_g));

        let mut rect_g = Box::new(GraphicsState::default());
        rect_g.with_vec4_input();
        rect_g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE(19);
        rect_g.targets.NumRenderTargets = 1;
        rect_g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let rect_pso = test_pipeline(make_pair(), Some(rect_g));

        if crate::pipeline::host_pipeline(tri_pso).is_none()
            || crate::pipeline::host_pipeline(rect_pso).is_none()
        {
            release_obj(tri_pso);
            release_obj(rect_pso);
            return;
        }

        let title_heap = test_view_heap(8);
        let sampler_heap = test_sampler_heap(4);
        let guest = heap(title_heap);
        let view_heap = guest.host().unwrap();
        // SAFETY: a live descriptor heap.
        let dynamic_cap = unsafe { view_heap.heap.GetDesc() }.NumDescriptors - view_heap.count();
        let blocker = view_heap.dynamic(dynamic_cap - 4).unwrap();

        let mut state = State::default();
        state.set_heaps(&[title_heap, sampler_heap]);
        state.set_table(false, 0, guest.start);

        // Indexed draw with an unshadowed/null index buffer.
        state.set_pipeline(tri_pso);
        state.set_topology(&list, 4);
        state.set_index_buffer(None);
        state.draw_indexed(&list, 3, 1, 0, 0, 0);
        assert!(state.dynamic.is_empty());
        assert!(state.fresh_tables.is_empty());

        // Rectangle-list draw with missing vertex buffer memory.
        state.set_pipeline(rect_pso);
        state.set_topology(&list, RECTANGLES);
        state.draw(&list, 3, 1, 0, 0);
        assert!(state.dynamic.is_empty());
        assert!(state.fresh_tables.is_empty());

        let four = view_heap.dynamic(4).expect("all four slots remain free");
        view_heap.release(four, 4);
        view_heap.release(blocker, dynamic_cap - 4);
        drop(state);
        release_obj(tri_pso);
        release_obj(rect_pso);
        release_obj(sampler_heap);
        release_obj(title_heap);
    }

    /// Runs a synthetic SM 6.6 compute shader on the host GPU after a rejected
    /// multi-table bind, reading a 16-byte-aligned constant buffer from entry
    /// 5 of a reclaimed 6-slot dynamic table and copying the result back to a
    /// title UAV buffer.
    #[test]
    fn reclaimed_dynamic_table_executes_on_gpu_and_reads_back_constants() {
        crate::on_failure(|msg| panic!("host failure: {msg}"));
        let Some(host) = crate::host::host() else {
            return;
        };
        const S_ENDPGM: u32 = 0xBF81_0000;
        let s_mov_b32 =
            |sdst: u8, src0: u8| 0xBE80_0000 | u32::from(sdst) << 16 | 0x03 << 8 | u32::from(src0);
        let smem = |op: u32, dst: u8, base: u8, off: u32| {
            [
                0x3D << 26 | op << 18 | u32::from(dst) << 6 | u32::from(base >> 1),
                125 << 25 | (off & 0x1F_FFFF),
            ]
        };
        let v_mov_b32_from_s =
            |vdst: u8, ssrc: u8| 0x7E00_0000 | u32::from(vdst) << 17 | 0x01 << 9 | u32::from(ssrc);
        let buffer_store_dword = |vdata: u8, srsrc: u8| {
            [
                0x38 << 26 | 0x1C << 18,
                0x80 << 24 | (u32::from(srsrc >> 2) & 0x1F) << 16 | u32::from(vdata) << 8,
            ]
        };

        let mut words = vec![
            s_mov_b32(2, 0),
            s_mov_b32(3, 0x80),
            s_mov_b32(4, 1),
            s_mov_b32(5, 0x80),
        ];
        words.extend_from_slice(&smem(0x02, 8, 2, 5 * 32)); // s_load_dwordx4 s[8:11], s[2:3], 160
        words.extend_from_slice(&smem(0x08, 6, 8, 0)); // s_buffer_load_dword s6, s[8:11], 0
        words.push(v_mov_b32_from_s(0, 6));
        words.extend_from_slice(&smem(0x02, 12, 4, 0)); // s_load_dwordx4 s[12:15], s[4:5], 0
        words.extend_from_slice(&buffer_store_dword(0, 12));
        words.push(S_ENDPGM);

        let used = vec![
            UsedResource {
                kind: ResourceType::Cbv,
                root: RootType::Table,
                table_offset_dwords: 5 * 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 1,
                slot: 1,
            },
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 1,
                    slot: 1,
                },
            ],
            used: used.clone(),
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let ir = victory_shader::ir::build(&native.code, false, native.vgprs).unwrap();
        let st = victory_shader::structure::structure(&ir).unwrap();
        let res =
            victory_shader::resource::track(&ir, &native.layout, native.stage, Some(&st)).unwrap();
        let io = victory_shader::io::raise(&ir, &native).unwrap();
        let container = Container {
            digest: [0; 16],
            version: (1, 0),
            parts: Vec::new(),
        };
        let host_shader =
            victory_shader::emit::emit(&container, &native, &ir, &st, &io, &res).unwrap();

        // NVIDIA's SM 6.6 DXIL compiler needs more than the default 2 MiB
        // test-thread stack in debug builds.
        let run_shader = Shader::for_test(
            "compute",
            Translated {
                host: host_shader,
                used,
            },
        );
        let Some(run_pso) = std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(move || HostPipeline::new(&[run_shader], None))
            .unwrap()
            .join()
            .unwrap()
        else {
            return;
        };
        let fail_cs = compile_hlsl(b"[numthreads(1,1,1)] void main() {}", s!("cs_5_1"));
        let fail_pso = HostPipeline::new(
            &[Shader::for_test(
                "compute",
                Translated {
                    host: empty_host_shader(HwStage::Cs, fail_cs.clone()),
                    used: cbv_table_entries(&[(0, 4), (1, 3)]),
                },
            )],
            None,
        )
        .unwrap();
        let seed_pso = HostPipeline::new(
            &[Shader::for_test(
                "compute",
                Translated {
                    host: empty_host_shader(HwStage::Cs, fail_cs),
                    used: cbv_table_entries(&[(0, 2)]),
                },
            )],
            None,
        )
        .unwrap();

        // Create synthetic title CBV and UAV buffers.
        let cbv_res = crate::resource::test_buffer(4096, 0);
        let uav_res = crate::resource::test_buffer(4096, 0x4); // ALLOW_UNORDERED_ACCESS
        let (cbv_ptr, _) = crate::resource::buffer_bytes(cbv_res).unwrap();
        let (uav_ptr, _) = crate::resource::buffer_bytes(uav_res).unwrap();
        // 16-byte aligned (not 256-byte aligned) constant buffer payload.
        let cbv_addr = cbv_ptr as u64 + 16;
        // SAFETY: inside the 4096-byte buffer.
        unsafe { (cbv_addr as *mut u32).write_unaligned(0x5eed_cafe) };

        let title_heap = test_view_heap(32);
        let sampler_heap = test_sampler_heap(4);
        let guest = heap(title_heap);
        let at = |i: u64| guest.start + i * 32;
        create_cbv(
            std::ptr::null(),
            &[cbv_addr as u32, (cbv_addr >> 32) as u32, 64, 0],
            at(2 + 5),
        );
        let uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: 16,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: `D3D12_UNORDERED_ACCESS_VIEW_DESC` is 40 bytes (`[u32; 10]`).
        let uav_words = unsafe { (&raw const uav_desc).cast::<[u32; 10]>().read_unaligned() };
        create_uav(
            std::ptr::null(),
            uav_res,
            std::ptr::null(),
            &uav_words,
            at(16),
        );

        let view_heap = guest.host().unwrap();
        // SAFETY: a live descriptor heap.
        let dynamic_cap = unsafe { view_heap.heap.GetDesc() }.NumDescriptors - view_heap.count();

        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.set_heaps(&[title_heap, sampler_heap]);

        // 1. Seed a 2-slot table, then leave 6 dynamic slots free.
        state.set_table(true, 0, at(0));
        assert!(state.bind(&list, &seed_pso, true));
        assert_eq!(state.dynamic.len(), 1);
        let blocker = view_heap.dynamic(dynamic_cap - 8).unwrap();

        // 2. Reject a 4+3-slot bind; its 4-slot reservation rolls back.
        state.set_table(true, 0, at(8));
        state.set_table(true, 1, at(12));
        assert!(!state.bind(&list, &fail_pso, true));
        assert_eq!(state.dynamic.len(), 1);

        // 3. Dispatch the SM 6.6 shader using a 6-slot fresh CBV table in the
        //    reclaimed span and verify the GPU writes 0x5eed_cafe back.
        state.set_table(true, 0, at(2));
        state.set_table(true, 1, at(16));
        assert!(state.bind(&list, &run_pso, true));
        // SAFETY: `state.bind` bound the compute pipeline, heaps and user data.
        unsafe { list.list.Dispatch(1, 1, 1) };
        assert_eq!(state.dynamic.len(), 2);

        list.close();
        host.execute(&[&list]);
        assert_eq!(host.log_messages(), None);
        // SAFETY: inside the 4096-byte title UAV buffer, updated by copy-back.
        let got = unsafe { uav_ptr.cast::<u32>().read_unaligned() };
        assert_eq!(got, 0x5eed_cafe);

        view_heap.release(blocker, dynamic_cap - 8);
        drop(state);
        release_obj(cbv_res);
        release_obj(uav_res);
        release_obj(sampler_heap);
        release_obj(title_heap);
    }
}
