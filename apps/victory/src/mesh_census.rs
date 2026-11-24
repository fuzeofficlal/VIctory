//! `victory mesh-census`: what the title's mesh programs are made of, for
//! running them where the console's 64-lane waves are not (every NVIDIA
//! GPU: #578, #778; and GPUs without mesh shaders: #3). It prints counts
//! only, nothing of the programs themselves, so the output can be shared.
//!
//! Per thread-group size, how many programs:
//! - run on 64-lane waves (wave64),
//! - share LDS,
//! - have `s_barrier`,
//! - read other lanes (DPP, swizzles, permutes, `v_readlane` / `v_writelane`
//!   / `v_permlane*`),
//! - count lanes with `v_mbcnt`,
//! - branch on EXEC (`s_cbranch_exec*`: where 32-lane halves of a console
//!   wave could part ways),
//! - translate as a mesh shader, and as a compute shader (the fallback).
//!
//! Then the reasons translations stop, most common first.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use victory_shader::gfx10::{self, Fields};
use victory_shader::native::StageIo;
use victory_shader::{Container, FourCC, HwStage, NativeProgram, ShaderError, native_programs};

/// What one mesh program is made of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Facts {
    pub wave64: bool,
    pub threads: u32,
    pub lds: bool,
    pub barrier: bool,
    pub lane_reads: bool,
    pub mbcnt: bool,
    pub exec_branches: bool,
    /// Translated as a mesh shader, else why not.
    pub as_mesh: Result<(), String>,
    /// Translated as compute (the fallback), else why not.
    pub as_compute: Result<(), String>,
}

/// A translation error without its code offset, so equal reasons count
/// together.
fn reason(e: &ShaderError) -> String {
    match e {
        ShaderError::Unsupported { what, .. } => (*what).to_owned(),
        ShaderError::Isa { reason, .. } => (*reason).to_owned(),
        other => other.to_string(),
    }
}

/// The facts of `native`'s code, if it is a mesh program; the translation
/// outcomes are given.
pub(crate) fn facts(
    native: &NativeProgram,
    as_mesh: Result<(), String>,
    as_compute: Result<(), String>,
) -> Option<Facts> {
    let StageIo::Ms(ms) = &native.io else {
        return None;
    };
    let mut f = Facts {
        wave64: native.wave64,
        threads: ms.threads,
        lds: ms.lds_bytes > 0,
        barrier: false,
        lane_reads: false,
        mbcnt: false,
        exec_branches: false,
        as_mesh,
        as_compute,
    };
    for i in gfx10::decode(&native.code).unwrap_or_default() {
        let name = i.name().unwrap_or("");
        f.barrier |= name == "s_barrier";
        f.mbcnt |= name.starts_with("v_mbcnt");
        f.exec_branches |= name.starts_with("s_cbranch_exec");
        f.lane_reads |= name.starts_with("ds_swizzle")
            || name.contains("permute")
            || name.starts_with("v_permlane")
            || matches!(name, "v_readlane_b32" | "v_writelane_b32")
            || matches!(i.fields(native.wave64), Fields::Valu(v) if v.dpp.is_some());
    }
    Some(f)
}

/// Programs of one thread-group size.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Row {
    programs: usize,
    wave64: usize,
    lds: usize,
    barrier: usize,
    lane_reads: usize,
    mbcnt: usize,
    exec_branches: usize,
    as_mesh: usize,
    as_compute: usize,
}

#[derive(Debug, Default)]
pub(crate) struct Census {
    rows: BTreeMap<u32, Row>,
    mesh_refused: BTreeMap<String, usize>,
    compute_refused: BTreeMap<String, usize>,
}

impl Census {
    pub(crate) fn add(&mut self, f: &Facts) {
        let r = self.rows.entry(f.threads).or_default();
        r.programs += 1;
        r.wave64 += usize::from(f.wave64);
        r.lds += usize::from(f.lds);
        r.barrier += usize::from(f.barrier);
        r.lane_reads += usize::from(f.lane_reads);
        r.mbcnt += usize::from(f.mbcnt);
        r.exec_branches += usize::from(f.exec_branches);
        r.as_mesh += usize::from(f.as_mesh.is_ok());
        r.as_compute += usize::from(f.as_compute.is_ok());
        if let Err(e) = &f.as_mesh {
            *self.mesh_refused.entry(e.clone()).or_default() += 1;
        }
        if let Err(e) = &f.as_compute {
            *self.compute_refused.entry(e.clone()).or_default() += 1;
        }
    }

    /// The table and the refusals, as text.
    pub(crate) fn report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let total: Row = self.rows.values().fold(Row::default(), |a, r| Row {
            programs: a.programs + r.programs,
            wave64: a.wave64 + r.wave64,
            lds: a.lds + r.lds,
            barrier: a.barrier + r.barrier,
            lane_reads: a.lane_reads + r.lane_reads,
            mbcnt: a.mbcnt + r.mbcnt,
            exec_branches: a.exec_branches + r.exec_branches,
            as_mesh: a.as_mesh + r.as_mesh,
            as_compute: a.as_compute + r.as_compute,
        });
        let _ = writeln!(out, "mesh programs: {}", total.programs);
        let _ = writeln!(
            out,
            "| threads | programs | wave64 | LDS | s_barrier | lane reads | v_mbcnt | EXEC branches | as mesh | as compute |"
        );
        let _ = writeln!(out, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        let line = |out: &mut String, label: String, r: &Row| {
            let _ = writeln!(
                out,
                "| {label} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                r.programs,
                r.wave64,
                r.lds,
                r.barrier,
                r.lane_reads,
                r.mbcnt,
                r.exec_branches,
                r.as_mesh,
                r.as_compute
            );
        };
        for (threads, r) in &self.rows {
            line(&mut out, threads.to_string(), r);
        }
        line(&mut out, "all".to_owned(), &total);
        for (title, refused) in [
            ("as mesh", &self.mesh_refused),
            ("as compute", &self.compute_refused),
        ] {
            let mut reasons: Vec<(&String, &usize)> = refused.iter().collect();
            reasons.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
            let _ = writeln!(
                out,
                "\nrefused {title}: {}",
                refused.values().sum::<usize>()
            );
            for (why, n) in reasons.into_iter().take(20) {
                let _ = writeln!(out, "  {n:6}  {why}");
            }
        }
        out
    }
}

pub fn run(paths: Vec<PathBuf>) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let mut census = Census::default();
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file()
                || entry.metadata().map(|m| m.len()).unwrap_or(0) > crate::shader::MAX_FILE
            {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            for source in crate::shader::sources(&data) {
                let Some(bytes) = source.bytes() else {
                    continue;
                };
                let Ok(container) = Container::parse(&bytes) else {
                    continue;
                };
                let Some(Ok(programs)) = container
                    .part(FourCC::DXIL)
                    .map(|p| native_programs(p.data))
                else {
                    continue;
                };
                let Some(ms) = programs.iter().find(|p| p.stage == HwStage::Ms) else {
                    continue;
                };
                let as_mesh = victory_shader::emit::translate(&bytes)
                    .map(drop)
                    .map_err(|e| reason(&e));
                let as_compute = victory_shader::emit::translate_mesh_as_compute(&bytes)
                    .map(drop)
                    .map_err(|e| reason(&e));
                if let Some(f) = facts(ms, as_mesh, as_compute) {
                    census.add(&f);
                }
            }
        }
    }
    print!("{}", census.report());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_shader::native::{MeshIo, VsOutput};

    const S_ENDPGM: u32 = 0xBF81_0000;
    const S_BARRIER: u32 = 0xBF8A_0000;
    /// `s_cbranch_execz` over nothing (to the next instruction).
    const S_CBRANCH_EXECZ: u32 = 0xBF88_0000;
    /// `v_mov_b32_dpp v0, v1 quad_perm:[1,0,3,2]` (VOP1 with src0 = DPP16).
    const V_MOV_DPP: [u32; 2] = [0x7E00_02FA, 0x00FF_00B1];

    fn mesh(words: &[u32], wave64: bool, threads: u32, lds_bytes: u32) -> NativeProgram {
        NativeProgram {
            stage: HwStage::Ms,
            wave64,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs: Vec::<VsOutput>::new(),
                clip_cntl: 0,
                threads,
                max_vertices: threads,
                max_primitives: threads,
                topology: 2,
                lds_bytes,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    #[test]
    fn facts_name_what_decides_the_host_form() {
        let plain = facts(&mesh(&[S_ENDPGM], true, 128, 0), Ok(()), Ok(())).unwrap();
        assert_eq!(
            (
                plain.barrier,
                plain.lane_reads,
                plain.exec_branches,
                plain.lds
            ),
            (false, false, false, false)
        );
        let busy = mesh(
            &[
                S_CBRANCH_EXECZ,
                S_BARRIER,
                V_MOV_DPP[0],
                V_MOV_DPP[1],
                S_ENDPGM,
            ],
            true,
            64,
            512,
        );
        let f = facts(&busy, Ok(()), Err("x".into())).unwrap();
        assert!(f.barrier && f.lane_reads && f.exec_branches && f.lds && f.wave64);
        assert_eq!(f.threads, 64);
        // A program of another stage is not counted.
        let mut cs = mesh(&[S_ENDPGM], true, 64, 0);
        cs.io = StageIo::Cs(victory_shader::native::ComputeIo {
            rsrc2: 0,
            threads: [64, 1, 1],
        });
        assert_eq!(facts(&cs, Ok(()), Ok(())), None);
    }

    #[test]
    fn the_report_counts_by_group_size_and_reason() {
        let mut c = Census::default();
        let f = |threads, barrier, mesh: Result<(), String>| Facts {
            wave64: true,
            threads,
            lds: barrier,
            barrier,
            lane_reads: false,
            mbcnt: false,
            exec_branches: barrier,
            as_mesh: mesh,
            as_compute: Ok(()),
        };
        c.add(&f(128, true, Ok(())));
        c.add(&f(
            128,
            false,
            Err("a cross-lane operation in a mesh shader".into()),
        ));
        c.add(&f(
            64,
            false,
            Err("a cross-lane operation in a mesh shader".into()),
        ));
        let report = c.report();
        assert!(report.starts_with("mesh programs: 3\n"), "{report}");
        assert!(
            report.contains("| 64 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |"),
            "{report}"
        );
        assert!(
            report.contains("| 128 | 2 | 2 | 1 | 1 | 0 | 0 | 1 | 1 | 2 |"),
            "{report}"
        );
        assert!(
            report.contains("| all | 3 | 3 | 1 | 1 | 0 | 0 | 1 | 1 | 3 |"),
            "{report}"
        );
        assert!(
            report
                .contains("refused as mesh: 2\n       2  a cross-lane operation in a mesh shader"),
            "{report}"
        );
        assert!(report.contains("refused as compute: 0"), "{report}");
    }
}
