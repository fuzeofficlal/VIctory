//! A/B comparison of two `victory translate --list` outputs, program by
//! program.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::Path;

use anyhow::{Context, Result, ensure};

const STAGES: &[&str] = &["Ls", "Hs", "Es", "Gs", "Vs", "Ps", "Ms", "Cs"];

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Program {
    pipeline_created: bool,
    pipeline_error: Option<String>,
    validated: bool,
    validator_error: Option<String>,
    translator_errors: BTreeSet<String>,
}

impl Program {
    fn push(&mut self, outcome: &str) {
        if outcome == "pipeline created" {
            self.pipeline_created = true;
        } else if outcome == "validated" {
            self.validated = true;
        } else if outcome.starts_with("pipeline: ") {
            self.pipeline_error = Some(outcome.to_owned());
        } else if outcome.starts_with("validator: ") {
            self.validator_error = Some(outcome.to_owned());
        } else {
            self.translator_errors.insert(outcome.to_owned());
        }
    }

    fn has_pipeline(&self) -> bool {
        self.pipeline_created && self.pipeline_error.is_none() && self.translator_errors.is_empty()
    }

    fn is_validated(&self) -> bool {
        self.validated && self.validator_error.is_none() && self.translator_errors.is_empty()
    }

    fn validator_refused_pipeline(&self) -> bool {
        self.has_pipeline() && self.validator_error.is_some()
    }

    fn reason(&self) -> String {
        if !self.translator_errors.is_empty() {
            return self
                .translator_errors
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("; ");
        }
        let mut parts = Vec::new();
        if self.pipeline_created {
            parts.push("pipeline created");
        } else if let Some(err) = &self.pipeline_error {
            parts.push(err.as_str());
        }
        if self.validated {
            parts.push("validated");
        } else if let Some(err) = &self.validator_error {
            parts.push(err.as_str());
        }
        parts.join("; ")
    }
}

type Programs = BTreeMap<String, Program>;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Loss {
    key: String,
    lost_pipeline: bool,
    lost_validation: bool,
    before: String,
    after: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Report {
    both: usize,
    only_before: Vec<(String, String)>,
    only_after: Vec<(String, String)>,
    before_pipelines: usize,
    after_pipelines: usize,
    pipelines_gained: usize,
    pipelines_lost: usize,
    before_refused_pipelines: usize,
    after_refused_pipelines: usize,
    gained_refused_pipelines: usize,
    before_validations: usize,
    after_validations: usize,
    validations_gained: usize,
    validations_lost: usize,
    transitions: Vec<((String, String), usize)>,
    losses: Vec<Loss>,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "programs     {} in both", self.both)?;
        if !self.only_before.is_empty() || !self.only_after.is_empty() {
            write!(
                f,
                " ({} only in before, {} only in after)",
                self.only_before.len(),
                self.only_after.len()
            )?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "pipelines    {} before -> {} after (+{} gained, -{} lost)",
            self.before_pipelines, self.after_pipelines, self.pipelines_gained, self.pipelines_lost
        )?;
        writeln!(
            f,
            "  refused    {} before -> {} after ({} of {} gained from validator-refused modules)",
            self.before_refused_pipelines,
            self.after_refused_pipelines,
            self.gained_refused_pipelines,
            self.pipelines_gained
        )?;
        writeln!(
            f,
            "validations  {} before -> {} after (+{} gained, -{} lost)",
            self.before_validations,
            self.after_validations,
            self.validations_gained,
            self.validations_lost
        )?;

        if self.transitions.is_empty() {
            writeln!(f, "changes      0")?;
        } else {
            let changed: usize = self.transitions.iter().map(|(_, n)| n).sum();
            writeln!(
                f,
                "changes      {changed} ({} distinct transitions)",
                self.transitions.len()
            )?;
            for ((from, to), count) in &self.transitions {
                writeln!(f, "  {count:>7}  {from} -> {to}")?;
            }
        }

        if self.losses.is_empty() {
            writeln!(f, "losses       0")?;
        } else {
            writeln!(f, "losses       {}", self.losses.len())?;
            for loss in &self.losses {
                let what = match (loss.lost_pipeline, loss.lost_validation) {
                    (true, true) => "pipeline, validation",
                    (true, false) => "pipeline",
                    (false, true) => "validation",
                    (false, false) => unreachable!("loss always loses at least one check"),
                };
                writeln!(f, "  [{what}] {}", loss.key)?;
                writeln!(f, "    before: {}", loss.before)?;
                writeln!(f, "    after:  {}", loss.after)?;
            }
        }

        if !self.only_before.is_empty() {
            writeln!(f, "only in before ({})", self.only_before.len())?;
            for (k, reason) in &self.only_before {
                writeln!(f, "  {k}  {reason}")?;
            }
        }
        if !self.only_after.is_empty() {
            writeln!(f, "only in after ({})", self.only_after.len())?;
            for (k, reason) in &self.only_after {
                writeln!(f, "  {k}  {reason}")?;
            }
        }
        Ok(())
    }
}

pub fn run(before: &Path, after: &Path) -> Result<()> {
    let before_programs = read(before)?;
    let after_programs = read(after)?;
    let report = compare(&before_programs, &after_programs);
    print!("{report}");
    Ok(())
}

fn read(path: &Path) -> Result<Programs> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = decode_listing(&bytes).with_context(|| format!("decoding {}", path.display()))?;
    parse(&text).with_context(|| {
        format!(
            "no program lines found in {} (expected `victory translate --list` output)",
            path.display()
        )
    })
}

fn decode_listing(bytes: &[u8]) -> Result<Cow<'_, str>> {
    match bytes {
        [0xff, 0xfe, rest @ ..] => decode_utf16(rest, u16::from_le_bytes).map(Cow::Owned),
        [0xfe, 0xff, rest @ ..] => decode_utf16(rest, u16::from_be_bytes).map(Cow::Owned),
        _ => {
            let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(bytes);
            std::str::from_utf8(bytes)
                .map(Cow::Borrowed)
                .context("invalid UTF-8 listing")
        }
    }
}

fn decode_utf16(bytes: &[u8], order: fn([u8; 2]) -> u16) -> Result<String> {
    let (units, tail) = bytes.as_chunks::<2>();
    ensure!(tail.is_empty(), "incomplete UTF-16 code unit");
    char::decode_utf16(units.iter().copied().map(order))
        .collect::<std::result::Result<String, _>>()
        .context("invalid UTF-16 listing")
}

fn parse(text: &str) -> Option<Programs> {
    let mut programs = Programs::new();
    for line in text.lines() {
        if is_summary_or_footer(line) {
            continue;
        }
        let Some((key, outcome)) = split(line) else {
            continue;
        };
        programs.entry(key).or_default().push(outcome);
    }
    (!programs.is_empty()).then_some(programs)
}

fn is_summary_or_footer(line: &str) -> bool {
    let trimmed = line.trim_end();
    if trimmed.is_empty() {
        return true;
    }
    const HEADERS: &[&str] = &[
        "validator    ",
        "containers   ",
        "programs     ",
        "structured   ",
        "resources    ",
        "  used-resource entries ",
        "I/O          ",
        "DXIL         ",
        "             ",
        "pair ",
    ];
    if HEADERS.iter().any(|prefix| trimmed.starts_with(prefix)) {
        return true;
    }
    // Footer failure tally lines: `  {n:>7}  {reason}  (e.g. {example})`
    if let Some(rest) = trimmed.strip_prefix("  ") {
        let after_spaces = rest.trim_start_matches(' ');
        let digits = after_spaces.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0 && after_spaces[digits..].starts_with("  ") {
            return true;
        }
    }
    false
}

/// One line of the list into the program's canonical key and the line's outcome.
fn split(line: &str) -> Option<(String, &str)> {
    // Both a path and a diagnostic can contain two spaces. The boundary
    // is the first pair preceded by a complete program label.
    line.match_indices("  ").find_map(|(at, _)| {
        let (label, outcome) = line.split_at(at);
        if !ends_a_label(label) {
            return None;
        }
        Some((key(label)?, &outcome[2..]))
    })
}

/// Whether `head` ends in a stage, as a program's label does, with an
/// optional `" at 0x..."` offset stripped.
fn ends_a_label(head: &str) -> bool {
    parse_label(head).is_some()
}

fn key(label: &str) -> Option<String> {
    let (file, rest, stage) = parse_label(label)?;
    Some(format!("{file}@0x{rest} {stage}"))
}

/// Parses `<dir>[/|\]<file>@0x<hex>[ <record>] <Stage>[ at 0x<hex>]` into
/// `(file, "<hex>[ <record>]", stage)`.
fn parse_label(label: &str) -> Option<(&str, &str, &str)> {
    let without_at = match label.rsplit_once(" at 0x") {
        Some((prefix, hex)) if !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()) => {
            prefix
        }
        _ => label,
    };
    let (head, stage) = without_at.rsplit_once(' ')?;
    if !STAGES.contains(&stage) {
        return None;
    }
    let (path, rest) = head.rsplit_once("@0x")?;
    let hex = rest
        .split_once(' ')
        .map_or(rest, |(h, r)| if r.is_empty() { "" } else { h });
    if hex.is_empty() || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let file = path.rsplit(['/', '\\']).next().unwrap_or(path);
    if file.is_empty() {
        return None;
    }
    Some((file, rest, stage))
}

fn compare(before: &Programs, after: &Programs) -> Report {
    let mut report = Report::default();
    let mut transitions: BTreeMap<(String, String), usize> = BTreeMap::new();

    for p in before.values() {
        report.before_pipelines += usize::from(p.has_pipeline());
        report.before_validations += usize::from(p.is_validated());
        report.before_refused_pipelines += usize::from(p.validator_refused_pipeline());
    }
    for p in after.values() {
        report.after_pipelines += usize::from(p.has_pipeline());
        report.after_validations += usize::from(p.is_validated());
        report.after_refused_pipelines += usize::from(p.validator_refused_pipeline());
    }

    for (key, b) in before {
        let Some(a) = after.get(key) else {
            report.only_before.push((key.clone(), b.reason()));
            continue;
        };
        report.both += 1;

        let b_pipe = b.has_pipeline();
        let a_pipe = a.has_pipeline();
        let b_val = b.is_validated();
        let a_val = a.is_validated();

        if !b_pipe && a_pipe {
            report.pipelines_gained += 1;
            if a.validator_refused_pipeline() {
                report.gained_refused_pipelines += 1;
            }
        }
        if b_pipe && !a_pipe {
            report.pipelines_lost += 1;
        }
        if !b_val && a_val {
            report.validations_gained += 1;
        }
        if b_val && !a_val {
            report.validations_lost += 1;
        }

        let b_reason = b.reason();
        let a_reason = a.reason();
        if b_reason != a_reason {
            *transitions
                .entry((b_reason.clone(), a_reason.clone()))
                .or_default() += 1;
        }
        if (b_pipe && !a_pipe) || (b_val && !a_val) {
            report.losses.push(Loss {
                key: key.clone(),
                lost_pipeline: b_pipe && !a_pipe,
                lost_validation: b_val && !a_val,
                before: b_reason,
                after: a_reason,
            });
        }
    }

    for (key, a) in after {
        if !before.contains_key(key) {
            report.only_after.push((key.clone(), a.reason()));
        }
    }

    let mut sorted: Vec<_> = transitions.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    report.transitions = sorted;
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_headers_footers_offsets_and_paths() {
        let text = "\
validator    C:\\sdk\\dxil.dll
containers   4 (1 duplicates skipped)
programs     3 translated of 4 (120 IR instructions)
structured   2 of 3 (10 vector instructions, 1 guarded, 0 on opaque masks; branches 1 per lane, 2 uniform)
resources    2 of 3 tracked (descriptors 1 loaded, 0 built, 1 constant; 3 scalar accesses; 0 with a computed term)
  used-resource entries 2: 0 not traced {}; 0 traced parameters not listed
I/O          2 of 2 vertex and pixel programs raised (4 interpolations, 2 vertex values; exports {}, 0 still f16 pairs)
DXIL         2 emitted of 2 vertex and pixel programs; 1 validated, 2 pipelines
             1 emitted of 1 compute programs; 1 validated, 1 pipelines
             0 emitted of 0 mesh programs; 0 validated, 0 pipelines
pair v + p: pipeline created
E:\\copy_a\\shaders.rpf@0x70 G_Wrapped Vs  pipeline created
E:\\copy_a\\shaders.rpf@0x70 G_Wrapped Vs  validated
E:\\copy_a\\raw.dxbc@0x0 Ps at 0x1c  resources: dynamic index
E:\\copy_a\\corrupt.rpf@0x10  corrupt record
        1  resources: dynamic index  (e.g. E:\\copy_a\\raw.dxbc@0x0 Ps at 0x1c)
";
        let programs = parse(text).unwrap();
        assert_eq!(programs.len(), 2);

        let vs = &programs["shaders.rpf@0x70 G_Wrapped Vs"];
        assert!(vs.has_pipeline());
        assert!(vs.is_validated());
        assert_eq!(vs.reason(), "pipeline created; validated");

        let ps = &programs["raw.dxbc@0x0 Ps"];
        assert!(!ps.has_pipeline());
        assert!(!ps.is_validated());
        assert_eq!(ps.reason(), "resources: dynamic index");

        assert_eq!(parse("containers   6\n"), None);
        assert_eq!(parse(""), None);
    }

    #[test]
    fn unicode_boms_preserve_the_first_record() {
        let listing = "shader.rpf@0x20 Cs  pipeline: par\u{e1}metro\r\n";
        let expected = parse(listing).unwrap();
        let mut utf8_bom = vec![0xef, 0xbb, 0xbf];
        utf8_bom.extend_from_slice(listing.as_bytes());
        let mut utf16_le = vec![0xff, 0xfe];
        utf16_le.extend(listing.encode_utf16().flat_map(u16::to_le_bytes));
        let mut utf16_be = vec![0xfe, 0xff];
        utf16_be.extend(listing.encode_utf16().flat_map(u16::to_be_bytes));
        for bytes in [listing.as_bytes(), &utf8_bom, &utf16_le, &utf16_be] {
            assert_eq!(parse(&decode_listing(bytes).unwrap()).unwrap(), expected);
        }
    }

    #[test]
    fn malformed_unicode_listings_fail() {
        assert!(decode_listing(&[0xff]).is_err());
        assert!(decode_listing(&[0xff, 0xfe, 0x41]).is_err());
        assert!(decode_listing(&[0xff, 0xfe, 0x00, 0xd8]).is_err());
    }

    #[test]
    fn stage_like_path_text_does_not_hide_the_program() {
        let label = "copy Ps  files/shader.rpf@0x20 A_Wrapped Cs";
        let expected_key = "shader.rpf@0x20 A_Wrapped Cs";
        let success = format!("{label}  pipeline created");
        let message = "emit: unsupported instruction  at a test offset";
        let failure = format!("{label} at 0x2c  {message}");
        assert_eq!(
            split(&success),
            Some((expected_key.into(), "pipeline created"))
        );
        assert_eq!(split(&failure), Some((expected_key.into(), message)));
    }

    #[test]
    fn aggregates_multiple_lines_and_strips_offsets() {
        let before = parse(
            "\
C:\\run1\\a.rpf@0x10 S_Multi Ps at 0x14  resources: unsupported descriptor
C:\\run1\\a.rpf@0x10 S_Multi Ps  structure: unstructured loop
C:\\run1\\a.rpf@0x20 S_Keep Vs  pipeline created
C:\\run1\\a.rpf@0x20 S_Keep Vs  validator: bitcode error
",
        )
        .unwrap();
        let after = parse(
            "\
D:\\run2\\a.rpf@0x10 S_Multi Ps at 0x48  resources: unsupported descriptor
D:\\run2\\a.rpf@0x10 S_Multi Ps  structure: unstructured loop
D:\\run2\\a.rpf@0x20 S_Keep Vs  pipeline created
D:\\run2\\a.rpf@0x20 S_Keep Vs  validated
",
        )
        .unwrap();
        let report = compare(&before, &after);
        assert_eq!(report.both, 2);
        assert_eq!(report.before_pipelines, 1);
        assert_eq!(report.after_pipelines, 1);
        assert_eq!(report.before_refused_pipelines, 1);
        assert_eq!(report.after_refused_pipelines, 0);
        assert_eq!(report.validations_gained, 1);
        assert_eq!(report.validations_lost, 0);
        assert_eq!(
            report.transitions,
            vec![(
                (
                    "pipeline created; validator: bitcode error".into(),
                    "pipeline created; validated".into()
                ),
                1
            )]
        );
    }

    #[test]
    fn report_counts_gains_losses_refused_pipelines_and_only_one_side() {
        let before = parse(
            "\
/a/shaders.rpf@0x10 G_Gain1 Vs at 0x10  emit: missing op
/a/shaders.rpf@0x20 G_GainRefused Cs at 0x20  I/O: unsupported export
/a/shaders.rpf@0x30 G_LostToTranslator Ps  pipeline created
/a/shaders.rpf@0x30 G_LostToTranslator Ps  validated
/a/shaders.rpf@0x40 G_LostVal Cs  pipeline created
/a/shaders.rpf@0x40 G_LostVal Cs  validated
/a/shaders.rpf@0x50 G_Same Ms  pipeline created
/a/shaders.rpf@0x50 G_Same Ms  validated
/a/shaders.rpf@0x60 G_OnlyBefore Vs  pipeline created
",
        )
        .unwrap();
        let after = parse(
            "\
E:\\b\\shaders.rpf@0x10 G_Gain1 Vs  pipeline created
E:\\b\\shaders.rpf@0x10 G_Gain1 Vs  validated
E:\\b\\shaders.rpf@0x20 G_GainRefused Cs  pipeline created
E:\\b\\shaders.rpf@0x20 G_GainRefused Cs  validator: invalid metadata
E:\\b\\shaders.rpf@0x30 G_LostToTranslator Ps  unsupported: new instruction
E:\\b\\shaders.rpf@0x40 G_LostVal Cs  pipeline created
E:\\b\\shaders.rpf@0x40 G_LostVal Cs  validator: bad signature
E:\\b\\shaders.rpf@0x50 G_Same Ms  pipeline created
E:\\b\\shaders.rpf@0x50 G_Same Ms  validated
E:\\b\\shaders.rpf@0x70 G_OnlyAfter Ps  validated
",
        )
        .unwrap();

        let report = compare(&before, &after);
        assert_eq!(report.both, 5);
        assert_eq!(report.only_before.len(), 1);
        assert_eq!(report.only_after.len(), 1);
        assert_eq!(report.before_pipelines, 4);
        assert_eq!(report.after_pipelines, 4);
        assert_eq!(report.pipelines_gained, 2);
        assert_eq!(report.pipelines_lost, 1);
        assert_eq!(report.before_refused_pipelines, 0);
        assert_eq!(report.after_refused_pipelines, 2);
        assert_eq!(report.gained_refused_pipelines, 1);
        assert_eq!(report.before_validations, 3);
        assert_eq!(report.after_validations, 3);
        assert_eq!(report.validations_gained, 1);
        assert_eq!(report.validations_lost, 2);
        assert_eq!(report.transitions.len(), 4);
        assert_eq!(
            report.losses,
            vec![
                Loss {
                    key: "shaders.rpf@0x30 G_LostToTranslator Ps".into(),
                    lost_pipeline: true,
                    lost_validation: true,
                    before: "pipeline created; validated".into(),
                    after: "unsupported: new instruction".into(),
                },
                Loss {
                    key: "shaders.rpf@0x40 G_LostVal Cs".into(),
                    lost_pipeline: false,
                    lost_validation: true,
                    before: "pipeline created; validated".into(),
                    after: "pipeline created; validator: bad signature".into(),
                },
            ]
        );

        let rendered = report.to_string();
        assert!(rendered.contains("1 of 2 gained from validator-refused modules"));
        assert!(rendered.contains("[pipeline, validation] shaders.rpf@0x30 G_LostToTranslator Ps"));
        assert!(rendered.contains("[validation] shaders.rpf@0x40 G_LostVal Cs"));
        assert!(rendered.contains("only in before (1)"));
        assert!(rendered.contains("only in after (1)"));
    }
}
