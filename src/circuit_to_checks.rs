use crate::{
    graph_construction::{
        ShapeCache, SurgeryGraph, SurgeryGraphConfig, bridge_surgery_graphs, surgery_graph_cached,
    },
    graph_to_checks::{CorrectionSupport, get_correction_support, graph_to_checks, operator_edge_basis},
    pbc::{
        ArchitectureQubit::{Magic, Memory, Processor}, CodePauli, CodeQubit, GraphPauli, Pauli, PauliAxis, PauliProductCircuit, PauliProductOperation, PauliString, PhysicalPauliString, PhysicalQubit, Sign, lift_code_to_merged, pauli_string_mult
    },
};
use ndarray;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Clone)]
pub struct DeformedCheckSequence{
    pub base_checks : Vec<GraphPauli<BlockKind>>,
    pub deformations : Vec<Deformation>,
}

/// One logical Pauli measurement lowered to merged-code checks: the deformed
/// `checks`, the split-time `corrections`, and the `edge_basis` — the Pauli the
/// vertex checks carry on edge qubits (the operator type; see
/// [`operator_edge_basis`]). Downstream lowering needs `edge_basis` to tell
/// vertex checks from cycle checks and to orient the byproduct.
#[derive(Clone)]
pub struct Deformation {
    pub checks: Vec<GraphPauli<BlockKind>>,
    pub corrections: Vec<CorrectionSupport<BlockKind>>,
    pub edge_basis: Pauli,
}

/// Lower a Pauli-product measurement circuit to its deformed stabilizer-check
/// sequence. Pure: all qldpc-derived data is precomputed in `codes` (see
/// [`CodeData::from_blocks`]), so this step shells out to nothing and never fails.


pub fn physical_supports_to_stabilizer_checks(supports: &[PhysicalSupport], codes: &CodeData, distance: usize, config: &SurgeryGraphConfig) -> DeformedCheckSequence{
    let lift_stabilizers = |block: &BlockData, kind| {
        block
            .stabilizers
            .iter()
            .map(move |x| lift_physical_to_merged(x, kind))
            .collect::<Vec<_>>()
    };
    let base_checks = [
        lift_stabilizers(&codes.processor, BlockKind::Processor),
        lift_stabilizers(&codes.memory, BlockKind::Memory),
        lift_stabilizers(&codes.magic, BlockKind::Magic),
    ]
    .concat();
    let deformations = physical_supports_to_stabilizer_sets(codes, distance, &supports, config);
    DeformedCheckSequence { base_checks, deformations }
}



/// The qldpc helper script (see its header for the wire protocol), compiled in so
/// the binary carries it and needs no helper file on disk at runtime.
const LOGICAL_BASIS_PY: &str = include_str!("get_logical_basis.py");

/// Compute a basis of nontrivial logical Pauli operators for the CSS code with
/// X/Z parity checks `h_x`/`h_z`, by delegating to qldpc: `CSSCode.reduce_logical_ops`
/// with BP+OSD, then `get_logical_ops`.
///
/// qldpc returns a `(2k, 2n)` GF(2) matrix — columns `0..n` are X-type support,
/// `n..2n` Z-type — whose rows we fold into one [`PhysicalPauliString`] each
/// (`X`/`Z`/`Y`/`I` per qubit). The interpreter is taken from `$QLDPC_PYTHON`
/// (default `python3`); that environment must have `qldpc` installed.
fn get_logical_basis(block: &CSSCodeBlock, reduce: bool) -> io::Result<Vec<PhysicalPauliString>> {
    let CSSCodeBlock { hx, hz } = block;
    debug_assert_eq!(
        hx.ncols(),
        hz.ncols(),
        "hx and hz must act on the same number of physical qubits",
    );

    // The leading flag tells the helper whether to run qldpc's BP+OSD weight
    // reduction (the multi-minute step on large memory codes); it is also part
    // of the cache key, so reduced and unreduced bases never alias.
    let stdout = run_qldpc_helper(LOGICAL_BASIS_PY, |w| {
        writeln!(w, "{}", if reduce { 1 } else { 0 })?;
        write_matrix(&mut *w, hx.view())?;
        write_matrix(&mut *w, hz.view())
    })?;
    parse_logical_ops(&stdout)
}

/// The qldpc helper script that builds a named code block's parity-check matrices
/// (see its header for the wire protocol), compiled in alongside the binary.
const CSS_CODE_PY: &str = include_str!("get_css_code.py");

/// Build the [`CSSCodeBlock`] for a named code from the architecture's paper
/// (e.g. `"bb18"`, `"lp3_5_20"`, `"lp3_7_20"`, `"lp3_7_24"`) by delegating its
/// `h_x`/`h_z` construction to qldpc; see `get_css_code.py` for the supported
/// names and wire protocol. The interpreter is taken from `$QLDPC_PYTHON`
/// (default `python3`); that environment must have `qldpc` installed.
pub fn css_block(name: &str) -> io::Result<CSSCodeBlock> {
    let stdout = run_qldpc_helper(CSS_CODE_PY, |w| w.write_all(name.as_bytes()))?;
    let mut tokens = Tokens::new(&stdout)?;
    let hx = tokens.read_matrix()?;
    let hz = tokens.read_matrix()?;
    if hx.ncols() != hz.ncols() {
        return Err(io::Error::other(format!(
            "code `{name}`: h_x has {} columns but h_z has {}",
            hx.ncols(),
            hz.ncols(),
        )));
    }
    Ok(CSSCodeBlock { hx, hz })
}

/// Spawn the qldpc Python helper `script` (run as `python -c <script>`), feed it
/// the bytes written by `write_input` on stdin, and return its stdout. The
/// interpreter is `$QLDPC_PYTHON` (default `python3`); that environment must have
/// `qldpc` installed.
fn run_qldpc_helper(
    script: &str,
    write_input: impl FnOnce(&mut dyn Write) -> io::Result<()>,
) -> io::Result<Vec<u8>> {
    // Serialize the input up front so it can both seed the content-addressed
    // cache key and be fed to the child. qldpc's logical-operator reduction on
    // the large memory codes takes minutes; since `(script, input)` fully
    // determines the output, caching it turns that into a one-time cost.
    let mut input = Vec::new();
    write_input(&mut input)?;
    let cache_path = qldpc_cache_path(script, &input);
    if let Some(path) = &cache_path {
        if let Ok(cached) = std::fs::read(path) {
            eprintln!("[qldpc-cache] hit {} ({} bytes)", path.display(), cached.len());
            return Ok(cached);
        }
    }

    let python = std::env::var("QLDPC_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut child = Command::new(&python)
        .arg("-c")
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| io::Error::new(e.kind(), format!("failed to launch `{python}`: {e}")))?;

    // The helper reads stdin to EOF before emitting anything, so writing the full
    // input and dropping stdin before reading stdout cannot deadlock.
    {
        let stdin = child.stdin.take().expect("stdin was piped");
        let mut w = io::BufWriter::new(stdin);
        w.write_all(&input)?;
        w.flush()?;
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "qldpc helper (`{python}`) failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim(),
        )));
    }
    if let Some(path) = &cache_path {
        store_qldpc_cache(path, &output.stdout);
    }
    Ok(output.stdout)
}

/// Cache file for a qldpc helper invocation, content-addressed by `(script,
/// input)`. Returns `None` (caching disabled) when no cache directory can be
/// resolved or `QLDPC_CACHE_DIR` is set empty. The directory is, in order:
/// `$QLDPC_CACHE_DIR`, `$XDG_CACHE_HOME/qleave/qldpc`, or `$HOME/.cache/qleave/qldpc`.
fn qldpc_cache_path(script: &str, input: &[u8]) -> Option<PathBuf> {
    let dir = match std::env::var("QLDPC_CACHE_DIR") {
        Ok(d) if d.is_empty() => return None, // explicit opt-out
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let base = std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
            base.join("qleave").join("qldpc")
        }
    };
    // Two FNV-1a streams over the inputs in opposite orders, plus the input
    // length, give a 128-bit key whose collision odds across the handful of
    // codes here are negligible.
    let h1 = fnv1a64(&[script.as_bytes(), input]);
    let h2 = fnv1a64(&[input, script.as_bytes()]);
    Some(dir.join(format!("{h1:016x}{h2:016x}-{}", input.len())))
}

/// 64-bit FNV-1a hash of the concatenation of `parts`.
fn fnv1a64(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for &b in *part {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Best-effort write of `bytes` to the cache `path` (creating its directory),
/// via a temp file + rename so a crash can't leave a truncated entry. Cache I/O
/// failures are non-fatal — they just mean the next run recomputes.
fn store_qldpc_cache(path: &PathBuf, bytes: &[u8]) {
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Write a boolean matrix in the helper's wire format: a `rows cols` header line
/// followed by the row-major `0`/`1` entries.
fn write_matrix(w: &mut dyn Write, m: ndarray::ArrayView2<bool>) -> io::Result<()> {
    writeln!(w, "{} {}", m.nrows(), m.ncols())?;
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            if j > 0 {
                write!(w, " ")?;
            }
            write!(w, "{}", if m[[i, j]] { 1 } else { 0 })?;
        }
        writeln!(w)?;
    }
    Ok(())
}

/// A cursor over the whitespace-separated integer tokens of a helper's stdout,
/// used to read back the `rows cols <bits>` matrices of the wire protocol.
struct Tokens<'a> {
    toks: Vec<&'a str>,
    cur: usize,
}

impl<'a> Tokens<'a> {
    fn new(stdout: &'a [u8]) -> io::Result<Self> {
        let text = std::str::from_utf8(stdout)
            .map_err(|e| io::Error::other(format!("helper stdout was not UTF-8: {e}")))?;
        Ok(Self {
            toks: text.split_whitespace().collect(),
            cur: 0,
        })
    }

    fn take(&mut self, what: &str) -> io::Result<usize> {
        let s = self
            .toks
            .get(self.cur)
            .ok_or_else(|| io::Error::other(format!("helper stdout ended while reading {what}")))?;
        self.cur += 1;
        s.parse::<usize>()
            .map_err(|e| io::Error::other(format!("helper stdout: bad {what} `{s}`: {e}")))
    }

    /// Read one `rows cols` header followed by the row-major `0`/`1` entries.
    fn read_matrix(&mut self) -> io::Result<ndarray::Array2<bool>> {
        let rows = self.take("row count")?;
        let cols = self.take("column count")?;
        let mut m = ndarray::Array2::<bool>::default((rows, cols));
        for i in 0..rows {
            for j in 0..cols {
                m[[i, j]] = self.take("matrix entry")? != 0;
            }
        }
        Ok(m)
    }
}

/// Parse the helper's `(2k, 2n)` logical-ops matrix into one Pauli string per row,
/// reading qubit `q`'s X bit from column `q` and Z bit from column `n + q`.
fn parse_logical_ops(stdout: &[u8]) -> io::Result<Vec<PhysicalPauliString>> {
    let matrix = Tokens::new(stdout)?.read_matrix()?;
    let cols = matrix.ncols();
    if cols % 2 != 0 {
        return Err(io::Error::other(format!(
            "logical-ops matrix has odd width {cols}; expected 2n columns",
        )));
    }
    let n = cols / 2;

    let mut basis = Vec::with_capacity(matrix.nrows());
    for row in matrix.rows() {
        let mut pairs = Vec::new();
        for q in 0..n {
            let pauli = match (row[q], row[n + q]) {
                (false, false) => continue,
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
            };
            pairs.push((PhysicalQubit(q), pauli));
        }
        basis.push(PhysicalPauliString::new(pairs));
    }
    Ok(basis)
}

fn get_stabilizers(
    code_block: &CSSCodeBlock
) -> Vec<PhysicalPauliString> {
    let mut stabilizers = Vec::new();
    for i in 0..code_block.hx.nrows() {
        let mut pairs: Vec<(PhysicalQubit, Pauli)> = Vec::new();
        for j in 0..code_block.hx.ncols() {
            if code_block.hx[[i, j]] {
                pairs.push((PhysicalQubit(j), Pauli::X));
            }
        }
        stabilizers.push(PhysicalPauliString::new(pairs));
    }
    for i in 0..code_block.hz.nrows() {
        let mut pairs: Vec<(PhysicalQubit, Pauli)> = Vec::new();
        for j in 0..code_block.hz.ncols() {
            if code_block.hz[[i, j]] {
                pairs.push((PhysicalQubit(j), Pauli::Z));
            }
        }
        stabilizers.push(PhysicalPauliString::new(pairs));
    }
    stabilizers
}

/// Identifies one of the architecture's code blocks. Used as the port-map key for
/// [`SurgeryGraph`](crate::graph_construction::SurgeryGraph) (`Ord` for the
/// `BTreeMap`; `Copy`/no allocation, unlike a `String` name).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum BlockKind {
    Memory,
    Processor,
    Magic,
}

impl std::fmt::Display for BlockKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockKind::Memory => write!(f, "Mem"),
            BlockKind::Processor => write!(f, "Proc"),
            BlockKind::Magic => write!(f, "Magic"),
        }
    }
}

/// Which of the three blocks a measurement's support touches. A measurement is
/// either confined to a single block (an in-block measurement, no bridge needed)
/// or spans exactly two of the three blocks (bridged via lattice surgery). The
/// two-block variants carry their supports in canonical memory < processor < magic
/// order; the single-block variant carries its support and which block it is.
enum BlockSupport<'a> {
    Single(&'a PhysicalPauliString, BlockKind),
    MemoryProcessor(&'a PhysicalPauliString, &'a PhysicalPauliString),
    MemoryMagic(&'a PhysicalPauliString, &'a PhysicalPauliString),
    ProcessorMagic(&'a PhysicalPauliString, &'a PhysicalPauliString),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PhysicalSupport {
    sign: Sign,
    memory: PhysicalPauliString,
    processor: PhysicalPauliString,
    magic: PhysicalPauliString,
}

impl PhysicalSupport {
    fn new() -> Self {
        Self {
            sign: Sign::One,
            memory: PhysicalPauliString::new(vec![]),
            processor: PhysicalPauliString::new(vec![]),
            magic: PhysicalPauliString::new(vec![]),
        }
    }

    /// The blocks this support touches as a [`BlockSupport`], ready to match on
    /// directly. A measurement is confined to one block or spans two of the three,
    /// so at least one of `memory`/`processor`/`magic` is empty; panics on an empty
    /// support (touches no block) or a support spanning all three blocks.
    fn block_support(&self) -> BlockSupport<'_> {
        match (
            self.memory.is_empty(),
            self.processor.is_empty(),
            self.magic.is_empty(),
        ) {
            (false, true, true) => BlockSupport::Single(&self.memory, BlockKind::Memory),
            (true, false, true) => BlockSupport::Single(&self.processor, BlockKind::Processor),
            (true, true, false) => BlockSupport::Single(&self.magic, BlockKind::Magic),
            (false, false, true) => BlockSupport::MemoryProcessor(&self.memory, &self.processor),
            (false, true, false) => BlockSupport::MemoryMagic(&self.memory, &self.magic),
            (true, false, false) => BlockSupport::ProcessorMagic(&self.processor, &self.magic),
            (m, p, g) => panic!(
                "support must touch one or two blocks; empty = \
                 (memory: {m}, processor: {p}, magic: {g})",
            ),
        }
    }
}

pub struct CSSCodeBlock {
    hx: ndarray::Array2<bool>,
    hz: ndarray::Array2<bool>,
}

/// The per-block code data the PPM lowering consumes.
struct BlockData {
    /// `(2k, 2n)` logical-operator basis — rows `0..k` logical-X, `k..2k` logical-Z
    /// — in the layout [`logical_image`] indexes into. The qldpc `reduce_logical_ops`
    /// result: the only field whose computation shells out to Python.
    logical_basis: Vec<PhysicalPauliString>,
    /// X/Z stabilizer generators read directly off `h_x` / `h_z`.
    stabilizers: Vec<PhysicalPauliString>,
}

/// Everything derived from an architecture's three CSS code blocks, computed once.
/// Separates the impure, qldpc-shelled setup ([`CodeData::from_blocks`]) from the
/// pure lowering that consumes it ([`pauli_product_circuit_to_stabilizer_checks`]),
/// so the latter is fast to call repeatedly and testable without a Python qldpc.
pub struct CodeData {
   pub memory: BlockData,
    pub processor: BlockData,
    magic: BlockData,
}

impl CodeData {
    /// Compute each block's logical basis (via qldpc — the slow, fallible step) and
    /// stabilizer generators. This is the only part of the PPM→checks path that
    /// shells out to Python.
    pub fn from_blocks(
        memory: &CSSCodeBlock,
        processor: &CSSCodeBlock,
        magic: &CSSCodeBlock,
        reduce_max_qubits: usize,
    ) -> io::Result<CodeData> {
        // Decide per block whether to run qldpc's BP+OSD weight reduction: its cost
        // grows steeply with code size (bb18/248q ~0.1s, lp3_5_20/1122q ~50s,
        // lp3_7_20/4350q >10min), so reduce the small processor/magic blocks — for
        // lower-weight logicals and cheaper surgery — but skip the large memory code,
        // whose reduction is intractable. `reduce_max_qubits` is that cutoff in
        // physical qubits (`hx` columns).
        let block = |b: &CSSCodeBlock| -> io::Result<BlockData> {
            let reduce = b.hx.ncols() <= reduce_max_qubits;
            Ok(BlockData {
                logical_basis: get_logical_basis(b, reduce)?,
                stabilizers: get_stabilizers(b),
            })
        };
        Ok(CodeData {
            memory: block(memory)?,
            processor: block(processor)?,
            magic: block(magic)?,
        })
    }

    /// The number of logical qubits in each block, as
    /// `(memory, processor, magic)`.
    pub fn logical_qubit_counts(&self) -> (usize, usize, usize) {
        (
            self.memory.logical_basis.len() / 2,
            self.processor.logical_basis.len() / 2,
            self.magic.logical_basis.len() / 2,
        )
    }

    pub fn memory_basis(&self, basis: Pauli) -> Vec<PauliAxis<PhysicalQubit>> {
        let basis_ops = &self.memory.logical_basis;
        let half = basis_ops.len() / 2;
        let (xs, zs) = basis_ops.split_at(half);
        let lift = |s: &PhysicalPauliString| PauliAxis { sign: Sign::One, pauli_string: s.clone() };
        match basis {
            Pauli::X => xs.iter().map(lift).collect(),
            Pauli::Z => zs.iter().map(lift).collect(),
            // The logical Y on each qubit is the product of its X and Z operators
            // (which carries the i phase from pauli_string_mult).
            Pauli::Y => xs.iter().zip(zs).map(|(x, z)| pauli_string_mult(x, z)).collect(),
            Pauli::I => Vec::new(),
        }
    }

    /// The memory-block stabilizer generators of pure Pauli type `basis` (`X` or
    /// `Z`) — exactly the ones reconstructable from a transversal `basis` readout
    /// of the data qubits, used to declare the final-round boundary detectors.
    /// Returns an empty list for `Y`/`I` (no single-type CSS stabilizer matches).
    pub fn memory_stabilizers(&self, basis: Pauli) -> Vec<PauliAxis<PhysicalQubit>> {
        self.memory
            .stabilizers
            .iter()
            .filter(|s| !s.is_empty() && s.iter().all(|&(_, p)| p == basis))
            .map(|s| PauliAxis { sign: Sign::One, pauli_string: s.clone() })
            .collect()
    }
}

fn operation_to_physical_support(
    op: &PauliProductOperation,
    memory_ops: &[PhysicalPauliString],
    processor_ops: &[PhysicalPauliString],
    magic_ops: &[PhysicalPauliString],
) -> PhysicalSupport {
    let PauliProductOperation::Measurement { axis, .. } = op else {
        unreachable!("Should only be applying this conversion to measurements");
    };

    let mut acc = PhysicalSupport::new();
    for &(qubit, pauli) in axis.pauli_string.iter() {
        // Route each physical factor to its block's logical basis and the matching
        // accumulator field; the per-Pauli image is then computed identically.
        let (ops, target, q) = match qubit {
            Memory(q) => (&memory_ops, &mut acc.memory, q),
            Processor(q) => (&processor_ops, &mut acc.processor, q),
            Magic(q) => (&magic_ops, &mut acc.magic, q),
        };
        let image = logical_image(ops, q, pauli);
        let prod = pauli_string_mult(&image.pauli_string, target);
        acc.sign = acc.sign * image.sign * prod.sign;
        *target = prod.pauli_string;
    }
    acc
}

pub fn pauli_product_circuit_to_physical_supports(
    circuit: &PauliProductCircuit,
    codes: &CodeData,
) -> Vec<PhysicalSupport> {
    circuit
        .instructions
        .iter()
        .map(|op| {
            operation_to_physical_support(
                op,
                &codes.memory.logical_basis,
                &codes.processor.logical_basis,
                &codes.magic.logical_basis,
            )
        })
        .collect()
}

fn lift_physical_to_code(p: &PhysicalPauliString, block_kind: BlockKind) -> CodePauli<BlockKind> {
    CodePauli {
        sign: Sign::One,
        pauli_string: PauliString::new(
            p.iter()
                .map(|&(PhysicalQubit(q), p)| {
                    (
                        CodeQubit {
                            block: block_kind,
                            index: q,
                        },
                        p,
                    )
                })
                .collect(),
        ),
    }
}

fn lift_physical_to_merged(p: &PhysicalPauliString, block_kind: BlockKind) -> GraphPauli<BlockKind> {
    lift_code_to_merged(&lift_physical_to_code(p, block_kind))
}

/// One block of a two-block measurement: its code stabilizers, the support the
/// measured operator restricts to on this block, and which block it is.
struct BlockSurgery<'a> {
    stabilizers: &'a Vec<PhysicalPauliString>,
    support: &'a PhysicalPauliString,
    kind: BlockKind,
}

/// Two-level memoization of per-block surgery graphs, scoped to one lowering pass
/// (the blocks' stabilizers and the config are fixed within it, and the sign never
/// enters graph construction).
///
/// `exact` keys finished graphs on `(kind, support)`: per block rather than per
/// whole measurement, so a block-side piece is reused across different partners —
/// e.g. every T-gadget measures the same magic-block logical, whose graph is then
/// built once for the whole circuit. Beneath it, `shape` memoizes the expensive
/// structural stages on the support's position-space shape (see [`ShapeCache`]),
/// catching supports that differ only by relabeling — e.g. the same gadget on a
/// different logical qubit.
struct SurgeryCaches {
    exact: HashMap<(BlockKind, PhysicalPauliString), SurgeryGraph<BlockKind>>,
    shape: ShapeCache,
}

impl SurgeryCaches {
    fn new() -> Self {
        Self {
            exact: HashMap::new(),
            shape: ShapeCache::new(),
        }
    }
}

/// The surgery graph for (`kind`, `support`), built on first use and cached.
fn cached_surgery_graph<'c>(
    caches: &'c mut SurgeryCaches,
    stabilizers: &Vec<PhysicalPauliString>,
    support: &PhysicalPauliString,
    kind: BlockKind,
    config: &SurgeryGraphConfig,
) -> &'c SurgeryGraph<BlockKind> {
    let SurgeryCaches { exact, shape } = caches;
    // With caching off, rebuild unconditionally — but still store the result, so
    // the map keeps owning the graph callers borrow (e.g. [`bridged_checks`]
    // re-borrowing both sides after filling them).
    if !config.caching {
        let graph = surgery_graph_cached(stabilizers, support, kind, config, shape);
        return match exact.entry((kind, support.clone())) {
            Entry::Occupied(mut entry) => {
                entry.insert(graph);
                entry.into_mut()
            }
            Entry::Vacant(entry) => entry.insert(graph),
        };
    }
    match exact.entry((kind, support.clone())) {
        Entry::Occupied(entry) => {
            eprintln!(
                "[surgery_graph] block cache hit! I've built this exact {kind}-block support \
                 (weight {}) before.",
                support.len(),
            );
            entry.into_mut()
        }
        Entry::Vacant(entry) => {
            entry.insert(surgery_graph_cached(stabilizers, support, kind, config, shape))
        }
    }
}

/// Build the merged-code checks for a support spanning the two blocks `a` and `b`:
/// bridge their surgery graphs, lift both blocks' stabilizers into the merged code
/// (in bridge order — `a` first, then `b`, so they align with the bridged graph's
/// `path_matching`), and join the two block supports into the measured operator.
/// The third, uninvolved block is unaffected by the surgery, so its
/// `static_stabilizers` are kept verbatim in the deformation's check set.
fn bridged_checks(
    a: BlockSurgery,
    b: BlockSurgery,
    static_stabilizers : Vec<GraphPauli<BlockKind>>,
    sign: Sign,
    distance: usize,
    config: &SurgeryGraphConfig,
    graph_cache: &mut SurgeryCaches,
) -> Deformation {
    // Fill both entries first, then re-borrow immutably: the two per-block
    // graphs must be alive at once for bridging, which one `&mut` helper call
    // at a time can't provide.
    cached_surgery_graph(graph_cache, a.stabilizers, a.support, a.kind, config);
    cached_surgery_graph(graph_cache, b.stabilizers, b.support, b.kind, config);
    let a_graph = &graph_cache.exact[&(a.kind, a.support.clone())];
    let b_graph = &graph_cache.exact[&(b.kind, b.support.clone())];
    let graph = bridge_surgery_graphs(a_graph, b_graph, distance);

    let code_stabilizers: Vec<CodePauli<BlockKind>> = a
        .stabilizers
        .iter()
        .map(|s| lift_physical_to_code(s, a.kind))
        .chain(b.stabilizers.iter().map(|s| lift_physical_to_code(s, b.kind)))
        .collect();

    let a_operator = lift_physical_to_code(a.support, a.kind);
    let b_operator = lift_physical_to_code(b.support, b.kind);
    let operator = CodePauli {
        sign,
        pauli_string: PauliString::new(
            a_operator
                .pauli_string
                .iter()
                .chain(b_operator.pauli_string.iter())
                .copied()
                .collect(),
        ),
    };

    let edge_basis = operator_edge_basis(&operator);
    let mut checks = graph_to_checks(&graph, &code_stabilizers, &operator);
    checks.extend(static_stabilizers);
    let corrections = get_correction_support(&graph, &operator);
    Deformation { checks, corrections, edge_basis }
}

/// Build the merged-code checks for a support confined to the single block `b`: an
/// in-block logical measurement needs no bridge, so we build that one block's
/// surgery graph alone, lift its stabilizers and the measured operator into the
/// merged code, and read off the checks. The other two blocks (uninvolved) pass
/// through as static stabilizers.
fn single_block_checks(
    b: BlockSurgery,
    static_stabilizers: Vec<GraphPauli<BlockKind>>,
    sign: Sign,
    config: &SurgeryGraphConfig,
    graph_cache: &mut SurgeryCaches,
) -> Deformation {
    let graph = cached_surgery_graph(graph_cache, b.stabilizers, b.support, b.kind, config);

    let code_stabilizers: Vec<CodePauli<BlockKind>> = b
        .stabilizers
        .iter()
        .map(|s| lift_physical_to_code(s, b.kind))
        .collect();

    let operator = CodePauli {
        sign,
        pauli_string: lift_physical_to_code(b.support, b.kind).pauli_string,
    };

    let edge_basis = operator_edge_basis(&operator);
    let mut checks = graph_to_checks(graph, &code_stabilizers, &operator);
    checks.extend(static_stabilizers);
    let corrections = get_correction_support(graph, &operator);
    Deformation { checks, corrections, edge_basis }
}

fn physical_supports_to_stabilizer_sets(
    codes: &CodeData,
    distance: usize,
    supports: &[PhysicalSupport],
    config: &SurgeryGraphConfig,
) -> Vec<Deformation> {
    let processor_stabilizers = &codes.processor.stabilizers;
    let memory_stabilizers = &codes.memory.stabilizers;
    let magic_stabilizers = &codes.magic.stabilizers;

    let block = |stabilizers, support, kind| BlockSurgery {
        stabilizers,
        support,
        kind,
    };

    // Lift the two blocks uninvolved in a single-block measurement into static
    // merged stabilizers, picked by which block the measurement lives in.
    let lift_static = |kinds: [BlockKind; 2]| -> Vec<GraphPauli<BlockKind>> {
        kinds
            .iter()
            .flat_map(|&kind| {
                let stabs = match kind {
                    BlockKind::Memory => memory_stabilizers,
                    BlockKind::Processor => processor_stabilizers,
                    BlockKind::Magic => magic_stabilizers,
                };
                stabs.iter().map(move |s| lift_physical_to_merged(s, kind))
            })
            .collect()
    };

    // Identical supports lower to identical deformations (codes/distance/config
    // are fixed for this call), so memoize on the support: circuits routinely
    // measure the same Pauli product many times, and surgery-graph construction
    // dominates this pass.
    let mut cache: HashMap<&PhysicalSupport, Deformation> = HashMap::new();
    // Beneath that, per-block graphs are memoized separately (see
    // [`SurgeryCaches`]), so measurements that share only one side — e.g. the
    // magic piece of every T-gadget — still reuse the expensive graph
    // construction, and shape-equal supports share the structural stages.
    let mut graph_cache = SurgeryCaches::new();

    let mut stabilizers = Vec::new();
    for (i, support) in supports.iter().enumerate() {
        eprintln!(
            "[surgery_graph] lowering support: {}/{}",
            i + 1,
            supports.len()
        );
        if config.caching {
            if let Some(hit) = cache.get(support) {
                eprintln!(
                    "[surgery_graph] full cache hit! I've seen this exact Pauli product before."
                );
                stabilizers.push(hit.clone());
                continue;
            }
        }
        let deformation = match support.block_support() {
            BlockSupport::Single(support_ps, kind) => {
                let (stabs, static_kinds) = match kind {
                    BlockKind::Memory => (memory_stabilizers, [BlockKind::Processor, BlockKind::Magic]),
                    BlockKind::Processor => (processor_stabilizers, [BlockKind::Memory, BlockKind::Magic]),
                    BlockKind::Magic => (magic_stabilizers, [BlockKind::Memory, BlockKind::Processor]),
                };
                single_block_checks(
                    block(stabs, support_ps, kind),
                    lift_static(static_kinds),
                    support.sign,
                    config,
                    &mut graph_cache,
                )
            }
            BlockSupport::MemoryProcessor(memory, processor) => bridged_checks(
                block(memory_stabilizers, memory, BlockKind::Memory),
                block(processor_stabilizers, processor, BlockKind::Processor),
                magic_stabilizers.iter().map(|s| lift_physical_to_merged(s, BlockKind::Magic)).collect(),
                support.sign,
                distance,
                config,
                &mut graph_cache,
            ),
            BlockSupport::MemoryMagic(memory, magic) => bridged_checks(
                block(memory_stabilizers, memory, BlockKind::Memory),
                block(magic_stabilizers, magic, BlockKind::Magic),
                processor_stabilizers.iter().map(|s| lift_physical_to_merged(s, BlockKind::Processor)).collect(),
                support.sign,
                distance,
                config,
                &mut graph_cache,
            ),
            BlockSupport::ProcessorMagic(processor, magic) => bridged_checks(
                block(processor_stabilizers, processor, BlockKind::Processor),
                block(magic_stabilizers, magic, BlockKind::Magic),
                memory_stabilizers.iter().map(|s| lift_physical_to_merged(s, BlockKind::Memory)).collect(),
                support.sign,
                distance,
                config,
                &mut graph_cache,
            ),
        };
        if config.caching {
            cache.insert(support, deformation.clone());
        }
        stabilizers.push(deformation);
    }
    stabilizers
}

/// The logical operator a physical Pauli `pauli` on logical qubit `q` maps to,
/// read from a `(2k, 2n)` logical basis `ops` whose rows `0..k` are logical-X and
/// `k..2k` logical-Z operators. `Y = iXZ`; identity maps to the empty string.
fn logical_image(ops: &[PhysicalPauliString], q: usize, pauli: Pauli) -> PauliAxis<PhysicalQubit> {
    let k = ops.len() / 2;
    match pauli {
        Pauli::I => PauliAxis {
            sign: Sign::One,
            pauli_string: PauliString::new(vec![]),
        },
        Pauli::X => PauliAxis {
            sign: Sign::One,
            pauli_string: ops[q].clone(),
        },
        Pauli::Z => PauliAxis {
            sign: Sign::One,
            pauli_string: ops[q + k].clone(),
        },
        Pauli::Y => {
            let xz = pauli_string_mult(&ops[q], &ops[q + k]);
            PauliAxis {
                sign: xz.sign * Sign::J,
                pauli_string: xz.pauli_string,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// End-to-end check against qldpc using the Steane `[[7, 1, 3]]` code, whose
    /// X and Z checks are both the Hamming `[7, 4, 3]` parity matrix. Ignored by
    /// default since it shells out to a Python interpreter with qldpc installed;
    /// run with e.g. `QLDPC_PYTHON=/path/to/venv/bin/python cargo test -- --ignored`.
    #[test]
    #[ignore = "requires a Python interpreter with qldpc (set QLDPC_PYTHON)"]
    fn steane_logical_basis() {
        let h = array![
            [false, false, false, true, true, true, true],
            [false, true, true, false, false, true, true],
            [true, false, true, false, true, false, true],
        ];
        let block = CSSCodeBlock {
            hx: h.clone(),
            hz: h,
        };
        let basis = get_logical_basis(&block, true).expect("qldpc helper succeeds");

        // k = 1 logical qubit ⇒ one logical-X and one logical-Z operator.
        assert_eq!(basis.len(), 2);
        // The logical X is X-type (no Z/Y factors), the logical Z is Z-type.
        assert!(basis[0].iter().all(|&(_, p)| p == Pauli::X));
        assert!(basis[1].iter().all(|&(_, p)| p == Pauli::Z));
        // Distance 3: each weight-3 representative is the minimum nontrivial weight.
        assert_eq!(basis[0].len(), 3);
        assert_eq!(basis[1].len(), 3);
    }

    /// GF(2) span diagnostics for a single-block merge: which operators does the
    /// emitted check group actually measure? Used to debug the X⊗Y mis-lowering
    /// (explicit-clifford H gadget). Run with
    /// `QLDPC_PYTHON=… cargo test debug_xy_merge_span -- --ignored --nocapture`.
    #[test]
    #[ignore = "requires a Python interpreter with qldpc (set QLDPC_PYTHON)"]
    fn debug_xy_merge_span() {
        use crate::pbc::{ArchitectureQubit, MeasId, MergedCodeQubit, PauliProductOperation};

        let blocks = crate::arch::SMALL.css_blocks().expect("css blocks");
        let codes =
            CodeData::from_blocks(&blocks.0, &blocks.1, &blocks.2, 2000).expect("code data");
        let config = SurgeryGraphConfig::default();

        let support_for = |sites: Vec<(ArchitectureQubit, Pauli)>| -> PhysicalSupport {
            let op = PauliProductOperation::Measurement {
                axis: PauliAxis { sign: Sign::One, pauli_string: PauliString::new(sites) },
                id: MeasId(0),
            };
            operation_to_physical_support(
                &op,
                &codes.memory.logical_basis,
                &codes.processor.logical_basis,
                &codes.magic.logical_basis,
            )
        };

        // Bit-pack a merged-code Pauli over an interned column space: x-part then
        // z-part, one bit per (qubit, axis).
        type Q = MergedCodeQubit<BlockKind>;
        fn pack(
            p: &GraphPauli<BlockKind>,
            col: &mut std::collections::BTreeMap<Q, usize>,
        ) -> Vec<(usize, bool, bool)> {
            p.pauli_string
                .iter()
                .filter(|&&(_, pl)| pl != Pauli::I)
                .map(|&(q, pl)| {
                    let next = col.len();
                    let c = *col.entry(q).or_insert(next);
                    (c, matches!(pl, Pauli::X | Pauli::Y), matches!(pl, Pauli::Z | Pauli::Y))
                })
                .collect()
        }

        // Test span membership of `targets` in the row space of `rows` (GF(2),
        // symplectic bit representation).
        fn in_span(rows: &[Vec<(usize, bool, bool)>], target: &Vec<(usize, bool, bool)>, ncols: usize) -> bool {
            let words = (2 * ncols).div_ceil(64);
            let to_vec = |r: &Vec<(usize, bool, bool)>| -> Vec<u64> {
                let mut v = vec![0u64; words];
                for &(c, x, z) in r {
                    if x {
                        v[c >> 6] ^= 1 << (c & 63);
                    }
                    if z {
                        let zc = ncols + c;
                        v[zc >> 6] ^= 1 << (zc & 63);
                    }
                }
                v
            };
            let mut basis: Vec<(usize, Vec<u64>)> = Vec::new(); // (pivot, row)
            let reduce = |v: &mut Vec<u64>, basis: &Vec<(usize, Vec<u64>)>| {
                for (p, b) in basis {
                    if (v[p >> 6] >> (p & 63)) & 1 == 1 {
                        for k in 0..v.len() {
                            v[k] ^= b[k];
                        }
                    }
                }
            };
            for r in rows {
                let mut v = to_vec(r);
                reduce(&mut v, &basis);
                if let Some(p) = (0..2 * ncols).find(|&i| (v[i >> 6] >> (i & 63)) & 1 == 1) {
                    basis.push((p, v));
                }
            }
            let mut t = to_vec(target);
            reduce(&mut t, &basis);
            t.iter().all(|&w| w == 0)
        }

        use ArchitectureQubit::Processor as P;
        for (name, sites) in [
            ("X[P0]*Y[P9]", vec![(P(0), Pauli::X), (P(9), Pauli::Y)]),
            ("Z[P0]*Y[P9]", vec![(P(0), Pauli::Z), (P(9), Pauli::Y)]),
        ] {
            let support = support_for(sites);
            let defs =
                physical_supports_to_stabilizer_sets(&codes, 3, &[support.clone()], &config);
            let def = &defs[0];
            let mut col: std::collections::BTreeMap<Q, usize> = std::collections::BTreeMap::new();
            let rows: Vec<_> = def.checks.iter().map(|c| pack(c, &mut col)).collect();

            let merged_l = lift_physical_to_merged(&support.processor, BlockKind::Processor);
            let k = codes.processor.logical_basis.len() / 2;
            let mut probes: Vec<(String, GraphPauli<BlockKind>)> =
                vec![("L (the measured operator)".into(), merged_l)];
            for q in [0usize, 9] {
                for (axis, row) in [("X", q), ("Z", q + k)] {
                    probes.push((
                        format!("{axis}_L(P{q})"),
                        lift_physical_to_merged(
                            &codes.processor.logical_basis[row],
                            BlockKind::Processor,
                        ),
                    ));
                }
            }
            let ncols = {
                // Pre-intern all probe columns so packing is stable.
                let mut all_rows = rows.clone();
                for (_, p) in &probes {
                    all_rows.push(pack(p, &mut col));
                }
                col.len()
            };
            eprintln!("=== operator {name}: {} checks ===", def.checks.len());
            for (pname, p) in &probes {
                let t = pack(p, &mut col);
                eprintln!("  {pname} in span of checks: {}", in_span(&rows, &t, ncols));
            }
        }
    }
}
