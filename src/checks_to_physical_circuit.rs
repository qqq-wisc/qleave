use std::collections::BTreeMap;

use crate::{
    checks_to_physical_circuit::PhysicalGate::DeclareObservable,
    circuit_to_checks::{
        BlockKind::{self, Memory},
        CodeData, Deformation, DeformedCheckSequence, physical_supports_to_stabilizer_checks,
    },
    graph_construction::SurgeryGraphConfig,
    graph_to_checks::CorrectionSupport,
    pbc::{GraphPauli, MergedCodeQubit, Pauli, PauliAxis, PauliStringIndex, PhysicalQubit, Sign},
};

/// The qubit type gates range over while the circuit is being built: a merged-code
/// qubit (an edge qubit or a code qubit of some block). Lowered to a flat `usize`
/// for stim output by [`PhysicalCircuit::flatten`].
type MergedQubit = MergedCodeQubit<BlockKind>;

/// A physical (stim) circuit whose gates range over qubit type `Q`. Built over
/// [`MergedQubit`] and then lowered to `PhysicalCircuit<usize>` via
/// [`PhysicalCircuit::flatten`] for stim-ready output.
pub struct PhysicalCircuit<Q> {
    gates: Vec<PhysicalGate<Q>>,
}

impl<Q> PhysicalCircuit<Q> {
    fn new() -> PhysicalCircuit<Q> {
        PhysicalCircuit { gates: vec![] }
    }

    fn add_gate(&mut self, gate: PhysicalGate<Q>) {
        self.gates.push(gate);
    }
}

impl<Q: PauliStringIndex> PhysicalCircuit<Q> {
    /// Lower to a stim-ready circuit by remapping every distinct qubit to a
    /// contiguous integer index, collapsing the edge / code-qubit / per-block
    /// distinction into the single flat namespace stim expects. Qubits are numbered
    /// in order of first appearance; gate order, measurement-record offsets, and
    /// detector / observable indices are untouched (they already count in stim's
    /// terms).
    pub fn flatten(&self) -> PhysicalCircuit<usize> {
        let mut ids: BTreeMap<Q, usize> = BTreeMap::new();
        let mut intern = |q: Q| -> usize {
            let next = ids.len();
            *ids.entry(q).or_insert(next)
        };
        let gates = self
            .gates
            .iter()
            .map(|gate| flatten_gate(gate, &mut intern))
            .collect();
        PhysicalCircuit { gates }
    }
}

/// Lower a single gate to its flat-qubit form, threading `intern` through so the
/// `REPEAT` body shares the same qubit numbering as the surrounding circuit.
fn flatten_gate<Q: PauliStringIndex>(
    gate: &PhysicalGate<Q>,
    intern: &mut impl FnMut(Q) -> usize,
) -> PhysicalGate<usize> {
    match gate {
        PhysicalGate::Reset(basis, q) => PhysicalGate::Reset(*basis, intern(*q)),
        PhysicalGate::Measure(basis, q) => PhysicalGate::Measure(*basis, intern(*q)),
        PhysicalGate::XCorrection(rec, q) => PhysicalGate::XCorrection(*rec, intern(*q)),
        PhysicalGate::DeclareObservable(index, recs) => {
            PhysicalGate::DeclareObservable(*index, recs.clone())
        }
        PhysicalGate::DeclareDetector(recs) => PhysicalGate::DeclareDetector(recs.clone()),
        PhysicalGate::MPP(pauli) => PhysicalGate::MPP(pauli.map_index(&mut *intern)),
        PhysicalGate::Repeat(n, body) => {
            PhysicalGate::Repeat(*n, body.iter().map(|g| flatten_gate(g, intern)).collect())
        }
    }
}

impl<Q: std::fmt::Display + PauliStringIndex> std::fmt::Display for PhysicalCircuit<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write_gates(f, &self.gates, "")
    }
}

/// Render a gate sequence to stim text, prefixing every line with `indent`.
///
/// Coalesces a maximal run of same-kind single-qubit gates onto one stim line
/// (`R q0 q1 q2`, `M …`, `CNOT rec[-r] q …`), the way stim's flattened circuits
/// group them. Multi-target / declaration gates print one per line, and a
/// `REPEAT` block prints `REPEAT n { .. }` with its body indented one level
/// deeper.
fn write_gates<Q: std::fmt::Display + PauliStringIndex>(
    f: &mut std::fmt::Formatter<'_>,
    gates: &[PhysicalGate<Q>],
    indent: &str,
) -> std::fmt::Result {
    let mut i = 0;
    while i < gates.len() {
        match &gates[i] {
            PhysicalGate::Reset(basis, _) => {
                let basis = *basis;
                write!(f, "{indent}R{basis}")?;
                while let Some(PhysicalGate::Reset(b, q)) = gates.get(i) {
                    if *b != basis {
                        break;
                    }
                    write!(f, " {q}")?;
                    i += 1;
                }
                writeln!(f)?;
            }
            PhysicalGate::Measure(basis, _) => {
                let basis = *basis;
                write!(f, "{indent}M{basis}")?;
                while let Some(PhysicalGate::Measure(b, q)) = gates.get(i) {
                    if *b != basis {
                        break;
                    }
                    write!(f, " {q}")?;
                    i += 1;
                }
                writeln!(f)?;
            }
            PhysicalGate::XCorrection(_, _) => {
                write!(f, "{indent}CNOT")?;
                while let Some(PhysicalGate::XCorrection(rec, q)) = gates.get(i) {
                    write!(f, " rec[-{rec}] {q}")?;
                    i += 1;
                }
                writeln!(f)?;
            }
            PhysicalGate::Repeat(n, body) => {
                writeln!(f, "{indent}REPEAT {n} {{")?;
                write_gates(f, body, &format!("{indent}    "))?;
                writeln!(f, "{indent}}}")?;
                i += 1;
            }
            gate => {
                writeln!(f, "{indent}{gate}")?;
                i += 1;
            }
        }
    }
    Ok(())
}

impl<Q: std::fmt::Display + PauliStringIndex> std::fmt::Display for PhysicalGate<Q> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Measurement-record offsets count backwards from the most recent
        // measurement, so a stored step `n` renders as the stim target `rec[-n]`.
        let write_recs = |f: &mut std::fmt::Formatter<'_>, recs: &[usize]| -> std::fmt::Result {
            for r in recs {
                write!(f, " rec[-{r}]")?;
            }
            Ok(())
        };
        match self {
            PhysicalGate::Reset(basis, q) => write!(f, "R{basis} {q}"),
            PhysicalGate::Measure(basis, q) => write!(f, "M{basis} {q}"),
            PhysicalGate::XCorrection(rec, q) => write!(f, "CNOT rec[-{rec}] {q}"),
            PhysicalGate::DeclareObservable(index, recs) => {
                write!(f, "OBSERVABLE_INCLUDE({index})")?;
                write_recs(f, recs)
            }
            PhysicalGate::DeclareDetector(recs) => {
                write!(f, "DETECTOR")?;
                write_recs(f, recs)
            }
            PhysicalGate::MPP(pauli) => {
                if pauli.sign == Sign::NegOne {
                    write!(f, "MPP !")?;
                } else {
                    write!(f, "MPP ")?;
                }
                let mut first = true;
                for (q, p) in pauli.pauli_string.iter() {
                    if *p == Pauli::I {
                        continue;
                    }
                    if !first {
                        write!(f, "*")?;
                    }
                    write!(f, "{p}{q}")?;
                    first = false;
                }
                Ok(())
            }
            PhysicalGate::Repeat(n, body) => {
                writeln!(f, "REPEAT {n} {{")?;
                write_gates(f, body, "    ")?;
                write!(f, "}}")
            }
        }
    }
}

fn concatenate_circuits(
    circuits: Vec<PhysicalCircuit<MergedQubit>>,
) -> PhysicalCircuit<MergedQubit> {
    let gates = circuits.into_iter().flat_map(|c| c.gates).collect();
    PhysicalCircuit { gates }
}

#[derive(Debug, Clone)]
enum PhysicalGate<Q> {
    /// Reset a qubit into the `+1` eigenstate of the given Pauli basis, lowered to
    /// stim's `RX` / `RY` / `RZ` (`Z` is the usual `|0>`).
    Reset(Pauli, Q),
    /// A single-qubit measurement in the given Pauli basis, lowered to stim's
    /// `MX` / `MY` / `MZ`.
    Measure(Pauli, Q),
    XCorrection(usize, Q),
    DeclareObservable(usize, Vec<usize>),
    DeclareDetector(Vec<usize>),
    MPP(PauliAxis<Q>),
    /// A stim `REPEAT n { .. }` block. The body is emitted `n` times; rec offsets
    /// inside it count relative to the running measurement count, so the same body
    /// works on every iteration (and reaches back into the round emitted just
    /// before the loop on the first iteration).
    Repeat(usize, Vec<PhysicalGate<Q>>),
}

/// Lower a deformed-check sequence to a physical circuit: reset the data qubits,
/// then for each deformation emit `step0` syndrome rounds, edge-qubit
/// initialization, the merge rounds, and the split with its `XCorrection`
/// byproducts. This emits no observable annotations; a memory experiment's
/// observables are added by [`compile_memory_experiment`], which decides their
/// record combinations with the symbolic frame solver.
pub fn checks_to_physical_circuit(
    checks: DeformedCheckSequence,
    rounds: usize,
) -> PhysicalCircuit<MergedQubit> {
    let mut circuit = PhysicalCircuit::new();
    let base_checks = checks.base_checks;
    let qubits: Vec<MergedCodeQubit<BlockKind>> = base_checks
        .iter()
        .flat_map(|p| p.pauli_string.iter())
        .map(|&(q, _)| q)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    for qubit in qubits {
        circuit.add_gate(PhysicalGate::Reset(Pauli::Z, qubit));
    }
    for Deformation { checks, corrections, edge_basis: _ } in checks.deformations.into_iter() {
        let new_subcircuit =
            one_ppm_to_physical_circuit(checks, corrections, base_checks.clone(), rounds);
        circuit = concatenate_circuits(vec![circuit, new_subcircuit]);
    }
    circuit
}

fn one_ppm_to_physical_circuit(
    checks: Vec<GraphPauli<BlockKind>>,
    corrections: Vec<CorrectionSupport<BlockKind>>,
    base_checks: Vec<GraphPauli<BlockKind>>,
    rounds: usize,
) -> PhysicalCircuit<MergedQubit> {
    let step0 = d_rounds(base_checks.clone(), rounds);
    let init = initialization(&base_checks);
    let merge = d_rounds(checks.clone(), rounds);
    let split = split_and_correct(&checks, &corrections);
    concatenate_circuits(vec![step0, init, merge, split])
}

fn d_rounds(checks: Vec<GraphPauli<BlockKind>>, rounds: usize) -> PhysicalCircuit<MergedQubit> {
    let check_count = checks.len();
    let mpp_gates: Vec<PhysicalGate<MergedQubit>> = checks
        .into_iter()
        .map(|check| PhysicalGate::MPP(check))
        .collect();
    let mut output_gates = Vec::new();
    if rounds == 0 {
        return PhysicalCircuit {
            gates: output_gates,
        };
    }
    // First round establishes the reference measurements; it has no preceding
    // round to compare against, so no detectors are declared yet.
    output_gates.extend_from_slice(&mpp_gates);
    if rounds > 1 {
        // The remaining rounds are identical: re-measure every check and compare
        // it against its value one round earlier. The rec offsets are the same on
        // every iteration (each iteration adds `check_count` measurements), so a
        // single `REPEAT (rounds - 1)` body reproduces the unrolled circuit. The
        // first iteration's "previous round" is the reference round above.
        let mut body = mpp_gates;
        for j in 0..check_count {
            let this = j + 1;
            let prev = check_count + j + 1;
            body.push(PhysicalGate::DeclareDetector(vec![this, prev]));
        }
        output_gates.push(PhysicalGate::Repeat(rounds - 1, body));
    }
    PhysicalCircuit {
        gates: output_gates,
    }
}

fn initialization(checks: &Vec<GraphPauli<BlockKind>>) -> PhysicalCircuit<MergedQubit> {
    let strings = checks.iter().map(|p| &p.pauli_string);
    let edge_qubits: Vec<MergedCodeQubit<BlockKind>> = strings
        .flat_map(|s| s.iter())
        .map(|&(q, _)| q)
        .filter(|q| matches!(q, MergedCodeQubit::EdgeQubit(_)))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let gates = edge_qubits
        .iter()
        .map(|e| PhysicalGate::Reset(Pauli::Z, *e))
        .collect();
    PhysicalCircuit { gates }
}

/// Split the merged code back apart: measure every edge qubit in `Z`, then apply
/// the surgery's `X(q)` Pauli byproduct on each operator-support qubit, controlled
/// on the parity of the edge outcomes along that qubit's correction path
/// (arXiv:2410.02213, Def. 10, Stage 4). The byproduct is a real frame gate; *how*
/// each memory readout combines with the resulting records to stay deterministic
/// is determined later by the symbolic frame solver, not annotated here.
fn split_and_correct(
    checks: &Vec<GraphPauli<BlockKind>>,
    corrections: &Vec<CorrectionSupport<BlockKind>>,
) -> PhysicalCircuit<MergedQubit> {
    let strings = checks.iter().map(|p| &p.pauli_string);
    let edge_qubits: Vec<MergedCodeQubit<BlockKind>> = strings
        .flat_map(|s| s.iter())
        .map(|&(q, _)| q)
        .filter(|q| matches!(q, MergedCodeQubit::EdgeQubit(_)))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut gates: Vec<PhysicalGate<MergedQubit>> = edge_qubits
        .iter()
        .map(|e| PhysicalGate::Measure(Pauli::Z, *e))
        .collect();
    for CorrectionSupport { qubit, path } in corrections.iter() {
        for (i, edge_qubit) in edge_qubits.iter().enumerate() {
            if path.contains(edge_qubit) {
                let rec_index = edge_qubits.len() - i;
                gates.push(PhysicalGate::XCorrection(rec_index, *qubit));
            }
        }
    }
    PhysicalCircuit { gates }
}

/// Compile a full memory experiment: lower `checks` to the `rounds`-deep physical
/// circuit, then for each memory logical qubit append its transversal `basis`
/// readout and declare it as an observable, XORed with exactly the earlier
/// measurement records that make it deterministic.
///
/// Which records those are is decided by [`solve_memory_observables`], a symbolic
/// stabilizer-frame simulation of the circuit — replacing the per-deformation
/// hand bookkeeping (σ-fold, byproduct compensation), which cannot capture the
/// cross-deformation record coupling that arises when consecutive surgeries
/// anticommute. The solve runs on a single-round (`rounds == 1`) build, since
/// determinism is independent of the number of syndrome rounds; its records are
/// then mapped onto the `rounds`-deep circuit (see [`record_map`]). A logical
/// qubit the circuit leaves non-deterministic is skipped (no observable).
pub fn compile_memory_experiment(
    checks: DeformedCheckSequence,
    rounds: usize,
    code: &CodeData,
    basis: Pauli,
) -> PhysicalCircuit<MergedQubit> {
    // Single-round build for the determinism solve, deep build for output.
    let single = checks_to_physical_circuit(checks.clone(), 1);
    let deep = checks_to_physical_circuit(checks, rounds);
    append_memory_readout(&single, deep, code, basis)
}

/// Compile a *plain* memory experiment: the bare architecture code idling for
/// `rounds` syndrome rounds, then the transversal `basis` readout — no logical
/// operations, no code surgery, no deformations. This is the surgery-free baseline
/// for debugging: it exercises only the round-over-round and final-readout
/// detectors, so its circuit distance should track the code distance directly
/// (unlike [`compile_memory_experiment`], whose surgery measurements are not yet
/// fully detector-covered). See `scripts/diagnose_distance.py`.
pub fn compile_plain_memory_experiment(
    code: &CodeData,
    rounds: usize,
    basis: Pauli,
) -> PhysicalCircuit<MergedQubit> {
    // Empty supports => no deformations; we only want the lifted base checks.
    let base_checks =
        physical_supports_to_stabilizer_checks(&[], code, 1, &SurgeryGraphConfig::default()).base_checks;
    let single = plain_memory_rounds(&base_checks, 1, basis);
    let deep = plain_memory_rounds(&base_checks, rounds, basis);
    append_memory_readout(&single, deep, code, basis)
}

/// Reset every data qubit the checks touch, then measure all of them for `rounds`
/// round-over-round syndrome rounds — the surgery-free analogue of
/// [`checks_to_physical_circuit`] (which only emits rounds inside a deformation).
fn plain_memory_rounds(
    base_checks: &[GraphPauli<BlockKind>],
    rounds: usize,
    basis: Pauli,
) -> PhysicalCircuit<MergedQubit> {
    let mut circuit = PhysicalCircuit::new();
    let qubits: Vec<MergedQubit> = base_checks
        .iter()
        .flat_map(|p| p.pauli_string.iter())
        .map(|&(q, _)| q)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    // Initialize in the readout basis so the transversal readout is deterministic.
    for q in qubits {
        circuit.add_gate(PhysicalGate::Reset(basis, q));
    }
    concatenate_circuits(vec![circuit, d_rounds(base_checks.to_vec(), rounds)])
}

/// Append the transversal `basis` readout to a built memory circuit and declare
/// its observables and final-boundary detectors. `single` is the 1-round build of
/// the same circuit (used by the frame solver, whose records map into `deep` via
/// [`record_map`]); `deep` is the full `rounds`-deep build whose gates we extend.
///
/// Both the logical observables and the final detectors follow the same recipe: a
/// single transversal readout of every data qubit, then for each operator (a
/// memory logical, or a `basis`-type stabilizer) the records that pin it = its
/// readout records XOR the earlier records the solver says make it deterministic.
/// Reconstructing the stabilizers this way ties the readout into the detector
/// network; without it a late data error flips a logical with no detector firing.
fn append_memory_readout(
    single: &PhysicalCircuit<MergedQubit>,
    deep: PhysicalCircuit<MergedQubit>,
    code: &CodeData,
    basis: Pauli,
) -> PhysicalCircuit<MergedQubit> {
    // Lift a memory-block operator basis (logicals or stabilizers) to merged-code
    // qubit/Pauli supports.
    let lift = |ops: Vec<PauliAxis<PhysicalQubit>>| -> Vec<Vec<(MergedQubit, Pauli)>> {
        ops.iter()
            .map(|m| {
                m.pauli_string
                    .iter()
                    .map(|&(q, p)| {
                        (MergedCodeQubit::CodeQubit { block: Memory, index: q.0 }, p)
                    })
                    .collect()
            })
            .collect()
    };
    let supports = lift(code.memory_basis(basis));
    let stab_supports = lift(code.memory_stabilizers(basis));

    // Solve logicals and stabilizers together against the single-round circuit.
    let all_ops: Vec<Vec<(MergedQubit, Pauli)>> =
        supports.iter().chain(stab_supports.iter()).cloned().collect();
    let FrameSolve { supports: solved, determinism_detectors } =
        solve_memory_observables(single, &all_ops);
    let map = record_map(single, &deep);

    let mut gates = deep.gates;

    // One transversal `basis` readout of every data qubit appearing in any logical
    // or stabilizer support, shared by all observables and final detectors.
    let readout_qubits: Vec<MergedQubit> = all_ops
        .iter()
        .flat_map(|s| s.iter().map(|&(q, _)| q))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let pre_readout = measurement_count(&gates);
    let mut rec_abs: BTreeMap<MergedQubit, usize> = BTreeMap::new();
    for (i, &q) in readout_qubits.iter().enumerate() {
        rec_abs.insert(q, pre_readout + i);
        gates.push(PhysicalGate::Measure(basis, q));
    }
    let running = pre_readout + readout_qubits.len();

    // The rec[-k] offsets that pin operator `op`: its transversal readout records
    // plus the earlier records the frame solver says make it deterministic.
    let offsets_for = |op: &[(MergedQubit, Pauli)], records: &[usize]| -> Vec<usize> {
        let mut offs: Vec<usize> = op.iter().map(|&(q, _)| running - rec_abs[&q]).collect();
        offs.extend(records.iter().map(|&r1| running - map[r1]));
        offs
    };

    for (i, support) in supports.iter().enumerate() {
        let Some(records) = &solved[i] else { continue };
        gates.push(DeclareObservable(i, offsets_for(support, records)));
    }
    for (j, stab) in stab_supports.iter().enumerate() {
        let Some(records) = &solved[supports.len() + j] else { continue };
        gates.push(PhysicalGate::DeclareDetector(offsets_for(stab, records)));
    }
    // Cover every deterministic measurement (reference-round checks and the split
    // edge `MZ` outcomes). Each detector's records are all pre-readout, so offsets
    // are computed straight off `map`.
    for records in &determinism_detectors {
        let offs = records.iter().map(|&r| running - map[r]).collect();
        gates.push(PhysicalGate::DeclareDetector(offs));
    }
    PhysicalCircuit { gates }
}

/// Number of measurement records a gate sequence produces (recursing into
/// `Repeat`), in stim's counting: `MPP` and `Measure` each yield one.
fn measurement_count<Q>(gates: &[PhysicalGate<Q>]) -> usize {
    gates
        .iter()
        .map(|g| match g {
            PhysicalGate::MPP(_) | PhysicalGate::Measure(_, _) => 1,
            PhysicalGate::Repeat(n, body) => n * measurement_count(body),
            _ => 0,
        })
        .sum()
}

/// Map each `single`-round measurement record to its absolute index in the
/// `deep` circuit. The two circuits are gate-for-gate identical except that each
/// repeated block is a single round in `single` and `round0 + REPEAT(rounds-1)`
/// in `deep`; so a `single` record maps to the corresponding *first-round* record
/// in `deep` (deterministically equal to the later rounds), and `deep`'s extra
/// `REPEAT` blocks are skipped.
fn record_map<Q>(single: &PhysicalCircuit<Q>, deep: &PhysicalCircuit<Q>) -> Vec<usize> {
    let mut map = Vec::new();
    let mut m_deep = 0usize;
    let mut single_iter = single.gates.iter();
    for g in &deep.gates {
        match g {
            PhysicalGate::Repeat(n, body) => {
                // A repeat exists only in the deep circuit; advance past its
                // records without consuming a single-circuit gate.
                m_deep += n * measurement_count(body);
            }
            _ => {
                let s = single_iter.next().expect("single circuit shorter than deep");
                match s {
                    PhysicalGate::MPP(_) | PhysicalGate::Measure(_, _) => {
                        map.push(m_deep);
                        m_deep += 1;
                    }
                    _ => {}
                }
            }
        }
    }
    map
}

/// A symbolic stabilizer-frame simulator: an Aaronson–Gottesman tableau
/// (destabilizer rows `0..n`, stabilizer rows `n..2n`) whose per-generator sign is
/// carried *symbolically* as the GF(2) set of measurement records whose parity
/// gives it (the constant phase is dropped, since it does not affect determinism).
/// Whether bit `q` is set in a bit-packed row.
#[inline]
fn get_bit(row: &[u64], q: usize) -> bool {
    (row[q >> 6] >> (q & 63)) & 1 != 0
}

/// Set bit `q` in a bit-packed row.
#[inline]
fn set_bit(row: &mut [u64], q: usize) {
    row[q >> 6] |= 1u64 << (q & 63);
}

struct FrameSim {
    n: usize,
    /// Number of `u64` words per bit-packed row (`ceil(n / 64)`).
    words: usize,
    x: Vec<Vec<u64>>,
    z: Vec<Vec<u64>>,
    sym: Vec<std::collections::BTreeSet<usize>>,
    /// Symbolic value of each measurement record (in measurement order).
    rec: Vec<std::collections::BTreeSet<usize>>,
}

impl FrameSim {
    fn new(n: usize) -> Self {
        let words = n.div_ceil(64);
        let mut x = vec![vec![0u64; words]; 2 * n];
        let mut z = vec![vec![0u64; words]; 2 * n];
        for i in 0..n {
            set_bit(&mut x[i], i); // destabilizer i = X_i
            set_bit(&mut z[n + i], i); // stabilizer i = Z_i
        }
        FrameSim {
            n,
            words,
            x,
            z,
            sym: vec![std::collections::BTreeSet::new(); 2 * n],
            rec: Vec::new(),
        }
    }

    /// Bit-pack a `&[bool]` Pauli support into one `u64` word per 64 qubits.
    fn pack(&self, p: &[bool]) -> Vec<u64> {
        let mut w = vec![0u64; self.words];
        for (i, &b) in p.iter().enumerate() {
            if b {
                set_bit(&mut w, i);
            }
        }
        w
    }

    /// Whether tableau row `r` anticommutes with the bit-packed Pauli `(px, pz)`.
    /// The symplectic inner product is the parity of the XOR of `x[r]&pz` and
    /// `z[r]&px` accumulated word-wise, then popcounted.
    fn anticommutes(&self, r: usize, px: &[u64], pz: &[u64]) -> bool {
        let mut acc = 0u64;
        for k in 0..self.words {
            acc ^= (self.x[r][k] & pz[k]) ^ (self.z[r][k] & px[k]);
        }
        acc.count_ones() & 1 == 1
    }

    /// Left-multiply row `i` by row `j` (symplectic XOR; symbolic-sign XOR).
    fn row_mul(&mut self, i: usize, j: usize) {
        for k in 0..self.words {
            self.x[i][k] ^= self.x[j][k];
            self.z[i][k] ^= self.z[j][k];
        }
        self.sym[i] = &self.sym[i] ^ &self.sym[j];
    }

    /// Measure Pauli `(px, pz)`, assigning it the next record id and recording its
    /// symbolic value (a fresh variable if random, else the determined combination).
    fn measure(&mut self, px: &[bool], pz: &[bool]) {
        let rid = self.rec.len();
        let pxw = self.pack(px);
        let pzw = self.pack(pz);
        let anti: Vec<usize> = (self.n..2 * self.n)
            .filter(|&r| self.anticommutes(r, &pxw, &pzw))
            .collect();
        if let Some(&pivot) = anti.first() {
            // Random: isolate `pivot` as the only anticommuting row, then replace it
            // with the measured Pauli carrying a fresh record variable.
            for r in 0..2 * self.n {
                if r != pivot && self.anticommutes(r, &pxw, &pzw) {
                    self.row_mul(r, pivot);
                }
            }
            let dest = pivot - self.n;
            self.x[dest] = self.x[pivot].clone();
            self.z[dest] = self.z[pivot].clone();
            self.sym[dest] = self.sym[pivot].clone();
            self.x[pivot] = pxw;
            self.z[pivot] = pzw;
            self.sym[pivot] = std::collections::BTreeSet::from([rid]);
            self.rec.push(std::collections::BTreeSet::from([rid]));
        } else {
            // Deterministic: value = product of stabilizers whose destabilizer
            // anticommutes with the measured Pauli.
            let mut s = std::collections::BTreeSet::new();
            for d in 0..self.n {
                if self.anticommutes(d, &pxw, &pzw) {
                    s = &s ^ &self.sym[self.n + d];
                }
            }
            self.rec.push(s);
        }
    }

    /// Reset qubit `q` into the `+1` eigenstate of `basis` (`Z`->`|0>`, `X`->`|+>`):
    /// project onto `basis_q` and force its sign constant. Unlike a measurement this
    /// allocates *no* record (a reset emits none), so the record indices stay aligned
    /// with the circuit's real measurements. `basis` must be `X` or `Z`.
    fn reset(&mut self, basis: Pauli, q: usize) {
        // A row anticommutes with `basis_q` iff it carries the conjugate single-qubit
        // Pauli on `q`: for `Z_q` an `X`/`Y` (x bit set), for `X_q` a `Z`/`Y` (z bit).
        let anticommutes = |sim: &Self, r: usize| match basis {
            Pauli::X => get_bit(&sim.z[r], q),
            _ => get_bit(&sim.x[r], q),
        };
        let anti: Vec<usize> = (self.n..2 * self.n).filter(|&r| anticommutes(self, r)).collect();
        if let Some(&pivot) = anti.first() {
            for r in 0..2 * self.n {
                if r != pivot && anticommutes(self, r) {
                    self.row_mul(r, pivot);
                }
            }
            let dest = pivot - self.n;
            self.x[dest] = self.x[pivot].clone();
            self.z[dest] = self.z[pivot].clone();
            self.sym[dest] = self.sym[pivot].clone();
            self.x[pivot] = vec![0u64; self.words];
            self.z[pivot] = vec![0u64; self.words];
            match basis {
                Pauli::X => set_bit(&mut self.x[pivot], q), // stabilizer becomes X_q
                _ => set_bit(&mut self.z[pivot], q),        // stabilizer becomes Z_q
            }
            self.sym[pivot].clear(); // a reset state is deterministic +1
        } else {
            // `basis_q` is already a stabilizer; clear its single-qubit generator's sign.
            for r in self.n..2 * self.n {
                let single_qubit_basis = match basis {
                    Pauli::X => {
                        get_bit(&self.x[r], q)
                            && self.z[r].iter().all(|&w| w == 0)
                            && self.x[r].iter().map(|w| w.count_ones()).sum::<u32>() == 1
                    }
                    _ => {
                        get_bit(&self.z[r], q)
                            && self.x[r].iter().all(|&w| w == 0)
                            && self.z[r].iter().map(|w| w.count_ones()).sum::<u32>() == 1
                    }
                };
                if single_qubit_basis {
                    self.sym[r].clear();
                    break;
                }
            }
        }
    }

    /// Apply `X_q` conditioned on record `rid` (the `XCorrection` byproduct): every
    /// generator anticommuting with `X_q` (i.e. carrying `Z`/`Y` on `q`) picks up
    /// the control's symbolic value.
    fn x_correction(&mut self, rid: usize, q: usize) {
        let ctrl = self.rec[rid].clone();
        for r in 0..2 * self.n {
            if get_bit(&self.z[r], q) {
                self.sym[r] = &self.sym[r] ^ &ctrl;
            }
        }
    }

    /// The record set making the Pauli `(px, pz)` deterministic, or `None` if it
    /// anticommutes with the current stabilizer group (non-deterministic).
    fn solve(&self, px: &[bool], pz: &[bool]) -> Option<Vec<usize>> {
        let pxw = self.pack(px);
        let pzw = self.pack(pz);
        if (self.n..2 * self.n).any(|r| self.anticommutes(r, &pxw, &pzw)) {
            return None;
        }
        let mut s = std::collections::BTreeSet::new();
        for d in 0..self.n {
            if self.anticommutes(d, &pxw, &pzw) {
                s = &s ^ &self.sym[self.n + d];
            }
        }
        Some(s.into_iter().collect())
    }
}

/// The result of running the frame solver over a deformation circuit: for each
/// requested `support`, the record combination pinning its readout (`None` if
/// non-deterministic); plus, for every deterministic measurement the circuit makes,
/// the record set forming a detector on it.
struct FrameSolve {
    /// One entry per requested support, in input order.
    supports: Vec<Option<Vec<usize>>>,
    /// Record sets (single-circuit numbering) whose parity is deterministically 0:
    /// each is `{rid}` of a deterministic measurement XOR the earlier records the
    /// solver says determine it. These cover (a) the split edge `MZ` outcomes the
    /// surgery folds into the observable, and (b) the *reference* round of each
    /// `d_rounds` segment — deterministic after the Z resets but left undetected
    /// since round-over-round detectors only compare rounds 2..N. Without them a
    /// data error before round 1, or a split-`MZ` flip, is a weight-1 logical path.
    determinism_detectors: Vec<Vec<usize>>,
}

/// Run the symbolic frame simulator over `circuit` (a single-round deformation
/// circuit, before the final readout) and solve, for each memory logical's
/// transversal `support`, the record combination that pins its readout — or
/// `None` if the circuit leaves it non-deterministic. Also collects a detector for
/// every deterministic measurement so the reference rounds and the surgery's edge
/// readouts are detector-covered.
fn solve_memory_observables(
    circuit: &PhysicalCircuit<MergedQubit>,
    supports: &[Vec<(MergedQubit, Pauli)>],
) -> FrameSolve {
    // Intern every qubit (circuit gates plus readout supports) to a dense index.
    let mut index: BTreeMap<MergedQubit, usize> = BTreeMap::new();
    let mut intern = |q: MergedQubit, index: &mut BTreeMap<MergedQubit, usize>| -> usize {
        let next = index.len();
        *index.entry(q).or_insert(next)
    };
    fn collect_qubits(gates: &[PhysicalGate<MergedQubit>], out: &mut Vec<MergedQubit>) {
        for g in gates {
            match g {
                PhysicalGate::Reset(_, q)
                | PhysicalGate::Measure(_, q)
                | PhysicalGate::XCorrection(_, q) => out.push(*q),
                PhysicalGate::MPP(p) => out.extend(p.pauli_string.iter().map(|&(q, _)| q)),
                PhysicalGate::Repeat(_, body) => collect_qubits(body, out),
                _ => {}
            }
        }
    }
    let mut qubits = Vec::new();
    collect_qubits(&circuit.gates, &mut qubits);
    for support in supports {
        qubits.extend(support.iter().map(|&(q, _)| q));
    }
    for q in qubits {
        intern(q, &mut index);
    }
    let n = index.len();
    let pauli_vecs = |pairs: &[(MergedQubit, Pauli)], idx: &BTreeMap<MergedQubit, usize>| {
        let mut x = vec![false; n];
        let mut z = vec![false; n];
        for &(q, p) in pairs {
            let i = idx[&q];
            match p {
                Pauli::X => x[i] = true,
                Pauli::Z => z[i] = true,
                Pauli::Y => {
                    x[i] = true;
                    z[i] = true;
                }
                Pauli::I => {}
            }
        }
        (x, z)
    };

    let mut sim = FrameSim::new(n);
    let mut measured = 0usize;
    // Record ids of every measurement (MPP checks and split `Measure`s), to
    // detector-cover the deterministic ones afterwards.
    let mut measure_rids: Vec<usize> = Vec::new();
    for g in &circuit.gates {
        match g {
            PhysicalGate::Reset(basis, q) => sim.reset(*basis, index[q]),
            PhysicalGate::Measure(pauli, q) => {
                let (x, z) = pauli_vecs(&[(*q, *pauli)], &index);
                sim.measure(&x, &z);
                measure_rids.push(measured);
                measured += 1;
            }
            PhysicalGate::MPP(p) => {
                let pairs: Vec<(MergedQubit, Pauli)> = p.pauli_string.iter().copied().collect();
                let (x, z) = pauli_vecs(&pairs, &index);
                sim.measure(&x, &z);
                measure_rids.push(measured);
                measured += 1;
            }
            PhysicalGate::XCorrection(offset, q) => {
                // `rec[-offset]` relative to the records measured so far.
                let rid = measured - offset;
                sim.x_correction(rid, index[q]);
            }
            _ => {}
        }
    }

    let solved = supports
        .iter()
        .map(|support| {
            let (x, z) = pauli_vecs(support, &index);
            sim.solve(&x, &z)
        })
        .collect();

    // A measurement is deterministic iff its symbolic value is a combination of
    // *earlier* records (it does not carry its own fresh variable). For those, the
    // measured outcome equals the parity of `sim.rec[rid]`, so `{rid} ∪ sim.rec[rid]`
    // is a record set with deterministic parity 0 — a detector covering it.
    let determinism_detectors = measure_rids
        .into_iter()
        .filter(|&rid| !sim.rec[rid].contains(&rid))
        .map(|rid| {
            let mut set = sim.rec[rid].clone();
            set.insert(rid);
            set.into_iter().collect()
        })
        .collect();

    FrameSolve { supports: solved, determinism_detectors }
}
