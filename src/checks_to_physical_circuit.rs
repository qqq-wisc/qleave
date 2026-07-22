use std::collections::BTreeMap;
use std::io::Write;

use crate::{
    checks_to_physical_circuit::PhysicalGate::DeclareObservable,
    circuit_to_checks::{
        BlockKind::{self, Memory},
        CodeData, Deformation, DeformedCheckSequence, physical_supports_to_stabilizer_checks,
    },
    graph_construction::SurgeryGraphConfig,
    graph_to_checks::CorrectionSupport,
    pbc::{GraphPauli, MergedCodeQubit, Pauli, PauliAxis, PauliStringIndex, PhysicalQubit, Sign},
    sec_schedule::SecSchedule,
};

/// The qubit type gates range over while the circuit is being built: a merged-code
/// qubit (an edge qubit or a code qubit of some block). Lowered to a flat `usize`
/// for stim output by [`PhysicalCircuit::flatten`].
pub type MergedQubit = MergedCodeQubit<BlockKind>;

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

    /// Number of `TICK` timing layers, with `REPEAT(n)` bodies counted `n` times.
    /// On a SEC-expanded circuit this is the `sec_depth` metric fed to
    /// [`PhysicalCircuit::stats`].
    pub fn tick_count(&self) -> usize {
        count_ticks(&self.gates)
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

    /// [`PhysicalCircuit::flatten`], consuming the merged-qubit circuit. Identical
    /// output; the difference is that each source gate is dropped as soon as its
    /// flat counterpart exists, so the two representations are never both fully
    /// resident — worth it on circuits whose `MPP` Pauli strings run to gigabytes.
    pub fn into_flat(self) -> PhysicalCircuit<usize> {
        let mut ids: BTreeMap<Q, usize> = BTreeMap::new();
        let mut intern = |q: Q| -> usize {
            let next = ids.len();
            *ids.entry(q).or_insert(next)
        };
        let gates = self
            .gates
            .into_iter()
            .map(|gate| into_flat_gate(gate, &mut intern))
            .collect();
        PhysicalCircuit { gates }
    }
}

/// Per-type qubit counts and depth metrics of a compiled physical circuit, for
/// `--stats`. Computed off the [`MergedQubit`] circuit (before `flatten`), so the
/// code-qubit / edge-qubit distinction is still available.
pub struct CircuitStats {
    /// Distinct code qubits ([`MergedCodeQubit::CodeQubit`]) the circuit touches.
    pub code_qubits: usize,
    /// Distinct edge / bridge qubits ([`MergedCodeQubit::EdgeQubit`]) the circuit touches.
    pub edge_qubits: usize,
    /// Syndrome-extraction ancilla pool size = the widest syndrome round (one
    /// measurement ancilla per check, reset and reused each round). This is what
    /// [`PhysicalCircuit::expand_mpp_to_sec`] allocates; it is only physically
    /// present when the SEC gadget is actually emitted.
    pub sec_ancilla_pool: usize,
    /// Number of stabilizer-measurement rounds across the whole circuit, with
    /// `REPEAT` bodies counted by their repeat factor.
    pub syndrome_cycles: usize,
    /// Total `TICK` layers once every syndrome round is lowered to the ancilla SEC
    /// gadget, with `REPEAT` bodies counted by their repeat factor.
    pub sec_depth: usize,
}

impl CircuitStats {
    /// Summarize a compiled circuit for `--stats`: per-type qubit counts,
    /// syndrome-round count, and the SEC (ancilla-gadget) depth.
    ///
    /// Four of the five metrics are read straight off `merged` — the circuit as built,
    /// before `flatten` and before SEC expansion — because the code-qubit / edge-qubit
    /// split and the `MPP` round structure both survive only there. `sec_depth` is the
    /// one metric that lives past expansion, so the caller supplies it via
    /// [`PhysicalCircuit::tick_count`] on the SEC lowering it has already built.
    pub fn compute(merged: &PhysicalCircuit<MergedQubit>, sec_depth: usize) -> CircuitStats {
        let mut qubits = std::collections::BTreeSet::new();
        collect_merged_qubits(&merged.gates, &mut qubits);
        let (mut code_qubits, mut edge_qubits) = (0, 0);
        for q in &qubits {
            match q {
                MergedCodeQubit::EdgeQubit(_) => edge_qubits += 1,
                MergedCodeQubit::CodeQubit { .. } => code_qubits += 1,
            }
        }
        CircuitStats {
            code_qubits,
            edge_qubits,
            sec_ancilla_pool: widest_syndrome_round(&merged.gates),
            syndrome_cycles: count_syndrome_rounds(&merged.gates),
            sec_depth,
        }
    }
}

/// Collect every distinct merged-code qubit any gate touches (recursing into
/// `REPEAT` bodies).
fn collect_merged_qubits(
    gates: &[PhysicalGate<MergedQubit>],
    out: &mut std::collections::BTreeSet<MergedQubit>,
) {
    for g in gates {
        match g {
            PhysicalGate::Reset(_, q)
            | PhysicalGate::Measure(_, q, _)
            | PhysicalGate::XCorrection(_, q) => {
                out.insert(*q);
            }
            PhysicalGate::Controlled(_, c, t) => {
                out.insert(*c);
                out.insert(*t);
            }
            PhysicalGate::MPP(p) => out.extend(p.pauli_string.iter().map(|&(q, _)| q)),
            PhysicalGate::Repeat(_, body) => collect_merged_qubits(body, out),
            _ => {}
        }
    }
}

/// Number of stabilizer-measurement rounds: each maximal run of consecutive `MPP`
/// gates is one round; a `REPEAT(n)` block contributes `n` times its body's rounds.
fn count_syndrome_rounds<Q>(gates: &[PhysicalGate<Q>]) -> usize {
    let mut rounds = 0;
    let mut in_round = false;
    for g in gates {
        match g {
            PhysicalGate::MPP(_) => {
                if !in_round {
                    rounds += 1;
                    in_round = true;
                }
            }
            PhysicalGate::Repeat(n, body) => {
                in_round = false;
                rounds += n * count_syndrome_rounds(body);
            }
            _ => in_round = false,
        }
    }
    rounds
}

/// The largest number of checks in any single syndrome round (a maximal run of
/// consecutive `MPP` gates), recursing into `REPEAT` bodies. Equals the SEC ancilla
/// pool size — one ancilla per check of the widest round, reused each round.
fn widest_syndrome_round<Q>(gates: &[PhysicalGate<Q>]) -> usize {
    let mut max = 0;
    let mut cur = 0;
    for g in gates {
        match g {
            PhysicalGate::MPP(_) => {
                cur += 1;
                max = max.max(cur);
            }
            PhysicalGate::Repeat(_, body) => {
                cur = 0;
                max = max.max(widest_syndrome_round(body));
            }
            _ => cur = 0,
        }
    }
    max
}

/// Number of `TICK` gates (timing layers), with `REPEAT(n)` bodies counted `n` times.
fn count_ticks<Q>(gates: &[PhysicalGate<Q>]) -> usize {
    gates
        .iter()
        .map(|g| match g {
            PhysicalGate::Tick => 1,
            PhysicalGate::Repeat(n, body) => n * count_ticks(body),
            _ => 0,
        })
        .sum()
}

impl PhysicalCircuit<usize> {
    /// Lower every `MPP` syndrome measurement to an ancilla-based, TICK-layered
    /// syndrome-extraction gadget, leaving every other gate (and the entire
    /// measurement-record stream) untouched. Each `MPP` over sites `(q_i, P_i)` becomes
    /// `RX a` / `C{P_i} a q_i` / `MX a` on a fresh measurement ancilla `a`, which
    /// produces exactly the one record `MPP` would have — so all downstream `rec[-k]`
    /// offsets, detectors and observables remain valid without change. A negative-sign
    /// check inverts its ancilla readout (`MX !a`).
    ///
    /// Run **after** [`PhysicalCircuit::flatten`]: data qubits occupy `0..n`, so ancillas
    /// are allocated from a reused pool `n..n+pool_size` where `pool_size` is the widest
    /// syndrome round. The j-th check of *every* round uses ancilla `n + j` — one
    /// dedicated ancilla per check slot, reset and reused each round (and shared by the
    /// `REPEAT` body and the reference round).
    ///
    /// Two schedules are available (see [`crate::sec_schedule`] for the hook-cancellation
    /// theory both rest on):
    ///
    /// * [`SecSchedule::Lrc`] — staggered left–right circuits (arXiv:2603.05481):
    ///   only sitewise-*anticommuting* checks are serialized; commuting checks
    ///   interleave freely under a bipartite edge coloring, and the two conflict
    ///   classes overlap across round boundaries. Records are permuted, so every
    ///   downstream `rec[-k]` is rewritten through the resulting permutation.
    /// * [`SecSchedule::Legacy`] — the original greedy check-coloring: checks sharing
    ///   *any* data qubit run in sequential lockstep color classes (depth = sum over
    ///   classes of the largest check weight); record order is untouched. Kept for
    ///   A/B comparison.
    ///
    /// The resulting layers double as the *physical* gate cycles for a downstream
    /// circuit-level noise model — this pass injects no noise. NOTE: one ancilla per
    /// check is the textbook parity gadget and is *not necessarily distance-optimal*
    /// under that noise (hook-error orientation / flag qubits would be the refinement);
    /// that is out of scope here.
    pub fn expand_mpp_to_sec(&self, schedule: SecSchedule) -> PhysicalCircuit<usize> {
        let mut gates = Vec::new();
        self.expand_mpp_to_sec_into(schedule, &mut gates);
        PhysicalCircuit { gates }
    }

    /// [`PhysicalCircuit::expand_mpp_to_sec`], streamed: every expanded gate is
    /// handed to `out` as it is produced instead of being collected. The expansion
    /// of a large circuit is orders of magnitude bigger than the `MPP` form it comes
    /// from, so the writing path uses this with a [`StimWriter`] sink and never holds
    /// the expanded circuit at all.
    pub(crate) fn expand_mpp_to_sec_into<S: GateSink>(&self, schedule: SecSchedule, out: &mut S) {
        let base = max_qubit(&self.gates).map_or(0, |m| m + 1);
        match schedule {
            SecSchedule::Lrc => crate::sec_schedule::expand_gates_lrc_into(&self.gates, base, out),
            SecSchedule::Legacy => expand_gates_into(&self.gates, base, out),
        }
    }

    /// The `TICK` count of this circuit's SEC expansion (the `sec_depth` metric),
    /// computed by streaming the expansion through a counting sink.
    pub fn sec_tick_count(&self, schedule: SecSchedule) -> usize {
        let mut counter = TickCounter::default();
        self.expand_mpp_to_sec_into(schedule, &mut counter);
        counter.ticks
    }

    /// Stream this circuit to `out` as stim text, lowering `MPP` rounds to the
    /// ancilla SEC gadget first when `schedule` is `Some`. Returns the `TICK` count
    /// of what was written. Consumes the circuit: with no SEC expansion the gates are
    /// handed to the writer by value, so they are freed as they are rendered.
    pub fn write_stim<W: Write>(
        self,
        schedule: Option<SecSchedule>,
        out: W,
    ) -> std::io::Result<usize> {
        let mut writer = StimWriter::new(out);
        match schedule {
            Some(schedule) => self.expand_mpp_to_sec_into(schedule, &mut writer),
            None => {
                for gate in self.gates {
                    writer.push(gate);
                }
            }
        }
        writer.finish()
    }
}

/// The largest flat qubit id any gate touches (recursing into `REPEAT` bodies), or
/// `None` for a qubit-free circuit. The ancilla pool starts one past this.
fn max_qubit(gates: &[PhysicalGate<usize>]) -> Option<usize> {
    fn bump(max: &mut Option<usize>, q: usize) {
        *max = Some(max.map_or(q, |m| m.max(q)));
    }
    let mut max = None;
    for g in gates {
        match g {
            PhysicalGate::Reset(_, q)
            | PhysicalGate::Measure(_, q, _)
            | PhysicalGate::XCorrection(_, q) => bump(&mut max, *q),
            PhysicalGate::Controlled(_, c, t) => {
                bump(&mut max, *c);
                bump(&mut max, *t);
            }
            PhysicalGate::MPP(p) => {
                for &(q, _) in p.pauli_string.iter() {
                    bump(&mut max, q);
                }
            }
            PhysicalGate::Repeat(_, body) => {
                if let Some(m) = max_qubit(body) {
                    bump(&mut max, m);
                }
            }
            _ => {}
        }
    }
    max
}

/// Expand the `MPP`s in a flat gate list, recursing into `REPEAT` bodies (which reuse
/// the same ancilla pool `base..`). Consecutive `MPP`s form one syndrome round; any
/// other gate flushes the pending round and is emitted unchanged.
fn expand_gates_into<S: GateSink>(gates: &[PhysicalGate<usize>], base: usize, out: &mut S) {
    let mut round: Vec<&PauliAxis<usize>> = Vec::new();
    for g in gates {
        match g {
            PhysicalGate::MPP(p) => round.push(p),
            other => {
                if !round.is_empty() {
                    emit_round(&round, base, out);
                    round.clear();
                }
                match other {
                    PhysicalGate::Repeat(n, body) => {
                        // A `REPEAT` body is one round, so it is small enough to
                        // materialize — the gate it becomes has to carry it anyway.
                        let mut expanded = Vec::new();
                        expand_gates_into(body, base, &mut expanded);
                        out.push(PhysicalGate::Repeat(*n, expanded));
                    }
                    g => out.push(g.clone()),
                }
            }
        }
    }
    if !round.is_empty() {
        emit_round(&round, base, out);
    }
}

/// Emit one syndrome round of `checks` as TICK-layered ancilla gadgets: a reset layer
/// (`RX` on each check's ancilla), the edge-colored controlled-Pauli layers, then a
/// measure layer (`MX`, inverted for a negative-sign check). The j-th check uses ancilla
/// `base + j`; measurements are emitted in check order, preserving the record stream.
fn emit_round<S: GateSink>(checks: &[&PauliAxis<usize>], base: usize, out: &mut S) {
    let m = checks.len();

    // Reset layer.
    for j in 0..m {
        out.push(PhysicalGate::Reset(Pauli::X, base + j));
    }
    out.push(PhysicalGate::Tick);

    // Greedy *check* coloring: checks sharing a data qubit get different colors, so each
    // color class is a set of mutually qubit-disjoint checks. We must color checks (not
    // individual gates): hook cancellation requires every check's controlled-Paulis to
    // stay *contiguous* relative to any check it shares a qubit with. Two commuting CSS
    // checks overlap on an even number of qubits, so when A's gates are all-before (or
    // all-after) B's the two `Z`-kickbacks onto B's ancilla cancel; interleaving A and B
    // on a shared qubit leaves one kickback and corrupts the measurement. Disjoint checks
    // (same class) never share a qubit, so they may run concurrently; shared-qubit checks
    // land in different classes and so run in disjoint, sequential time spans.
    let mut color = vec![0usize; m];
    let mut qubit_colors: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
        std::collections::HashMap::new();
    for (j, check) in checks.iter().enumerate() {
        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &(q, p) in check.pauli_string.iter() {
            if p != Pauli::I {
                if let Some(cs) = qubit_colors.get(&q) {
                    used.extend(cs.iter().copied());
                }
            }
        }
        let mut c = 0;
        while used.contains(&c) {
            c += 1;
        }
        color[j] = c;
        for &(q, p) in check.pauli_string.iter() {
            if p != Pauli::I {
                qubit_colors.entry(q).or_default().insert(c);
            }
        }
    }
    let num_colors = color.iter().copied().max().map_or(0, |c| c + 1);

    // Emit one color class at a time (sequential, keeping shared-qubit checks contiguous);
    // within a class the qubit-disjoint checks run in lockstep — at step `k`, each member
    // emits its `k`-th controlled-Pauli, so every data qubit is touched at most once per
    // TICK layer.
    for c in 0..num_colors {
        let members: Vec<usize> = (0..m).filter(|&j| color[j] == c).collect();
        let max_weight = members
            .iter()
            .map(|&j| checks[j].pauli_string.iter().filter(|&&(_, p)| p != Pauli::I).count())
            .max()
            .unwrap_or(0);
        for k in 0..max_weight {
            for &j in &members {
                let sites = checks[j].pauli_string.iter().filter(|&&(_, p)| p != Pauli::I);
                if let Some(&(q, p)) = sites.clone().nth(k) {
                    out.push(PhysicalGate::Controlled(p, base + j, q));
                }
            }
            out.push(PhysicalGate::Tick);
        }
    }

    // Measure layer.
    for (j, check) in checks.iter().enumerate() {
        debug_assert!(
            matches!(check.sign, Sign::One | Sign::NegOne),
            "non-Hermitian sign on a syndrome MPP"
        );
        let invert = check.sign == Sign::NegOne;
        out.push(PhysicalGate::Measure(Pauli::X, base + j, invert));
    }
    out.push(PhysicalGate::Tick);
}

/// Lower a single gate to its flat-qubit form, threading `intern` through so the
/// `REPEAT` body shares the same qubit numbering as the surrounding circuit.
fn flatten_gate<Q: PauliStringIndex>(
    gate: &PhysicalGate<Q>,
    intern: &mut impl FnMut(Q) -> usize,
) -> PhysicalGate<usize> {
    match gate {
        PhysicalGate::Reset(basis, q) => PhysicalGate::Reset(*basis, intern(*q)),
        PhysicalGate::Measure(basis, q, invert) => {
            PhysicalGate::Measure(*basis, intern(*q), *invert)
        }
        PhysicalGate::Controlled(p, c, t) => {
            // Intern the control before the target to keep numbering deterministic.
            let c = intern(*c);
            PhysicalGate::Controlled(*p, c, intern(*t))
        }
        PhysicalGate::Tick => PhysicalGate::Tick,
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

/// [`flatten_gate`], taking the gate by value so its record lists move instead of
/// being cloned.
fn into_flat_gate<Q: PauliStringIndex>(
    gate: PhysicalGate<Q>,
    intern: &mut impl FnMut(Q) -> usize,
) -> PhysicalGate<usize> {
    match gate {
        PhysicalGate::DeclareObservable(index, recs) => {
            PhysicalGate::DeclareObservable(index, recs)
        }
        PhysicalGate::DeclareDetector(recs) => PhysicalGate::DeclareDetector(recs),
        PhysicalGate::Repeat(n, body) => PhysicalGate::Repeat(
            n,
            body.into_iter().map(|g| into_flat_gate(g, intern)).collect(),
        ),
        gate => flatten_gate(&gate, intern),
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
            PhysicalGate::Measure(basis, _, _) => {
                let basis = *basis;
                write!(f, "{indent}M{basis}")?;
                while let Some(PhysicalGate::Measure(b, q, invert)) = gates.get(i) {
                    if *b != basis {
                        break;
                    }
                    if *invert {
                        write!(f, " !{q}")?;
                    } else {
                        write!(f, " {q}")?;
                    }
                    i += 1;
                }
                writeln!(f)?;
            }
            PhysicalGate::Controlled(pauli, _, _) => {
                let pauli = *pauli;
                write!(f, "{indent}C{pauli}")?;
                while let Some(PhysicalGate::Controlled(p, c, t)) = gates.get(i) {
                    if *p != pauli {
                        break;
                    }
                    write!(f, " {c} {t}")?;
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
            PhysicalGate::Measure(basis, q, invert) => {
                if *invert {
                    write!(f, "M{basis} !{q}")
                } else {
                    write!(f, "M{basis} {q}")
                }
            }
            PhysicalGate::Controlled(pauli, c, t) => write!(f, "C{pauli} {c} {t}"),
            PhysicalGate::Tick => write!(f, "TICK"),
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

// ---------------------------------------------------------------------------
// Streaming gate output
//
// The SEC expansion of a large circuit is far too big to hold in memory (tens of
// millions of gates, gigabytes of stim text), so the expansion is written against
// a *sink* rather than returning a `Vec`. `Vec<PhysicalGate<usize>>` is the
// materializing sink used by tests and by the small-circuit paths; [`StimWriter`]
// renders each gate to text the moment it is produced and drops it, so peak memory
// stays at the (unexpanded) input circuit plus an output buffer.
// ---------------------------------------------------------------------------

/// A consumer of lowered, flat-qubit gates. See the module note above.
pub(crate) trait GateSink {
    fn push(&mut self, gate: PhysicalGate<usize>);
    /// Named `extend_gates`, not `extend`, so that it does not collide with
    /// `Extend::extend` on the `Vec` sink.
    fn extend_gates<I: IntoIterator<Item = PhysicalGate<usize>>>(&mut self, gates: I) {
        for gate in gates {
            self.push(gate);
        }
    }
}

impl GateSink for Vec<PhysicalGate<usize>> {
    fn push(&mut self, gate: PhysicalGate<usize>) {
        Vec::push(self, gate);
    }
    fn extend_gates<I: IntoIterator<Item = PhysicalGate<usize>>>(&mut self, gates: I) {
        Extend::extend(self, gates);
    }
}

/// A sink that keeps only the `TICK` count (with `REPEAT` bodies counted by their
/// repeat factor) — the `sec_depth` metric, obtained without materializing the
/// expansion it measures.
#[derive(Default)]
pub(crate) struct TickCounter {
    ticks: usize,
}

impl GateSink for TickCounter {
    fn push(&mut self, gate: PhysicalGate<usize>) {
        self.ticks += count_ticks(std::slice::from_ref(&gate));
    }
}

/// The line-coalescing run a [`StimWriter`] is in the middle of emitting.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Run {
    None,
    Reset(Pauli),
    Measure(Pauli),
    Controlled(Pauli),
    XCorrection,
}

/// A [`GateSink`] that streams stim text straight to an `io::Write`.
///
/// Produces byte-identical output to [`write_gates`] — a maximal run of same-kind
/// single-qubit gates is coalesced onto one line — but does so from a gate *stream*,
/// keeping only the in-progress line's state. The first I/O error is latched and
/// returned by [`StimWriter::finish`], which also reports the `TICK` count so
/// `--stats` needs no second pass over the expansion.
pub(crate) struct StimWriter<W: Write> {
    out: W,
    run: Run,
    indent: usize,
    ticks: usize,
    result: std::io::Result<()>,
}

impl<W: Write> StimWriter<W> {
    pub(crate) fn new(out: W) -> StimWriter<W> {
        StimWriter { out, run: Run::None, indent: 0, ticks: 0, result: Ok(()) }
    }

    /// Flush the pending line, flush the underlying writer, and report the total
    /// `TICK` count (or the first I/O error hit along the way).
    pub(crate) fn finish(mut self) -> std::io::Result<usize> {
        self.end_run();
        self.result?;
        self.out.flush()?;
        Ok(self.ticks)
    }

    /// Run `f` unless a previous write already failed, latching its error.
    fn write(&mut self, f: impl FnOnce(&mut W) -> std::io::Result<()>) {
        if self.result.is_ok() {
            self.result = f(&mut self.out);
        }
    }

    /// Terminate the coalesced line in progress, if any.
    fn end_run(&mut self) {
        if self.run != Run::None {
            self.run = Run::None;
            self.write(|w| writeln!(w));
        }
    }

    /// Start (or continue) a coalesced line of kind `run` whose head is `head`.
    fn start_run(&mut self, run: Run, head: std::fmt::Arguments<'_>) {
        if self.run == run {
            return;
        }
        self.end_run();
        self.run = run;
        let indent = self.indent;
        self.write(|w| write!(w, "{:indent$}{head}", ""));
    }

    /// Emit `gate` on a line of its own, ending any coalesced line first.
    fn standalone(&mut self, gate: &PhysicalGate<usize>) {
        self.end_run();
        let indent = self.indent;
        self.write(|w| writeln!(w, "{:indent$}{gate}", ""));
    }
}

impl<W: Write> GateSink for StimWriter<W> {
    fn push(&mut self, gate: PhysicalGate<usize>) {
        match gate {
            PhysicalGate::Reset(basis, q) => {
                self.start_run(Run::Reset(basis), format_args!("R{basis}"));
                self.write(|w| write!(w, " {q}"));
            }
            PhysicalGate::Measure(basis, q, invert) => {
                self.start_run(Run::Measure(basis), format_args!("M{basis}"));
                let bang = if invert { "!" } else { "" };
                self.write(|w| write!(w, " {bang}{q}"));
            }
            PhysicalGate::Controlled(pauli, c, t) => {
                self.start_run(Run::Controlled(pauli), format_args!("C{pauli}"));
                self.write(|w| write!(w, " {c} {t}"));
            }
            PhysicalGate::XCorrection(rec, q) => {
                self.start_run(Run::XCorrection, format_args!("CNOT"));
                self.write(|w| write!(w, " rec[-{rec}] {q}"));
            }
            PhysicalGate::Tick => {
                self.ticks += 1;
                self.standalone(&PhysicalGate::Tick);
            }
            PhysicalGate::Repeat(n, body) => {
                self.end_run();
                let indent = self.indent;
                self.write(|w| writeln!(w, "{:indent$}REPEAT {n} {{", ""));
                self.indent += 4;
                let before = self.ticks;
                for gate in body {
                    self.push(gate);
                }
                self.end_run();
                self.indent -= 4;
                // The body's ticks were counted once by the loop above; the
                // remaining n−1 iterations are accounted for here.
                self.ticks += n.saturating_sub(1) * (self.ticks - before);
                let indent = self.indent;
                self.write(|w| writeln!(w, "{:indent$}}}", ""));
            }
            gate => self.standalone(&gate),
        }
    }
}

fn concatenate_circuits(
    circuits: Vec<PhysicalCircuit<MergedQubit>>,
) -> PhysicalCircuit<MergedQubit> {
    // Reuse the first circuit's buffer and grow it once to the final size. A
    // `flat_map(..).collect()` instead reallocates its way up from empty (nested
    // iterators have no usable size hint) *and* moves every gate of the first
    // circuit, which is the expensive one when concatenating onto an accumulator.
    let total: usize = circuits.iter().map(|c| c.gates.len()).sum();
    let mut circuits = circuits.into_iter();
    let mut gates = circuits.next().map(|c| c.gates).unwrap_or_default();
    gates.reserve(total - gates.len());
    for c in circuits {
        gates.extend(c.gates);
    }
    PhysicalCircuit { gates }
}

#[derive(Debug, Clone)]
pub(crate) enum PhysicalGate<Q> {
    /// Reset a qubit into the `+1` eigenstate of the given Pauli basis, lowered to
    /// stim's `RX` / `RY` / `RZ` (`Z` is the usual `|0>`).
    Reset(Pauli, Q),
    /// A single-qubit measurement in the given Pauli basis, lowered to stim's
    /// `MX` / `MY` / `MZ`. The `bool` inverts the recorded outcome (stim's `M_ !q`),
    /// used by the ancilla syndrome gadget to realize a negative-sign check.
    Measure(Pauli, Q, bool),
    XCorrection(usize, Q),
    DeclareObservable(usize, Vec<usize>),
    DeclareDetector(Vec<usize>),
    MPP(PauliAxis<Q>),
    /// A qubit-controlled Pauli (control, then target), lowered to stim's
    /// `CX` / `CY` / `CZ`. Emitted by the ancilla syndrome gadget to entangle a
    /// measurement ancilla (control) with a data qubit (target). Distinct from
    /// [`PhysicalGate::XCorrection`], which is *record*-controlled.
    Controlled(Pauli, Q, Q),
    /// A stim `TICK`: a timing-layer boundary. Carries no qubits and no records;
    /// emitted by the ancilla syndrome gadget to delimit physical gate layers.
    Tick,
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
    for deformation in checks.deformations.iter() {
        let Deformation { checks, corrections, edge_basis } = &**deformation;
        let new_subcircuit =
            one_ppm_to_physical_circuit(checks, corrections, &base_checks, rounds, *edge_basis);
        // Append in place: concatenating into a fresh circuit each iteration would
        // move the whole accumulated gate list once per deformation, i.e. quadratic
        // in the deformation count with a full Pauli string riding on every `MPP`.
        circuit.gates.extend(new_subcircuit.gates);
    }
    circuit
}

/// Walk the lowering of `checks` one deformation at a time, handing each chunk of
/// merged-qubit gates to `emit` and dropping it before the next is built.
///
/// Chunking exists so the caller never has to hold the whole circuit: each
/// deformation re-emits `step0` (a full `rounds`-deep pass over *every* base check),
/// so the gates a deformation expands to outweigh the deformation itself by an order
/// of magnitude, and on a large circuit their sum is what exhausts memory.
///
/// The cut points are deliberate. A syndrome round is a maximal run of consecutive
/// `MPP`s, and every deformation ends in its split's `M` / `CNOT` gates, so no round
/// — and no `REPEAT`-to-reference-round pairing — ever straddles a chunk boundary.
/// Consumers that track round structure or look ahead within a round (the SEC
/// expansions) therefore see exactly what they would have seen on the whole circuit.
fn for_each_deformation_chunk(
    checks: &DeformedCheckSequence,
    rounds: usize,
    mut emit: impl FnMut(&[PhysicalGate<MergedQubit>]),
) {
    let qubits: Vec<MergedQubit> = checks
        .base_checks
        .iter()
        .flat_map(|p| p.pauli_string.iter())
        .map(|&(q, _)| q)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let resets: Vec<PhysicalGate<MergedQubit>> = qubits
        .into_iter()
        .map(|q| PhysicalGate::Reset(Pauli::Z, q))
        .collect();
    emit(&resets);

    for deformation in &checks.deformations {
        let Deformation { checks: deformed, corrections, edge_basis } = &**deformation;
        let sub = one_ppm_to_physical_circuit(
            deformed,
            corrections,
            &checks.base_checks,
            rounds,
            *edge_basis,
        );
        emit(&sub.gates);
    }
}

/// The number of distinct merged qubits the streamed circuit will touch — equivalently
/// the size of the flat id space [`PhysicalCircuit::flatten`] hands out, and so the
/// first free ancilla id the SEC expansion needs *before* the first gate is written.
///
/// Read off the checks rather than off the gate stream, and over the *distinct*
/// deformations only (the support cache leaves the same `Rc` repeated many times), so
/// it costs a pass over a few hundred deformations instead of a second full lowering.
/// Only the count is taken from here: the ids themselves are still assigned in order
/// of first appearance while streaming, which is what keeps the output identical to
/// the materializing path.
fn merged_qubit_count(checks: &DeformedCheckSequence) -> usize {
    let mut qubits: std::collections::BTreeSet<MergedQubit> = Default::default();
    for p in &checks.base_checks {
        qubits.extend(p.pauli_string.iter().map(|&(q, _)| q));
    }
    let mut seen: std::collections::HashSet<*const Deformation> = Default::default();
    for deformation in &checks.deformations {
        if !seen.insert(std::rc::Rc::as_ptr(deformation)) {
            continue;
        }
        for p in &deformation.checks {
            qubits.extend(p.pauli_string.iter().map(|&(q, _)| q));
        }
        qubits.extend(deformation.corrections.iter().map(|c| c.qubit));
    }
    qubits.len()
}

/// How far back this circuit's `rec[-k]` rewrites can reach, and so how many records
/// the streaming LRC expansion has to retain.
///
/// The only rewritten rec-consumer on the streaming path is the split's
/// `XCorrection`s, which reach back over that same split's edge measurements — the
/// round detectors of both schedules are emitted directly in new-record space and
/// never go through the rewrite, and the frame-solver observables that *can* reach
/// arbitrarily far exist only on the materializing path. The round sizes are folded
/// in anyway so the bound survives a future round gaining a rewritten detector, and
/// a flat margin covers the rest: an entry costs 8 bytes, so over-provisioning here
/// is free next to getting it wrong (which [`RecordMap::get`] turns into a panic
/// rather than a silently misdirected record).
fn record_window(checks: &DeformedCheckSequence) -> usize {
    let mut widest = checks.base_checks.len();
    let mut seen: std::collections::HashSet<*const Deformation> = Default::default();
    for deformation in &checks.deformations {
        if !seen.insert(std::rc::Rc::as_ptr(deformation)) {
            continue;
        }
        widest = widest.max(deformation.checks.len());
        let edge_qubits = deformation
            .checks
            .iter()
            .flat_map(|p| p.pauli_string.iter())
            .map(|&(q, _)| q)
            .filter(|q| matches!(q, MergedCodeQubit::EdgeQubit(_)))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        widest = widest.max(edge_qubits);
    }
    4 * widest + 65536
}

/// The per-chunk half of the streaming lowering: flatten a chunk of merged gates
/// through an interner that persists across chunks, optionally SEC-expand it, and
/// hand the result to `sink` — accumulating the `--stats` counters that can only be
/// read off the merged, pre-expansion form on the way past.
struct ChunkPipe<S: GateSink> {
    ids: BTreeMap<MergedQubit, usize>,
    /// Reused across chunks so the per-deformation flattening does not reallocate.
    flat: Vec<PhysicalGate<usize>>,
    /// The LRC expansion's old→new record map, carried across chunks so a later
    /// chunk's `rec[-k]` still resolves against earlier records. Windowed: keeping
    /// every record is what made `--stats` and `--syndrome-extraction-circuits`
    /// memory-bound long after the gates themselves stopped being.
    lrc_map: crate::sec_schedule::RecordMap,
    schedule: Option<SecSchedule>,
    ancilla_base: usize,
    syndrome_cycles: usize,
    sec_ancilla_pool: usize,
    sink: S,
}

impl<S: GateSink> ChunkPipe<S> {
    fn new(
        schedule: Option<SecSchedule>,
        ancilla_base: usize,
        record_window: usize,
        sink: S,
    ) -> ChunkPipe<S> {
        ChunkPipe {
            ids: BTreeMap::new(),
            flat: Vec::new(),
            lrc_map: crate::sec_schedule::RecordMap::windowed(record_window),
            schedule,
            ancilla_base,
            syndrome_cycles: 0,
            sec_ancilla_pool: 0,
            sink,
        }
    }

    fn push_chunk(&mut self, gates: &[PhysicalGate<MergedQubit>]) {
        // The whole chunking scheme rests on this: a syndrome round is a maximal run
        // of consecutive `MPP`s, so a chunk that ended mid-run would be expanded — and
        // counted — as two rounds where the materialized circuit has one. Deformations
        // normally end in their split's `M`/`CNOT` gates, but `split_and_correct` emits
        // nothing for a deformation whose checks touch no edge qubit, which at
        // `rounds == 1` (no trailing `REPEAT`) would leave the merge's `MPP`s last.
        debug_assert!(
            !matches!(gates.last(), Some(PhysicalGate::MPP(_))),
            "chunk ends mid-syndrome-round; splitting here would regroup the round",
        );
        // Both metrics count maximal `MPP` runs, which never span a chunk (see
        // `for_each_deformation_chunk`), so per-chunk accumulation is exact.
        self.syndrome_cycles += count_syndrome_rounds(gates);
        self.sec_ancilla_pool = self.sec_ancilla_pool.max(widest_syndrome_round(gates));

        self.flat.clear();
        let ids = &mut self.ids;
        let mut intern = |q: MergedQubit| -> usize {
            let next = ids.len();
            *ids.entry(q).or_insert(next)
        };
        self.flat
            .extend(gates.iter().map(|g| flatten_gate(g, &mut intern)));

        match self.schedule {
            None => {
                for gate in self.flat.drain(..) {
                    self.sink.push(gate);
                }
            }
            Some(SecSchedule::Lrc) => crate::sec_schedule::expand_gates_lrc_with_map(
                &self.flat,
                self.ancilla_base,
                &mut self.sink,
                &mut self.lrc_map,
            ),
            Some(SecSchedule::Legacy) => {
                expand_gates_into(&self.flat, self.ancilla_base, &mut self.sink)
            }
        }
    }

    /// The qubit-count metrics, available once every chunk has been interned.
    fn qubit_counts(&self) -> (usize, usize) {
        self.ids.keys().fold((0, 0), |(code, edge), q| match q {
            MergedCodeQubit::EdgeQubit(_) => (code, edge + 1),
            MergedCodeQubit::CodeQubit { .. } => (code + 1, edge),
        })
    }
}

/// Lower `checks` straight to stim text without ever materializing the circuit,
/// returning the `TICK` count of what was written alongside the `--stats` metrics
/// gathered on the way. The streaming counterpart of
/// [`checks_to_physical_circuit`] + [`PhysicalCircuit::into_flat`] +
/// [`PhysicalCircuit::write_stim`], and byte-for-byte equivalent to that chain.
///
/// `sec_depth` is left at zero: with `schedule` set it is the returned tick count,
/// and without it the caller has to decide whether the metric is worth a second
/// pass (see [`stream_sec_tick_count`]).
///
/// Only the observable-free path is served here; a memory experiment's observables
/// come from a whole-circuit frame solve that has nothing to stream against.
pub fn stream_stim<W: Write>(
    checks: &DeformedCheckSequence,
    rounds: usize,
    schedule: Option<SecSchedule>,
    out: W,
) -> std::io::Result<(usize, CircuitStats)> {
    let mut pipe = ChunkPipe::new(
        schedule,
        merged_qubit_count(checks),
        record_window(checks),
        StimWriter::new(out),
    );
    for_each_deformation_chunk(checks, rounds, |gates| pipe.push_chunk(gates));
    let (code_qubits, edge_qubits) = pipe.qubit_counts();
    let stats = CircuitStats {
        code_qubits,
        edge_qubits,
        sec_ancilla_pool: pipe.sec_ancilla_pool,
        syndrome_cycles: pipe.syndrome_cycles,
        sec_depth: 0,
    };
    let ticks = pipe.sink.finish()?;
    Ok((ticks, stats))
}

/// The `sec_depth` of `checks`'s SEC lowering, counted by streaming the expansion
/// through a counting sink. Needed only when `--stats` is asked for without
/// `--syndrome-extraction-circuits`, where the written circuit is not the expanded
/// one; it regenerates the gate stream, so it costs time but holds nothing.
pub fn stream_sec_tick_count(
    checks: &DeformedCheckSequence,
    rounds: usize,
    schedule: SecSchedule,
) -> usize {
    let mut pipe = ChunkPipe::new(
        Some(schedule),
        merged_qubit_count(checks),
        record_window(checks),
        TickCounter::default(),
    );
    for_each_deformation_chunk(checks, rounds, |gates| pipe.push_chunk(gates));
    pipe.sink.ticks
}

fn one_ppm_to_physical_circuit(
    checks: &[GraphPauli<BlockKind>],
    corrections: &[CorrectionSupport<BlockKind>],
    base_checks: &[GraphPauli<BlockKind>],
    rounds: usize,
    edge_basis: Pauli,
) -> PhysicalCircuit<MergedQubit> {
    let step0 = d_rounds(base_checks, rounds);
    // Initialize this merge's edge qubits in the basis CONJUGATE to the vertex
    // checks' edge Pauli: |0> for X-edge vertex checks, |+> for Z-edge ones. This
    // way every vertex check `A_v = P(edges)·L_q` is individually random and only
    // the collective product `∏A_v = L` is determined — the gauging measurement.
    // Initializing in the vertex-edge basis instead makes each `A_v` equivalent
    // to its lone code-qubit factor `L_q`, collapsing the merge into transversal
    // single-qubit measurements of the support (an entirely different channel;
    // this was a real bug for every pure-Z operator, whose edge basis is Z).
    // The explicit reset also matters because edge-qubit ids are reused across
    // deformations: without it, a later merge starts from the previous split's
    // leftover MZ eigenstates.
    let init = initialization(checks, conjugate_basis(edge_basis));
    let merge = d_rounds(checks, rounds);
    let split = split_and_correct(checks, corrections, conjugate_basis(edge_basis));
    concatenate_circuits(vec![step0, init, merge, split])
}

/// The single-qubit Pauli basis conjugate (anticommuting) to `p`; edge bases are
/// only ever X or Z.
fn conjugate_basis(p: Pauli) -> Pauli {
    match p {
        Pauli::X => Pauli::Z,
        Pauli::Z => Pauli::X,
        _ => unreachable!("edge basis is always X or Z"),
    }
}

fn d_rounds(checks: &[GraphPauli<BlockKind>], rounds: usize) -> PhysicalCircuit<MergedQubit> {
    let check_count = checks.len();
    if rounds == 0 {
        return PhysicalCircuit { gates: Vec::new() };
    }
    // First round establishes the reference measurements; it has no preceding
    // round to compare against, so no detectors are declared yet.
    let mut output_gates: Vec<PhysicalGate<MergedQubit>> =
        Vec::with_capacity(check_count + usize::from(rounds > 1));
    output_gates.extend(checks.iter().map(|check| PhysicalGate::MPP(check.clone())));
    if rounds > 1 {
        // The remaining rounds are identical: re-measure every check and compare
        // it against its value one round earlier. The rec offsets are the same on
        // every iteration (each iteration adds `check_count` measurements), so a
        // single `REPEAT (rounds - 1)` body reproduces the unrolled circuit. The
        // first iteration's "previous round" is the reference round above.
        let mut body = Vec::with_capacity(2 * check_count);
        body.extend_from_slice(&output_gates);
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

fn initialization(
    checks: &[GraphPauli<BlockKind>],
    basis: Pauli,
) -> PhysicalCircuit<MergedQubit> {
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
        .map(|e| PhysicalGate::Reset(basis, *e))
        .collect();
    PhysicalCircuit { gates }
}

/// Split the merged code back apart: measure every edge qubit in `split_basis` —
/// the conjugate of the vertex checks' edge Pauli, so the split destroys the
/// individual `A_v` (gauge) checks while the collective `∏A_v = L` and the
/// deformed code stabilizers survive — then apply the surgery's `X(q)` Pauli
/// byproduct on each operator-support qubit, controlled on the parity of the edge
/// outcomes along that qubit's correction path (arXiv:2410.02213, Def. 10,
/// Stage 4). Measuring instead in the vertex-edge basis would *commute* with each
/// `A_v` and thereby complete a measurement of every single code-qubit factor
/// `L_q` — the same transversal collapse the conjugate-basis edge init avoids.
/// The byproduct is a real frame gate; *how* each memory readout combines with
/// the resulting records to stay deterministic is determined later by the
/// symbolic frame solver, not annotated here.
fn split_and_correct(
    checks: &[GraphPauli<BlockKind>],
    corrections: &[CorrectionSupport<BlockKind>],
    split_basis: Pauli,
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
        .map(|e| PhysicalGate::Measure(split_basis, *e, false))
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
    add_observables: bool,
) -> PhysicalCircuit<MergedQubit> {
    // The transversal readout, observables and final detectors come from a
    // stabilizer-frame solve that dominates compile time; skip it when the caller
    // only wants the bare syndrome circuit.
    if !add_observables {
        return checks_to_physical_circuit(checks, rounds);
    }
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
pub fn compile_plain_memory_experiment(
    code: &CodeData,
    rounds: usize,
    basis: Pauli,
    add_observables: bool,
) -> PhysicalCircuit<MergedQubit> {
    // Empty supports => no deformations; we only want the lifted base checks.
    let base_checks =
        physical_supports_to_stabilizer_checks(&[], code, 1, &SurgeryGraphConfig::default()).base_checks;
    let deep = plain_memory_rounds(&base_checks, rounds, basis);
    // The readout/observable/detector pass is the expensive stabilizer-frame solve.
    if !add_observables {
        return deep;
    }
    let single = plain_memory_rounds(&base_checks, 1, basis);
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
    concatenate_circuits(vec![circuit, d_rounds(base_checks, rounds)])
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
        gates.push(PhysicalGate::Measure(basis, q, false));
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
            PhysicalGate::MPP(_) | PhysicalGate::Measure(_, _, _) => 1,
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
                    PhysicalGate::MPP(_) | PhysicalGate::Measure(_, _, _) => {
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
            // `basis_q` is already stabilized, with symbolic sign `s` (the product
            // of the stabilizers whose destabilizer partner anticommutes with it,
            // as in `measure`'s deterministic branch). Resetting forces that sign
            // to +1 — physically a conjugate-Pauli flip conditioned on `s` — so
            // every generator anticommuting with the flip picks up `s`. Clearing
            // just a single-qubit generator's sign instead silently corrupts any
            // other generator sharing the qubit (e.g. a surviving cycle-check row
            // when a mid-circuit edge re-init hits a qubit still entangled into
            // earlier check rows).
            let mut s = std::collections::BTreeSet::new();
            for d in 0..self.n {
                if anticommutes(self, d) {
                    s = &s ^ &self.sym[self.n + d];
                }
            }
            // The flip is the conjugate Pauli at `q`: `X_q` for a Z reset (rows
            // with Z-content at `q` flip), `Z_q` for an X reset (X-content).
            let flip_anticommutes = |sim: &Self, r: usize| match basis {
                Pauli::X => get_bit(&sim.x[r], q),
                _ => get_bit(&sim.z[r], q),
            };
            for r in 0..2 * self.n {
                if flip_anticommutes(self, r) {
                    self.sym[r] = &self.sym[r] ^ &s;
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
                | PhysicalGate::Measure(_, q, _)
                | PhysicalGate::XCorrection(_, q) => out.push(*q),
                PhysicalGate::Controlled(_, c, t) => {
                    out.push(*c);
                    out.push(*t);
                }
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
            PhysicalGate::Measure(pauli, q, _) => {
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

    // Debug aid (`FRAME_DEBUG=1`): print each record's determinism verdict, for
    // diffing the symbolic frame sim against an external stim replay.
    if std::env::var("FRAME_DEBUG").is_ok() {
        for &rid in &measure_rids {
            eprintln!("[frame] rec {rid} det={}", !sim.rec[rid].contains(&rid));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::PauliString;

    fn axis(sites: &[(usize, Pauli)]) -> PauliAxis<usize> {
        PauliAxis { sign: Sign::One, pauli_string: PauliString::new(sites.to_vec()) }
    }

    /// A circuit hitting every printing path: coalesced runs of resets /
    /// measurements / controlled-Paulis, a sign-inverted check, a `d_rounds`-shaped
    /// `REPEAT` segment (which the LRC schedule lowers to the staggered form), and
    /// the three kinds of record consumer.
    fn sample() -> PhysicalCircuit<usize> {
        let checks = [
            axis(&[(0, Pauli::X), (1, Pauli::X), (2, Pauli::X)]),
            axis(&[(1, Pauli::X), (2, Pauli::X), (3, Pauli::X)]),
            axis(&[(0, Pauli::Z), (1, Pauli::Z), (2, Pauli::Z)]),
            PauliAxis {
                sign: Sign::NegOne,
                pauli_string: PauliString::new(vec![
                    (1, Pauli::Z),
                    (2, Pauli::Z),
                    (3, Pauli::Z),
                ]),
            },
        ];
        let m = checks.len();
        let mut gates: Vec<PhysicalGate<usize>> =
            (0..4).map(|q| PhysicalGate::Reset(Pauli::Z, q)).collect();
        gates.extend(checks.iter().cloned().map(PhysicalGate::MPP));
        let mut body: Vec<PhysicalGate<usize>> =
            checks.iter().cloned().map(PhysicalGate::MPP).collect();
        body.extend((0..m).map(|j| PhysicalGate::DeclareDetector(vec![j + 1, m + j + 1])));
        gates.push(PhysicalGate::Repeat(3, body));
        gates.push(PhysicalGate::XCorrection(2, 0));
        gates.push(PhysicalGate::XCorrection(1, 3));
        gates.extend((0..4).map(|q| PhysicalGate::Measure(Pauli::Z, q, q == 2)));
        gates.push(PhysicalGate::DeclareDetector(vec![1, 4]));
        gates.push(PhysicalGate::DeclareObservable(0, vec![1, 2]));
        PhysicalCircuit { gates }
    }

    /// The streaming writer is a drop-in for materializing the circuit and
    /// formatting it: same bytes, and the same TICK count `--stats` reports.
    #[test]
    fn streamed_output_matches_display() {
        for schedule in [SecSchedule::Lrc, SecSchedule::Legacy] {
            let expanded = sample().expand_mpp_to_sec(schedule);
            let mut buf = Vec::new();
            let ticks = sample().write_stim(Some(schedule), &mut buf).unwrap();
            assert_eq!(String::from_utf8(buf).unwrap(), expanded.to_string());
            assert_eq!(ticks, expanded.tick_count());
            assert_eq!(ticks, sample().sec_tick_count(schedule));
        }
    }

    /// The same, for the unexpanded (MPP) form written with no SEC lowering.
    #[test]
    fn streamed_mpp_output_matches_display() {
        let mut buf = Vec::new();
        sample().write_stim(None, &mut buf).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), sample().to_string());
    }
}
