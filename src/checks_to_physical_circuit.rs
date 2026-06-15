use std::collections::BTreeMap;

use crate::{circuit_to_checks::{BlockKind, DeformedCheckSequence}, graph_to_checks::CorrectionSupport, pbc::{GraphPauli, MergedCodeQubit, Pauli, PauliAxis, PauliStringIndex, Sign}
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

impl<Q> PhysicalCircuit<Q>{
    fn new() -> PhysicalCircuit<Q>{
        PhysicalCircuit { gates: vec![] }
    }

    fn add_gate(&mut self, gate: PhysicalGate<Q>){
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
        PhysicalGate::Reset(q) => PhysicalGate::Reset(intern(*q)),
        PhysicalGate::Measure(q) => PhysicalGate::Measure(intern(*q)),
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
            PhysicalGate::Reset(_) => {
                write!(f, "{indent}R")?;
                while let Some(PhysicalGate::Reset(q)) = gates.get(i) {
                    write!(f, " {q}")?;
                    i += 1;
                }
                writeln!(f)?;
            }
            PhysicalGate::Measure(_) => {
                write!(f, "{indent}M")?;
                while let Some(PhysicalGate::Measure(q)) = gates.get(i) {
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
            PhysicalGate::Reset(q) => write!(f, "R {q}"),
            PhysicalGate::Measure(q) => write!(f, "M {q}"),
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

fn concatenate_circuits(circuits: Vec<PhysicalCircuit<MergedQubit>>) -> PhysicalCircuit<MergedQubit> {
    let gates = circuits.into_iter().flat_map(|c| c.gates).collect();
    PhysicalCircuit { gates }
}

#[derive(Debug, Clone)]
enum PhysicalGate<Q> {
    Reset(Q),
    Measure(Q),
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
    for qubit in qubits{
        circuit.add_gate(PhysicalGate::Reset(qubit));

    }
    let mut observable_index = 0;
    for (checks, corrections )in checks.deformations{
        let new_subcircuit = one_ppm_to_physical_circuit(checks, corrections, base_checks.clone(), rounds, observable_index);
        circuit = concatenate_circuits(vec![circuit, new_subcircuit]);
        observable_index += 1;
    }
    circuit
}

fn one_ppm_to_physical_circuit(
    checks: Vec<GraphPauli<BlockKind>>,
    corrections: Vec<CorrectionSupport<BlockKind>>,
    base_checks: Vec<GraphPauli<BlockKind>>,
    rounds: usize,
    observable_index : usize
) -> PhysicalCircuit<MergedQubit> {
    let step0 = d_rounds(base_checks.clone(), rounds);
    let init = initialization(&base_checks);
    let merge = d_rounds(checks.clone(), rounds);
    let observable_decl = declare_observable(&checks, observable_index);
    let split = split_and_correct(&checks, &corrections);
    concatenate_circuits(vec![step0, init, merge, observable_decl, split])
}

fn d_rounds(checks: Vec<GraphPauli<BlockKind>>, rounds: usize) -> PhysicalCircuit<MergedQubit> {
    let check_count = checks.len();
    let mpp_gates: Vec<PhysicalGate<MergedQubit>> = checks
        .into_iter()
        .map(|check| PhysicalGate::MPP(check))
        .collect();
    let mut output_gates = Vec::new();
    if rounds == 0 {
        return PhysicalCircuit { gates: output_gates };
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
        .map(|e| PhysicalGate::Reset(*e))
        .collect();
    PhysicalCircuit { gates }
}

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
        .map(|e| PhysicalGate::Measure(*e))
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

fn get_vertex_check_indices(checks: &Vec<GraphPauli<BlockKind>>) -> Vec<usize> {
    checks
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            p.pauli_string
                .iter()
                .any(|&(q, pauli)| matches!(q, MergedCodeQubit::EdgeQubit(_)) && pauli == Pauli::Z)
        })
        .map(|(i, _)| i)
        .collect()
}

fn declare_observable(checks: &Vec<GraphPauli<BlockKind>>, observable_index: usize) -> PhysicalCircuit<MergedQubit> {
    let vertex_check_indices = get_vertex_check_indices(checks);
    let check_count: usize = vertex_check_indices.len();
    let rec_indices = vertex_check_indices
        .into_iter()
        .map(|i| check_count - i)
        .collect();
    let observable = PhysicalGate::DeclareObservable(observable_index, rec_indices);
    PhysicalCircuit {
        gates: vec![observable],
    }
}
