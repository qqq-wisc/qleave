use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;

use crate::graph_construction::{Edge, PhysicalQubit, SurgeryGraph};
use crate::pbc::{Pauli, PauliAxis, PauliString, PauliStringIndex, Sign};


type Stabilizer = PauliAxis<MergedCodeQubit>;
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum MergedCodeQubit{
    EdgeQubit(usize),
    CodeQubit(PhysicalQubit),
}
impl PauliStringIndex for MergedCodeQubit {

}

/// Build the merged-code stabilizers `𝒬̄` of Definition 2 (arXiv:2503.10390)
/// from a surgery graph, the original code stabilizers, and the logical operator
/// being measured.
///
/// The qubits of the merged code are the graph's *edges* (ancilla / edge qubits)
/// plus the original *code qubits* (the operator's support). Three families of
/// checks are emitted:
///
/// 1. **Vertex checks** — for every vertex `v`, `A_v = ∏_{e∋v} Z(e)`; if `v` is
///    the port `f(q)`, multiply in the logical-operator component `ℒ_q` on code
///    qubit `q`.
/// 2. **Cycle checks** — for every cycle `C` in the threaded cycle basis,
///    `B_C = ∏_{e∈C} X(e)`.
/// 3. **Modified code checks** — each original stabilizer `S` is kept; if it
///    anticommutes with `ℒ`, it is extended by `X` on its path-matching edges,
///    `S̄ = S ∏_{e∈μ(S,ℒ)} X(e)`.
///
/// `code_stabilizers` must be in the same order as the `stabilizers` passed to
/// [`crate::graph_construction::surgery_graph`], so that `code_stabilizers[i]`
/// pairs with `graph.path_matching[i]`.
fn graph_to_checks(
    graph: &SurgeryGraph,
    code_stabilizers: &Vec<Stabilizer>,
    operator: PauliString<PhysicalQubit>,
) -> Vec<Stabilizer> {
    let g = &graph.graph;
    let p = &graph.ports;

    // Resolve a graph edge (vertex pair) to its merged-code edge qubit.
    let edge_qubit = |(a, b): Edge| -> MergedCodeQubit {
        let e = g
            .find_edge(NodeIndex::new(a), NodeIndex::new(b))
            .expect("threaded edge must exist in the surgery graph");
        MergedCodeQubit::EdgeQubit(e.index())
    };

    let mut joint_stabilizers: Vec<Stabilizer> = Vec::new();

    // (1) Vertex checks: Z on every incident edge, plus ℒ_q on the code qubit at
    // a port.
    for v in g.node_indices() {
        let mut pairs: Vec<(MergedCodeQubit, Pauli)> = g
            .edges(v)
            .map(|e| (MergedCodeQubit::EdgeQubit(e.id().index()), Pauli::Z))
            .collect();
        if p.contains(&v) {
            // Port vertex f(q): node id is the physical qubit index q.0.
            let phys_qubit = PhysicalQubit(v.index());
            let pauli_on_qubit = operator
                .iter()
                .find(|&&(q, _)| q == phys_qubit)
                .map(|&(_, p)| p)
                .unwrap_or(Pauli::I);
            pairs.push((MergedCodeQubit::CodeQubit(phys_qubit), pauli_on_qubit));
        }
        let pauli_string = PauliString::new(pairs);
        joint_stabilizers.push(Stabilizer {
            sign: Sign::One,
            pauli_string,
        });
    }

    // (2) Cycle checks: X on every edge of each cycle in the threaded basis.
    for cycle in &graph.cycle_checks {
        let pauli_string = PauliString::new(
            cycle
                .iter()
                .map(|&e| (edge_qubit(e), Pauli::X))
                .collect::<Vec<_>>(),
        );
        joint_stabilizers.push(Stabilizer {
            sign: Sign::One,
            pauli_string,
        });
    }

    // (3) Modified code checks: keep S, extending it by X on its path-matching
    // edges where S anticommutes with ℒ (empty path matching ⇒ S unchanged).
    for (i, stab) in code_stabilizers.iter().enumerate() {
        let path_edges = graph.path_matching.get(i).map_or(&[][..], Vec::as_slice);
        if path_edges.is_empty() {
            joint_stabilizers.push(stab.clone());
            continue;
        }
        let mut pairs: Vec<(MergedCodeQubit, Pauli)> = stab.pauli_string.iter().copied().collect();
        pairs.extend(path_edges.iter().map(|&e| (edge_qubit(e), Pauli::X)));
        joint_stabilizers.push(Stabilizer {
            sign: stab.sign,
            pauli_string: PauliString::new(pairs),
        });
    }

    joint_stabilizers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_construction::surgery_graph;
    use crate::pbc::{axes_commute, pauli_string_mult};

    /// A logical operator / stabilizer over the original code qubits 0..n.
    fn physical(paulis: &[Pauli]) -> PauliString<PhysicalQubit> {
        PauliString::new(
            paulis
                .iter()
                .enumerate()
                .filter(|&(_, &p)| p != Pauli::I)
                .map(|(i, &p)| (PhysicalQubit(i), p))
                .collect(),
        )
    }

    /// The same stabilizer expressed over the merged code's code qubits.
    fn code_stabilizer(paulis: &[Pauli]) -> Stabilizer {
        Stabilizer {
            sign: Sign::One,
            pauli_string: PauliString::new(
                paulis
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| p != Pauli::I)
                    .map(|(i, &p)| (MergedCodeQubit::CodeQubit(PhysicalQubit(i)), p))
                    .collect(),
            ),
        }
    }

    /// A small merged code: measure XXXX against the two-stabilizer code
    /// {ZZII, IIZZ}. Both stabilizers anticommute with the operator (on {0,1}
    /// and {2,3}), so both gain path-matching edges.
    fn sample() -> (SurgeryGraph, Vec<Stabilizer>, PauliString<PhysicalQubit>) {
        let operator = physical(&[Pauli::X, Pauli::X, Pauli::X, Pauli::X]);
        let stab_paulis = [
            [Pauli::Z, Pauli::Z, Pauli::I, Pauli::I],
            [Pauli::I, Pauli::I, Pauli::Z, Pauli::Z],
        ];
        let stabs: Vec<_> = stab_paulis.iter().map(|s| physical(s)).collect();
        let code_stabilizers: Vec<_> = stab_paulis.iter().map(|s| code_stabilizer(s)).collect();
        let graph = surgery_graph(stabs, physical(&[Pauli::X, Pauli::X, Pauli::X, Pauli::X]));
        (graph, code_stabilizers, operator)
    }

    #[test]
    fn all_checks_pairwise_commute() {
        let (graph, code_stabilizers, operator) = sample();
        let checks = graph_to_checks(&graph, &code_stabilizers, operator);
        // A valid stabilizer group: every pair of checks must commute.
        for (i, a) in checks.iter().enumerate() {
            for b in &checks[i + 1..] {
                assert!(
                    axes_commute(&a.pauli_string, &b.pauli_string),
                    "checks {} and a later check anticommute",
                    i
                );
            }
        }
    }

    #[test]
    fn operator_is_product_of_vertex_checks() {
        let (graph, code_stabilizers, operator) = sample();
        // Vertex checks are emitted first, one per graph vertex.
        let num_vertices = graph.graph.node_count();
        let checks = graph_to_checks(&graph, &code_stabilizers, operator.clone());

        // ∏_v A_v: every edge qubit appears in exactly two vertex checks (its two
        // endpoints), so all Z(e) cancel, leaving only the ℒ components on the
        // code qubits — i.e. the original operator.
        let mut product = PauliAxis {
            sign: Sign::One,
            pauli_string: PauliString::new(vec![]),
        };
        for check in &checks[..num_vertices] {
            let prod = pauli_string_mult(&product.pauli_string, &check.pauli_string);
            product = PauliAxis {
                sign: product.sign * prod.sign * check.sign,
                pauli_string: prod.pauli_string,
            };
        }

        let expected: Vec<(MergedCodeQubit, Pauli)> = operator
            .iter()
            .map(|&(q, p)| (MergedCodeQubit::CodeQubit(q), p))
            .collect();
        let got: Vec<(MergedCodeQubit, Pauli)> = product.pauli_string.iter().copied().collect();
        assert_eq!(product.sign, Sign::One);
        assert_eq!(got, expected);
    }
}