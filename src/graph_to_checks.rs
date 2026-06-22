use petgraph::algo::astar;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

use crate::graph_construction::{Edge, SurgeryGraph};
use crate::pbc::{CodePauli, GraphPauli, MergedCodeQubit, Pauli, PauliString, Sign, lift_code_to_merged};


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
/// The vertex-check *edge* Pauli for measuring `operator`, chosen to be compatible
/// with the surgery's hardcoded split convention ([`split_and_correct`]: edge
/// qubits reset to `|0>`, measured in `Z`, with an `X` byproduct). This is about
/// the *edge / ancilla* qubits of the merge, not the logical readout basis — once
/// the edge basis is right the surgery faithfully realizes the logical
/// measurement, and the resulting circuit then reads out correctly in *any*
/// transversal basis (the determinism solver is basis-agnostic).
///
/// The gauging-measurement construction is basis-symmetric (Remark 3 of
/// arXiv:2503.10390): the edge basis of the vertex / cycle / deformed-code checks
/// can be either `X`/`Z` or `Z`/`X` (arXiv:2410.02213, Def. 2 / Lemma 1). Given
/// the `Z`/`X`-flavored split above, the byproduct frame is tracked correctly when
/// the edge basis is `X` for any operator component that the split's `X` byproduct
/// must anticommute with — i.e. whenever `operator` carries an `X` or `Y` factor;
/// a purely `Z`-type operator keeps the original `Z` edges.
///
/// So the rule is: use the `X` edge basis whenever `operator` has any `X` or `Y`
/// factor, else `Z`. For a pure `X`- or `Z`-type operator this reduces to matching
/// the operator type; for a `Y` or mixed `X`/`Z` operator (e.g. a bridged
/// `Z[mem]⊗X[proc]` joint measurement) it picks `X`. Choosing `Z` there silently
/// mis-tracks the byproduct frame, so the surgery realizes the *wrong* logical
/// operation (a target logical that should be non-deterministic stays
/// deterministic). The merged code itself is valid for *any* uniform basis (every
/// check pair commutes and `ℒ` is the product of the vertex checks, since the path
/// matching is built from the operator's actual Paulis); the basis choice only
/// affects whether the byproduct/split faithfully implements the measurement.
///
/// Verified on the `H q[0]` circuit (which compiles to a Litinski H-gadget whose
/// third PPM is the mixed `Z[mem]⊗X[proc]`): with this rule both the `Z`- and the
/// `X`-basis memory experiments give the physically-correct observables
/// (`H|0> = |+>`: Z-random / X-deterministic on the target, and the mirror on the
/// bystanders). Not yet exercised: single-qubit-`Y` supports and the reverse mixed
/// `X[mem]⊗Z[proc]`.
pub fn operator_edge_basis<K: Ord + Copy>(operator: &CodePauli<K>) -> Pauli {
    let has_x_or_y = operator
        .pauli_string
        .iter()
        .any(|&(_, p)| matches!(p, Pauli::X | Pauli::Y));
    if has_x_or_y { Pauli::X } else { Pauli::Z }
}

/// The conjugate (anticommuting) single-qubit Pauli used on the cycle and
/// deformed-code checks, given the vertex-check edge basis.
fn conjugate_basis(p: Pauli) -> Pauli {
    match p {
        Pauli::X => Pauli::Z,
        Pauli::Z => Pauli::X,
        _ => unreachable!("edge basis is always X or Z"),
    }
}

pub fn graph_to_checks<K : Ord + Copy>(
    graph: &SurgeryGraph<K>,
    code_stabilizers: &Vec<CodePauli<K>>,
    operator: &CodePauli<K>
) -> Vec<GraphPauli<K>> {
    let g = &graph.graph;

    // Edge basis for this measurement: `vertex_edge` on the vertex checks,
    // `cycle_edge` (its conjugate) on the cycle and deformed-code checks.
    let vertex_edge = operator_edge_basis(operator);
    let cycle_edge = conjugate_basis(vertex_edge);

    // Resolve a graph edge (vertex pair) to its merged-code edge qubit.
    let edge_qubit = |(a, b): Edge| -> MergedCodeQubit<K> {
        let e = g
            .find_edge(NodeIndex::new(a), NodeIndex::new(b))
            .expect("threaded edge must exist in the surgery graph");
        MergedCodeQubit::EdgeQubit(e.index())
    };

    let mut joint_stabilizers: Vec<GraphPauli<K>> = Vec::new();

    // (1) Vertex checks: `vertex_edge` on every incident edge, plus ℒ_q on the
    // code qubit at a port.
    for v in g.node_indices() {
        let mut pairs: Vec<(MergedCodeQubit<K>, Pauli)> = g
            .edges(v)
            .map(|e| (MergedCodeQubit::EdgeQubit(e.id().index()), vertex_edge))
            .collect();
        // Port vertex f(q): the node weight carries its code qubit (block, index).
        // The within-block physical qubit index is `cq.index` — recorded at
        // construction and preserved through bridging, so no offset math here.
        if let Some(cq) = g[v] {
            let pauli_on_qubit = operator
                .pauli_string
                .iter()
                .find(|&&(q, _)| q == cq)
                .map(|&(_, p)| p)
                .unwrap_or(Pauli::I);
            pairs.push((
                MergedCodeQubit::CodeQubit {
                    block: cq.block,
                    index: cq.index,
                },
                pauli_on_qubit,
            ));
        }
        let pauli_string = PauliString::new(pairs);
        joint_stabilizers.push(GraphPauli {
            sign: Sign::One,
            pauli_string,
        });
    }

    // (2) Cycle checks: `cycle_edge` on every edge of each cycle in the threaded
    // basis.
    for cycle in &graph.cycle_checks {
        let pauli_string = PauliString::new(
            cycle
                .iter()
                .map(|&e| (edge_qubit(e), cycle_edge))
                .collect::<Vec<_>>(),
        );
        joint_stabilizers.push(GraphPauli {
            sign: Sign::One,
            pauli_string,
        });
    }

    // (3) Modified code checks: keep S, extending it by `cycle_edge` on its
    // path-matching edges where S anticommutes with ℒ (empty path matching ⇒ S
    // unchanged). The deformed check `s̃ = s · ∏_{e∈γ} cycle_edge(e)` commutes
    // with the vertex checks (its edges anticommute with `vertex_edge`, canceling
    // the code-qubit anticommutation) and with the cycle checks (same edge basis).
    for (i, stab) in code_stabilizers.iter().enumerate() {
        let path_edges = graph.path_matching.get(i).map_or(&[][..], Vec::as_slice);
        if path_edges.is_empty() {
            joint_stabilizers.push(lift_code_to_merged(stab));
            continue;
        }
        let mut pairs: Vec<(MergedCodeQubit<K>, Pauli)> = lift_code_to_merged(stab).pauli_string.iter().copied().collect();
        pairs.extend(path_edges.iter().map(|&e| (edge_qubit(e), cycle_edge)));
        joint_stabilizers.push(GraphPauli {
            sign: stab.sign,
            pauli_string: PauliString::new(pairs),
        });
    }

    joint_stabilizers
}

#[derive(Clone)]
pub struct CorrectionSupport<K>{pub qubit : MergedCodeQubit<K>, pub path : Vec<MergedCodeQubit<K>>}

/// BFS shortest path from `source` to node 0 in `graph`, returned as the
/// sequence of edge qubits traversed (empty if `source` is node 0).
fn edge_path_to_root<K>(
    graph: &UnGraph<Option<crate::pbc::CodeQubit<K>>, ()>,
    source: NodeIndex,
) -> Vec<MergedCodeQubit<K>> {
    let target = NodeIndex::new(0);

    // Unweighted shortest path (unit edge costs, zero heuristic ⇒ BFS-equivalent).
    let (_, nodes) = astar(graph, source, |n| n == target, |_| 1, |_| 0)
        .expect("surgery graph must be connected");

    // Convert the node path source → root into its traversed edge qubits.
    nodes
        .windows(2)
        .map(|pair| {
            let edge = graph
                .find_edge(pair[0], pair[1])
                .expect("consecutive path nodes must be adjacent");
            MergedCodeQubit::EdgeQubit(edge.index())
        })
        .collect()
}

pub fn get_correction_support<K : Ord + Copy>(graph: &SurgeryGraph<K>, operator: &CodePauli<K>) -> Vec<CorrectionSupport<K>> {
    operator
        .pauli_string
        .iter()
        .map(|(qubit, _)| {
            // Ports are labeled with their absolute within-block qubit id `index`,
            // but `ports_by_block()` lists them by *port position*, not by that id —
            // so find the port whose label matches this qubit rather than indexing
            // by `qubit.index` (which ranges over all of the block's qubits).
            let ports = graph.ports_by_block();
            let vertex = *ports
                .get(&qubit.block)
                .expect("should be a port")
                .iter()
                .find(|&&v| graph.graph[v] == Some(*qubit))
                .expect("operator qubit must label a port");
            let path = edge_path_to_root(&graph.graph, vertex);
            CorrectionSupport { qubit: MergedCodeQubit::from(*qubit), path }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_construction::{SurgeryGraphConfig, surgery_graph};
    use crate::pbc::{
        CodeQubit, PauliAxis, PhysicalPauliString, PhysicalQubit, axes_commute, pauli_string_mult,
    };

    /// A logical operator / stabilizer over the original code qubits 0..n.
    fn physical(paulis: &[Pauli]) -> PhysicalPauliString {
        PauliString::new(
            paulis
                .iter()
                .enumerate()
                .filter(|&(_, &p)| p != Pauli::I)
                .map(|(i, &p)| (PhysicalQubit(i), p))
                .collect(),
        )
    }

    /// The block key for the single-block sample code.
    const BLOCK: &str = "test";

    /// The same stabilizer expressed over the merged code's code qubits.
    fn code_pauli(paulis: &[Pauli]) -> CodePauli<&'static str> {
        CodePauli {
            sign: Sign::One,
            pauli_string: PauliString::new(
                paulis
                    .iter()
                    .enumerate()
                    .filter(|&(_, &p)| p != Pauli::I)
                    .map(|(i, &p)| {
                        (
                            CodeQubit {
                                block: BLOCK,
                                index: i,
                            },
                            p,
                        )
                    })
                    .collect(),
            ),
        }
    }

    /// A small merged code: measure XXXX against the two-stabilizer code
    /// {ZZII, IIZZ}. Both stabilizers anticommute with the operator (on {0,1}
    /// and {2,3}), so both gain path-matching edges.
    fn sample() -> (
        SurgeryGraph<&'static str>,
        Vec<CodePauli<&'static str>>,
        CodePauli<&'static str>,
    ) {
        let operator = code_pauli(&[Pauli::X, Pauli::X, Pauli::X, Pauli::X]);
        let stab_paulis = [
            [Pauli::Z, Pauli::Z, Pauli::I, Pauli::I],
            [Pauli::I, Pauli::I, Pauli::Z, Pauli::Z],
        ];
        let stabs: Vec<_> = stab_paulis.iter().map(|s| physical(s)).collect();
        let code_stabilizers: Vec<_> = stab_paulis.iter().map(|s| code_pauli(s)).collect();
        let graph = surgery_graph(&stabs, &physical(&[Pauli::X, Pauli::X, Pauli::X, Pauli::X]), BLOCK, &SurgeryGraphConfig::default());
        (graph, code_stabilizers, operator)
    }

    #[test]
    fn all_checks_pairwise_commute() {
        let (graph, code_stabilizers, operator) = sample();
        let checks = graph_to_checks(&graph, &code_stabilizers, &operator);
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
        let checks = graph_to_checks(&graph, &code_stabilizers, &operator);

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

        let expected: Vec<(MergedCodeQubit<&str>, Pauli)> = operator
            .pauli_string
            .iter()
            .map(|&(q, p)| {
                (
                    MergedCodeQubit::CodeQubit {
                        block: q.block,
                        index: q.index,
                    },
                    p,
                )
            })
            .collect();
        let got: Vec<(MergedCodeQubit<&str>, Pauli)> = product.pauli_string.iter().copied().collect();
        assert_eq!(product.sign, Sign::One);
        assert_eq!(got, expected);
    }
}