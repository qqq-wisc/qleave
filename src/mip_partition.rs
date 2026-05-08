use crate::circuit::{Circuit, Qubit};
use highs::{ColProblem, HighsModelStatus, RowProblem, Sense};
use std::collections::HashMap;

/// MIP-based optimal circuit partitioner. Drop-in replacement for the SAT version.
///
/// Variables (all binary, except ld/st which are continuous in [0,1] but come out integer):
///   g[i,j]   -- gate i in subcircuit j
///   m[q,j]   -- qubit q resident in subcircuit j
///   ld[q,j]  -- qubit q loaded at start of subcircuit j
///   st[q,j]  -- qubit q stored after subcircuit j-1 (only j >= 1)
///   u[j]     -- subcircuit j is used (for symmetry breaking)
///
/// Objective: minimize sum of ld + sum of st.
pub fn optimize_partition(
    circ: &Circuit,
    proc_cap: usize,
    max_subcircuits: usize,
    initial_best_cost: Option<usize>,
    timeout_secs: Option<u64>,
) -> Option<Vec<Circuit>> {
    if circ.gates.is_empty() || max_subcircuits == 0 {
        return Some(vec![]);
    }

    let gates: Vec<Vec<Qubit>> = circ.gates.iter().map(|g| g.qubits()).collect();
    let n_gates = gates.len();
    let mut qubits: Vec<Qubit> = circ.qubits.iter().cloned().collect();
    qubits.sort_unstable_by_key(|q| q.0);
    let n_qubits = qubits.len();
    let qubit_idx: HashMap<Qubit, usize> =
        qubits.iter().enumerate().map(|(i, q)| (*q, i)).collect();

    let dag = circ.to_dag();
    let edges: Vec<(usize, usize)> = (0..dag.n)
        .flat_map(|u| dag.successors[u].iter().map(move |v| (u, v.0)))
        .collect();

    let max_k = max_subcircuits;

    // -------- Build the MIP via highs::RowProblem --------
    // RowProblem lets us add columns (variables) first, then rows (constraints) referencing them
    // by column index. We track all column indices in dense Vec layouts mirroring the SAT version.

    let mut pb = RowProblem::default();

    // Helper to allocate a binary column with a coefficient in the objective.
    // Bounds are [0, 1] and integrality is set per column.
    let add_binary = |pb: &mut RowProblem, obj_coef: f64| -> highs::Col {
        pb.add_integer_column(obj_coef, 0..=1)
    };
    let add_continuous = |pb: &mut RowProblem, obj_coef: f64| -> highs::Col {
        pb.add_column(obj_coef, 0.0..=1.0)
    };

    // g[i][j]
    let g: Vec<Vec<highs::Col>> = (0..n_gates)
        .map(|_| (0..max_k).map(|_| add_binary(&mut pb, 0.0)).collect())
        .collect();
    // m[q][j]
    let m: Vec<Vec<highs::Col>> = (0..n_qubits)
        .map(|_| (0..max_k).map(|_| add_binary(&mut pb, 0.0)).collect())
        .collect();
    // ld[q][j] -- objective coefficient 1.0 for every entry
    let ld: Vec<Vec<highs::Col>> = (0..n_qubits)
        .map(|_| (0..max_k).map(|_| add_continuous(&mut pb, 1.0)).collect())
        .collect();
    // st[q][j] -- objective coefficient 1.0 for j >= 1, 0.0 for j == 0 (unused but keeps indexing simple)
    let st: Vec<Vec<highs::Col>> = (0..n_qubits)
        .map(|_| {
            (0..max_k)
                .map(|j| add_continuous(&mut pb, if j >= 1 { 1.0 } else { 0.0 }))
                .collect()
        })
        .collect();
    // u[j]
    let u: Vec<highs::Col> = (0..max_k).map(|_| add_binary(&mut pb, 0.0)).collect();

    // -------- Constraints --------

    // 1. Each gate placed in exactly one subcircuit:  sum_j g[i,j] == 1
    for i in 0..n_gates {
        let row: Vec<(highs::Col, f64)> = (0..max_k).map(|j| (g[i][j], 1.0)).collect();
        pb.add_row(1.0..=1.0, &row);
    }

    // 2. DAG order: for each edge (u→v), u's slot ≤ v's slot.
    //    Encoded as: sum_j j*g[u,j] <= sum_j j*g[v,j]
    //    One constraint per edge (vs. O(max_k) dense rows before), much tighter LP relaxation.
    for &(uu, vv) in &edges {
        // j=0 terms have coefficient 0 so start from j=1.
        if max_k <= 1 {
            continue;
        }
        let row: Vec<(highs::Col, f64)> = (1..max_k)
            .flat_map(|j| [(g[uu][j], j as f64), (g[vv][j], -(j as f64))])
            .collect();
        pb.add_row(f64::NEG_INFINITY..=0.0, &row);
    }

    // 3. Gate-executable: g[i,j] - m[qubit_idx[q], j] <= 0  for each q in gate_i
    for (i, gq) in gates.iter().enumerate() {
        for j in 0..max_k {
            for &q in gq {
                let qi = qubit_idx[&q];
                pb.add_row(
                    f64::NEG_INFINITY..=0.0,
                    &[(g[i][j], 1.0), (m[qi][j], -1.0)],
                );
            }
        }
    }

    // 4. Capacity per subcircuit: sum_q m[q,j] <= proc_cap
    if proc_cap < n_qubits {
        for j in 0..max_k {
            let row: Vec<(highs::Col, f64)> = (0..n_qubits).map(|q| (m[q][j], 1.0)).collect();
            pb.add_row(f64::NEG_INFINITY..=(proc_cap as f64), &row);
        }
    }

    // 5. Load/store as transitions (one-sided is sufficient because the objective drives them down):
    //    ld[q,0] >= m[q,0]                    -> m[q,0] - ld[q,0] <= 0
    //    ld[q,j] >= m[q,j] - m[q,j-1]         -> m[q,j] - m[q,j-1] - ld[q,j] <= 0
    //    st[q,j] >= m[q,j-1] - m[q,j]         -> m[q,j-1] - m[q,j] - st[q,j] <= 0
    for q in 0..n_qubits {
        pb.add_row(
            f64::NEG_INFINITY..=0.0,
            &[(m[q][0], 1.0), (ld[q][0], -1.0)],
        );
        for j in 1..max_k {
            pb.add_row(
                f64::NEG_INFINITY..=0.0,
                &[(m[q][j], 1.0), (m[q][j - 1], -1.0), (ld[q][j], -1.0)],
            );
            pb.add_row(
                f64::NEG_INFINITY..=0.0,
                &[(m[q][j - 1], 1.0), (m[q][j], -1.0), (st[q][j], -1.0)],
            );
        }
    }

    // 6. Symmetry breaking: u[j] >= g[i,j], and u[j+1] <= u[j].
    //    Forward direction (u[j] => some gate placed) is implied by minimization + the next bound.
    for j in 0..max_k {
        for i in 0..n_gates {
            // g[i,j] - u[j] <= 0
            pb.add_row(f64::NEG_INFINITY..=0.0, &[(g[i][j], 1.0), (u[j], -1.0)]);
        }
    }
    for j in 0..(max_k.saturating_sub(1)) {
        // u[j+1] - u[j] <= 0
        pb.add_row(f64::NEG_INFINITY..=0.0, &[(u[j + 1], 1.0), (u[j], -1.0)]);
    }

    // 7. Optional cutoff from initial best cost: objective <= initial_best_cost.
    //    Encoded as a constraint over all ld/st columns.
    if let Some(ub) = initial_best_cost {
        let mut cutoff_row: Vec<(highs::Col, f64)> =
            Vec::with_capacity(n_qubits * max_k + n_qubits * (max_k - 1).max(0));
        for q in 0..n_qubits {
            for j in 0..max_k {
                cutoff_row.push((ld[q][j], 1.0));
            }
            for j in 1..max_k {
                cutoff_row.push((st[q][j], 1.0));
            }
        }
        pb.add_row(f64::NEG_INFINITY..=(ub as f64), &cutoff_row);
    }

    // -------- Solve --------
    let mut model = pb.optimise(Sense::Minimise);

    // Time limit. HiGHS option key is "time_limit" (seconds, double).
    if let Some(t) = timeout_secs {
        model.set_option("time_limit", t as f64);
    }
    // Quiet by default; flip to true if you want HiGHS log output.
    model.set_option("output_flag", true);
    // Optional: parallelise.
    model.set_option("parallel", "on");

    let solved = model.solve();
    let status = solved.status();

    // Acceptable terminal statuses for "we have something usable":
    //   Optimal, ReachedTimeLimit (with feasible incumbent), ReachedObjectiveBound, etc.
    // Unacceptable:
    //   Infeasible, ModelError, etc.
    match status {
        HighsModelStatus::Optimal
        | HighsModelStatus::ReachedTimeLimit
        | HighsModelStatus::ReachedIterationLimit
        | HighsModelStatus::ObjectiveBound => {
            let solution = solved.get_solution();
            let cols = solution.columns();

            // If timed out with no feasible incumbent, columns will all be zero/NaN; check
            // the objective and the gate-assignment validity to decide.
            // The simplest robust check: every gate must have exactly one j with g[i,j] >= 0.5.
            let mut assignment: Vec<usize> = Vec::with_capacity(n_gates);
            for i in 0..n_gates {
                let chosen = (0..max_k).find(|&j| cols[g[i][j].index()] > 0.5);
                match chosen {
                    Some(j) => assignment.push(j),
                    None => {
                        // No feasible incumbent recovered.
                        eprintln!(
                            "MIP partition: status {:?} but no feasible incumbent for gate {i}",
                            status
                        );
                        return None;
                    }
                }
            }

            let total_ops: f64 = (0..n_qubits)
                .map(|q| {
                    (0..max_k).map(|j| cols[ld[q][j].index()]).sum::<f64>()
                        + (1..max_k).map(|j| cols[st[q][j].index()]).sum::<f64>()
                })
                .sum();
            println!(
                "MIP status: {:?}, total l/s ops = {:.0}",
                status,
                total_ops.round()
            );

            let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_k];
            for (gate_idx, &sub_idx) in assignment.iter().enumerate() {
                groups[sub_idx].push(gate_idx);
            }

            Some(
                groups
                    .into_iter()
                    .filter(|v| !v.is_empty())
                    .map(|indices| {
                        let mut sub = Circuit::new(circ.num_qubits);
                        for idx in indices {
                            sub.apply(circ.gates[idx].clone());
                        }
                        sub
                    })
                    .collect(),
            )
        }
        HighsModelStatus::Infeasible => {
            // This shouldn't happen unless max_subcircuits is too small for the circuit
            // or initial_best_cost is below the true optimum.
            eprintln!("MIP partition: infeasible (max_subcircuits or initial_best_cost too tight?)");
            None
        }
        other => {
            eprintln!("MIP partition: solver returned status {:?}", other);
            None
        }
    }
}