use crate::circuit::{Circuit, Qubit};
use rustsat::{
    clause,
    encodings::card::{BoundUpper, Totalizer},
    instances::{BasicVarManager, Cnf},
    solvers::{Interrupt, InterruptSolver, Solve, SolveIncremental, SolverResult},
    types::{Clause, Lit, TernaryVal, Var},
};
use rustsat_cadical::CaDiCaL;
use std::{
    collections::HashMap,
    sync::mpsc,
    time::{Duration, Instant},
};

/// SAT-based optimal circuit partitioner. Finds the assignment of gates to subcircuits
/// that minimizes total load/store operations, subject to processor capacity constraints.
///
/// This is a port of the Python `optimize()` function in `satstore/main.py`, preserving
/// the incremental totalizer + assumption-based optimization pattern (ITotalizer equivalent).
pub fn optimize_partition(
    circ: &Circuit,
    proc_cap: usize,
    max_subcircuits: usize,
    initial_best_cost: Option<usize>,
    timeout_secs: Option<u64>,
    pre_loaded: &HashMap<Qubit, bool>,
) -> Option<(Vec<Circuit>, HashMap<Qubit, bool>)> {
    if circ.gates.is_empty() || max_subcircuits == 0 {
        return Some((vec![], HashMap::new()));
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

    // Variable layout — flat u32 index into SAT variable space:
    //   gate(i, j)  = i * max_k + j
    //   mem(q, j)   = n_gates*max_k + q*max_k + j
    //   load(q, j)  = n_gates*max_k + n_qubits*max_k + q*max_k + j
    //   store(q, j) = n_gates*max_k + 2*n_qubits*max_k + q*max_k + j
    //   used(j)     = n_gates*max_k + 3*n_qubits*max_k + j
    let base_mem: u32 = (n_gates * max_subcircuits) as u32;
    let base_load: u32 = base_mem + (n_qubits * max_subcircuits) as u32;
    let base_store: u32 = base_load + (n_qubits * max_subcircuits) as u32;
    let base_used: u32 = base_store + (n_qubits * max_subcircuits) as u32;
    let base_mem_pre: u32 = base_used + max_subcircuits as u32;
    let n_base: u32 = base_mem_pre + n_qubits as u32;

    let gate_lit =
        |i: usize, j: usize| -> Lit { Var::new((i * max_subcircuits + j) as u32).pos_lit() };
    let mem_lit = |q: usize, j: usize| -> Lit {
        Var::new(base_mem + (q * max_subcircuits + j) as u32).pos_lit()
    };
    let load_lit = |q: usize, j: usize| -> Lit {
        Var::new(base_load + (q * max_subcircuits + j) as u32).pos_lit()
    };
    let store_lit = |q: usize, j: usize| -> Lit {
        Var::new(base_store + (q * max_subcircuits + j) as u32).pos_lit()
    };
    let used_lit = |j: usize| -> Lit { Var::new(base_used + j as u32).pos_lit() };
    let mem_pre_lit = |q: usize| -> Lit { Var::new(base_mem_pre + q as u32).pos_lit() };

    let mut clauses = Cnf::new();
    let mut var_mgr = BasicVarManager::from_next_free(Var::new(n_base));

    // 1. all_gates_executed: exactly one subcircuit per gate
    for i in 0..n_gates {
        let alo: Clause = (0..max_subcircuits).map(|j| gate_lit(i, j)).collect();
        clauses.add_clause(alo);
        for a in 0..max_subcircuits {
            for b in (a + 1)..max_subcircuits {
                clauses.add_clause(clause![!gate_lit(i, a), !gate_lit(i, b)]);
            }
        }
    }

    // 2. order_preserved: for DAG edge (u,v), u cannot be in a later subcircuit than v
    for &(u, v) in &edges {
        for i in 1..max_subcircuits {
            for j in 0..i {
                clauses.add_clause(clause![!gate_lit(u, i), !gate_lit(v, j)]);
            }
        }
    }

    // 3. gates_executable: gate i in subcircuit j -> all its qubits must be loaded in j
    for (i, gq) in gates.iter().enumerate() {
        for j in 0..max_subcircuits {
            for &q in gq {
                clauses.add_clause(clause![!gate_lit(i, j), mem_lit(qubit_idx[&q], j)]);
            }
        }
    }

    // 4. max_capacity: at most proc_cap qubits loaded per subcircuit (hard constraint)
    for j in 0..max_subcircuits {
        if proc_cap < n_qubits {
            let mem_lits: Vec<Lit> = (0..n_qubits).map(|q| mem_lit(q, j)).collect();
            let mut tot: Totalizer = mem_lits.into_iter().collect();
            tot.encode_ub(0..=proc_cap, &mut clauses, &mut var_mgr)
                .unwrap();
            if let Ok(unit_lits) = tot.enforce_ub(proc_cap) {
                for lit in unit_lits {
                    clauses.add_clause(clause![lit]);
                }
            }
        }
    }

    // 5. must_load_store: uniform transition constraints using mem_pre as the virtual slot -1.
    //   mem_pre(q) ∨ ¬mem(q,0) ∨ load(q,0)       [absent→present at slot 0 = load]
    //   ¬mem_pre(q) ∨ mem(q,0) ∨ store(q,0)       [present→absent at slot 0 = store/evict]
    //   mem(q,i) ∨ ¬mem(q,i+1) ∨ load(q,i+1)     [absent→present = load]
    //   ¬mem(q,i) ∨ mem(q,i+1) ∨ store(q,i+1)    [present→absent = store]
    for q in 0..n_qubits {
        clauses.add_clause(clause![mem_pre_lit(q), !mem_lit(q, 0), load_lit(q, 0)]);
        clauses.add_clause(clause![!mem_pre_lit(q), mem_lit(q, 0), store_lit(q, 0)]);
        // Fix mem_pre based on the pre-loaded state passed in by the caller.
        if *pre_loaded.get(&qubits[q]).unwrap_or(&false) {
            clauses.add_clause(clause![mem_pre_lit(q)]);
        } else {
            clauses.add_clause(clause![!mem_pre_lit(q)]);
        }
    }
    for i in 0..max_subcircuits.saturating_sub(1) {
        for q in 0..n_qubits {
            clauses.add_clause(clause![
                mem_lit(q, i),
                !mem_lit(q, i + 1),
                load_lit(q, i + 1)
            ]);
            clauses.add_clause(clause![
                !mem_lit(q, i),
                mem_lit(q, i + 1),
                store_lit(q, i + 1)
            ]);
        }
    }

    // 5b. Reverse implications: load/store forced false when no transition (tighter propagation).
    //   load(q,0) → mem(q,0)
    //   load(q,i+1) → ¬mem(q,i) ∧ mem(q,i+1)
    //   store(q,i+1) → mem(q,i) ∧ ¬mem(q,i+1)
    // for q in 0..n_qubits {
    //     clauses.add_clause(clause![!load_lit(q, 0), mem_lit(q, 0)]);
    // }
    // for i in 0..max_subcircuits.saturating_sub(1) {
    //     for q in 0..n_qubits {
    //         clauses.add_clause(clause![!load_lit(q, i + 1), !mem_lit(q, i)]);
    //         clauses.add_clause(clause![!load_lit(q, i + 1), mem_lit(q, i + 1)]);
    //         clauses.add_clause(clause![!store_lit(q, i + 1), mem_lit(q, i)]);
    //         clauses.add_clause(clause![!store_lit(q, i + 1), !mem_lit(q, i + 1)]);
    //     }
    // }

    // 6. no-gaps symmetry breaking: occupied slots must form a prefix (empty slots at the end).
    // used[j] ↔ (∃i. gate_lit(i,j))
    // backward: gate_lit(i,j) → used[j]
    for j in 0..max_subcircuits {
        for i in 0..n_gates {
            clauses.add_clause(clause![!gate_lit(i, j), used_lit(j)]);
        }
    }
    // forward: used[j] → ∨_i gate_lit(i,j)
    for j in 0..max_subcircuits {
        let fwd: Clause = std::iter::once(!used_lit(j))
            .chain((0..n_gates).map(|i| gate_lit(i, j)))
            .collect();
        clauses.add_clause(fwd);
    }
    // no-gaps: used[j+1] → used[j]
    for j in 0..max_subcircuits.saturating_sub(1) {
        clauses.add_clause(clause![!used_lit(j + 1), used_lit(j)]);
    }

    let mut solver = CaDiCaL::default();
    for cl in &clauses {
        solver.add_clause_ref(cl).unwrap();
    }

    // Objective: minimize total load + store operations
    // loads: load(q, j) for all q, j in 0..max_k
    // stores: store(q, j) for all q, j in 1..max_k
    let obj_lits: Vec<Lit> = (0..n_qubits)
        .flat_map(|q| (0..max_subcircuits).map(move |j| load_lit(q, j)))
        .chain((0..n_qubits).flat_map(|q| (0..max_subcircuits).map(move |j| store_lit(q, j))))
        .collect();
    let n_obj = obj_lits.len();

    // Build the optimization totalizer (equivalent to PySAT's ITotalizer)
    // encode_ub(0..n_obj) creates output literals for bounds 0..n_obj-1.
    // enforce_ub(k) returns the assumption literal for "sum <= k".
    // enforce_ub(n_obj) returns Ok([]) (trivially true), used for the first unconstrained solve.
    let mut opt_tot: Totalizer = obj_lits.iter().copied().collect();
    if n_obj > 0 {
        let mut tot_cnf = Cnf::new();
        opt_tot
            .encode_ub(0..n_obj, &mut tot_cnf, &mut var_mgr)
            .unwrap();
        for cl in &tot_cnf {
            solver.add_clause_ref(cl).unwrap();
        }
    }

    let mut best_assignment: Option<Vec<usize>> = None;
    let mut best_mem_state: Vec<bool> = vec![false; n_qubits];
    // initial_best_k comes from the greedy Belady count; fall back to n_obj+1 (unconstrained)
    // Start one above the initial bound so the first enforce_ub(initial_best_k) call succeeds
    // (the greedy is a valid solution at exactly that count). Without this, starting at
    // initial_best_k would immediately try enforce_ub(initial_best_k - 1), which is UNSAT
    // when the greedy is already optimal, leaving best_assignment = None.
    let mut best_cost = initial_best_cost.map_or(n_obj + 1, |g| g.saturating_add(1).min(n_obj + 1));
    let deadline = timeout_secs.map(|s| Instant::now() + Duration::from_secs(s));

    loop {
        println!("best_cost so far: {best_cost}");
        if best_cost == 0 {
            break;
        }
        let assumps = match opt_tot.enforce_ub(best_cost - 1) {
            Ok(a) => a,
            Err(_) => break,
        };

        let result = match deadline {
            Some(d) => {
                let remaining = d.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                // Mirror Python's solve_limited + timer.cancel() pattern:
                // use a channel so the timer thread exits immediately when solve returns early.
                let (cancel_tx, cancel_rx) = mpsc::channel::<()>();
                let interrupter = solver.interrupter();
                let handle = std::thread::spawn(move || {
                    if cancel_rx.recv_timeout(remaining).is_err() {
                        interrupter.interrupt();
                    }
                });
                let r = solver.solve_assumps(&assumps).unwrap();
                let _ = cancel_tx.send(());
                handle.join().ok();
                r
            }
            None => solver.solve_assumps(&assumps).unwrap(),
        };

        match result {
            SolverResult::Sat => {
                let count = obj_lits
                    .iter()
                    .filter(|&&lit| solver.lit_val(lit).unwrap() == TernaryVal::True)
                    .count();
                let assignment: Vec<usize> = (0..n_gates)
                    .map(|i| {
                        (0..max_subcircuits)
                            .find(|&j| solver.lit_val(gate_lit(i, j)).unwrap() == TernaryVal::True)
                            .unwrap_or(0)
                    })
                    .collect();
                best_mem_state = (0..n_qubits)
                    .map(|q| solver.lit_val(mem_lit(q, max_subcircuits - 1)).unwrap() == TernaryVal::True)
                    .collect();
                best_assignment = Some(assignment);
                best_cost = count;
            }
            SolverResult::Unsat => {
                println!("Found optimal solution, with {best_cost} l/s ops");
                break;
            }
            SolverResult::Interrupted => {
                println!(
                    "Solver interrupted, returning best solution found with {best_cost} l/s ops"
                );
                break;
            }
        }
    }

    let assignment = best_assignment?;

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_subcircuits];
    for (gate_idx, &sub_idx) in assignment.iter().enumerate() {
        groups[sub_idx].push(gate_idx);
    }

    let subcircuits = groups
        .into_iter()
        .filter(|v| !v.is_empty())
        .map(|indices| {
            let mut sub = Circuit::new(circ.num_qubits);
            for idx in indices {
                sub.apply(circ.gates[idx].clone());
            }
            sub
        })
        .collect();

    let final_mem: HashMap<Qubit, bool> =
        qubits.iter().copied().zip(best_mem_state).collect();

    Some((subcircuits, final_mem))
}

/// Splits `circ` into `num_slices` contiguous gate-slices and runs `optimize_partition` on each
/// in sequence. The `timeout_secs` budget is shared globally across all slices.
pub fn slice_and_optimize(
    circ: &Circuit,
    proc_cap: usize,
    max_subcircuits: usize,
    num_slices: usize,
    initial_best_cost: Option<usize>,
    timeout_secs: Option<u64>,
) -> Option<Vec<Circuit>> {
    if circ.gates.is_empty() || num_slices == 0 {
        return Some(vec![]);
    }

    let deadline = timeout_secs.map(|s| Instant::now() + Duration::from_secs(s));
    let n_gates = circ.gates.len();
    let chunk_size = n_gates.div_ceil(num_slices);

    let mut result = Vec::new();
    let mut pre_loaded: HashMap<Qubit, bool> = HashMap::new();

    for chunk_start in (0..n_gates).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n_gates);
        let slice_gate_count = chunk_end - chunk_start;
        let remaining_total = deadline.map(|d| d.saturating_duration_since(Instant::now()));
        if matches!(remaining_total, Some(d) if d.is_zero()) {
            return None;
        }
        let slice_budget = remaining_total.map(|r| {
            let gates_left = n_gates - chunk_start;
            let frac = slice_gate_count as f64 / gates_left as f64;
            (r.as_secs_f64() * frac) as u64
        });

        let mut slice_circ = Circuit::new(circ.num_qubits);
        for idx in chunk_start..chunk_end {
            slice_circ.apply(circ.gates[idx].clone());
        }

        let (partitioned, final_mem) = optimize_partition(
            &slice_circ,
            proc_cap,
            max_subcircuits,
            initial_best_cost,
            slice_budget,
            &pre_loaded,
        )?;

        pre_loaded = final_mem;
        result.extend(partitioned);
    }

    Some(result)
}
