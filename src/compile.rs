use crate::circuit::{Circuit, Gate, GateId, Qubit, shift_register_hashmap};
use crate::pbc::{
    AllOf, ArchitectureQubit, CliffordFrame, MeasId,
    PPRAngle::{PiOver2, PiOver4, PiOver8},
    Pauli, PauliAxis, PauliProductCircuit, PauliProductOperation, PauliString,
    Sign::{NegOne, One},
};
use std::{
    collections::{HashMap, HashSet},
    fmt,
};

#[derive(Clone)]
pub struct LoadStoreCircuit {
    pub ops: Vec<LoadStoreOp>,
}

#[derive(Clone)]
pub enum LoadStoreOp {
    Load(ArchitectureQubit, ArchitectureQubit),
    Store(ArchitectureQubit, ArchitectureQubit),
    Gate(Gate<ArchitectureQubit>),
}

impl fmt::Display for LoadStoreOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadStoreOp::Load(src, dst) => write!(f, "Load({src} -> {dst})"),
            LoadStoreOp::Store(src, dst) => write!(f, "Store({src} -> {dst})"),
            LoadStoreOp::Gate(g) => write!(f, "{g}"),
        }
    }
}

impl fmt::Display for LoadStoreCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for op in &self.ops {
            writeln!(f, "{op}")?;
        }
        Ok(())
    }
}

fn get_processor_subcircuits(circ: Circuit, max_size: usize) -> Vec<Circuit> {
    let dag = circ.to_dag();
    let n = dag.n;
    let successors = dag.successors;
    let mut in_degree = dag.in_degree;

    let mut ready: HashSet<GateId> = (0..n)
        .map(GateId)
        .filter(|id| in_degree[id.0] == 0)
        .collect();
    let mut gates: Vec<Option<Gate<Qubit>>> = circ.gates.into_iter().map(Some).collect();
    // Precompute once; avoids repeated Vec allocation inside the hot inner loop.
    let gate_qubits: Vec<Vec<Qubit>> = gates.iter().map(|g| g.as_ref().unwrap().qubits()).collect();
    let mut subcircuits: Vec<Circuit> = Vec::new();
    let mut total_assigned = 0;

    while total_assigned < n {
        let mut active_qubits: HashSet<Qubit> = HashSet::new();
        let mut subcircuit_gates: Vec<Gate<Qubit>> = Vec::new();
        let mut made_progress = true;

        while made_progress {
            made_progress = false;
            let best = ready
                .iter()
                .filter_map(|id| {
                    let gq = &gate_qubits[id.0];
                    let new = gq.iter().filter(|q| !active_qubits.contains(q)).count();
                    if active_qubits.len() + new <= max_size {
                        Some((*id, new))
                    } else {
                        None
                    }
                })
                .min_by_key(|&(id, new)| (new, id.0))
                .map(|(id, _)| id);

            if let Some(id) = best {
                active_qubits.extend(gate_qubits[id.0].iter().copied());
                subcircuit_gates.push(gates[id.0].take().unwrap());
                ready.remove(&id);
                total_assigned += 1;
                made_progress = true;

                for &succ in &successors[id.0] {
                    in_degree[succ.0] -= 1;
                    if in_degree[succ.0] == 0 {
                        ready.insert(succ);
                    }
                }
            }
        }

        assert!(
            !subcircuit_gates.is_empty(),
            "cycle in gate dependency graph"
        );
        let num_qubits = active_qubits.len();
        subcircuits.push(Circuit {
            gates: subcircuit_gates,
            qubits: active_qubits,
            num_qubits,
        });
    }
    subcircuits
}

fn gate_to_pbc_instructions(
    gate: &Gate<ArchitectureQubit>,
    next_id: &mut u32,
) -> Vec<PauliProductOperation> {
    let mut alloc = || {
        let id = MeasId(*next_id);
        *next_id += 1;
        id
    };

    match gate {
        Gate::X(q) => vec![PauliProductOperation::Rotation {
            axis: PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(*q, Pauli::X)]),
            },
            angle: PiOver2,
        }],
        Gate::Y(q) => vec![PauliProductOperation::Rotation {
            axis: PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(*q, Pauli::Y)]),
            },
            angle: PiOver2,
        }],
        Gate::Z(q) => vec![PauliProductOperation::Rotation {
            axis: PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
            },
            angle: PiOver2,
        }],
        Gate::H(q) => vec![
            PauliProductOperation::Rotation {
                axis: PauliAxis {
                    sign: One,
                    pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
                },
                angle: PiOver4,
            },
            PauliProductOperation::Rotation {
                axis: PauliAxis {
                    sign: One,
                    pauli_string: PauliString::new(vec![(*q, Pauli::X)]),
                },
                angle: PiOver4,
            },
            PauliProductOperation::Rotation {
                axis: PauliAxis {
                    sign: One,
                    pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
                },
                angle: PiOver4,
            },
        ],
        Gate::S(q) => vec![PauliProductOperation::Rotation {
            axis: PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
            },
            angle: PiOver4,
        }],
        Gate::Sdg(q) => vec![PauliProductOperation::Rotation {
            axis: PauliAxis {
                sign: NegOne,
                pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
            },
            angle: PiOver4,
        }],
        Gate::T(q) => {
            let m0 = alloc();
            let m1 = alloc();
            vec![
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![
                            (*q, Pauli::Z),
                            (ArchitectureQubit::Magic(0), Pauli::X),
                        ]),
                    },
                    id: m0,
                },
                // S correction on q conditioned on m0 (Clifford correction for T-gate injection).
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
                    },
                    angle: PiOver4,
                    condition: AllOf::single(m0),
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(
                            ArchitectureQubit::Magic(0),
                            Pauli::X,
                        )]),
                    },
                    id: m1,
                },
                // Disentangle magic
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf::single(m1),
                },
            ]
        }
        Gate::Tdg(q) => {
            let m0 = alloc();
            let m1 = alloc();
            vec![
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: NegOne,
                        pauli_string: PauliString::new(vec![
                            (*q, Pauli::Z),
                            (ArchitectureQubit::Magic(0), Pauli::X),
                        ]),
                    },
                    id: m0,
                },
                // S correction on q conditioned on m0.
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: NegOne,
                        pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
                    },
                    angle: PiOver4,
                    condition: AllOf::single(m0),
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(
                            ArchitectureQubit::Magic(0),
                            Pauli::X,
                        )]),
                    },
                    id: m1,
                },
                // Disentangle magic
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*q, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf::single(m1),
                },
            ]
        }
        Gate::CNOT { control, target } => vec![
            PauliProductOperation::Rotation {
                axis: PauliAxis {
                    sign: One,
                    pauli_string: PauliString::new(vec![(*control, Pauli::Z), (*target, Pauli::X)]),
                },
                angle: PiOver4,
            },
            PauliProductOperation::Rotation {
                axis: PauliAxis {
                    sign: NegOne,
                    pauli_string: PauliString::new(vec![(*control, Pauli::Z)]),
                },
                angle: PiOver4,
            },
            PauliProductOperation::Rotation {
                axis: PauliAxis {
                    sign: NegOne,
                    pauli_string: PauliString::new(vec![(*target, Pauli::X)]),
                },
                angle: PiOver4,
            },
        ],
        // See fig 14 in active volume paper
        Gate::CCZ {
            control1,
            control2,
            target,
        } => {
            let m0 = alloc(); // Z[c1]⊗X[Magic0]
            let m1 = alloc(); // Z[c2]⊗X[Magic1]
            let m2 = alloc(); // Z[t]⊗X[Magic2]
            let m3 = alloc(); // X[Magic0]
            let m4 = alloc(); // X[Magic1]
            let m5 = alloc(); // X[Magic2]
            vec![
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![
                            (*control1, Pauli::Z),
                            (ArchitectureQubit::Magic(0), Pauli::X),
                        ]),
                    },
                    id: m0,
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![
                            (*control2, Pauli::Z),
                            (ArchitectureQubit::Magic(1), Pauli::X),
                        ]),
                    },
                    id: m1,
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![
                            (*target, Pauli::Z),
                            (ArchitectureQubit::Magic(2), Pauli::X),
                        ]),
                    },
                    id: m2,
                },
                // conditional cz ctrl2, tar
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![
                            (*control2, Pauli::Z),
                            (*target, Pauli::Z),
                        ]),
                        sign: One,
                    },
                    condition: AllOf(vec![m0]),
                },
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![(*control2, Pauli::Z)]),
                        sign: NegOne,
                    },
                    condition: AllOf(vec![m0]),
                },
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![(*target, Pauli::Z)]),
                        sign: NegOne,
                    },
                    condition: AllOf(vec![m0]),
                },
                // conditional cz ctrl1, tar
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![
                            (*control1, Pauli::Z),
                            (*target, Pauli::Z),
                        ]),
                        sign: One,
                    },
                    condition: AllOf(vec![m1]),
                },
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![(*control1, Pauli::Z)]),
                        sign: NegOne,
                    },
                    condition: AllOf(vec![m1]),
                },
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![(*target, Pauli::Z)]),
                        sign: NegOne,
                    },
                    condition: AllOf(vec![m1]),
                },
                // conditional cz ctrl1, ctrl2
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![
                            (*control1, Pauli::Z),
                            (*control2, Pauli::Z),
                        ]),
                        sign: One,
                    },
                    condition: AllOf(vec![m2]),
                },
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![(*control1, Pauli::Z)]),
                        sign: NegOne,
                    },
                    condition: AllOf(vec![m2]),
                },
                PauliProductOperation::ConditionalRotation {
                    angle: PiOver4,
                    axis: PauliAxis {
                        pauli_string: PauliString::new(vec![(*control2, Pauli::Z)]),
                        sign: NegOne,
                    },
                    condition: AllOf(vec![m1]),
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(
                            ArchitectureQubit::Magic(0),
                            Pauli::X,
                        )]),
                    },
                    id: m3,
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(
                            ArchitectureQubit::Magic(1),
                            Pauli::X,
                        )]),
                    },
                    id: m4,
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(
                            ArchitectureQubit::Magic(2),
                            Pauli::X,
                        )]),
                    },
                    id: m5,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*control1, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m1, m2, m3]),
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*control2, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m0, m2, m4]),
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*target, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m0, m1, m5]),
                },
            ]
        }
    }
}

fn ls_circuit_op_to_pbc_op(op: LoadStoreOp, next_id: &mut u32) -> Vec<PauliProductOperation> {
    let mut alloc = || {
        let id = MeasId(*next_id);
        *next_id += 1;
        id
    };

    match op {
        LoadStoreOp::Load(mem, proc) => {
            let m0 = alloc();
            let m1 = alloc();
            vec![
                PauliProductOperation::FrameReset(proc),
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(mem, Pauli::Z), (proc, Pauli::Z)]),
                    },
                    id: m0,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(proc, Pauli::X)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m0]),
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(mem, Pauli::X)]),
                    },
                    id: m1,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(proc, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m1]),
                },
            ]
        }
        LoadStoreOp::Store(proc, mem) => {
            let m0 = alloc();
            let m1 = alloc();
            vec![
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(proc, Pauli::Z), (mem, Pauli::Z)]),
                    },
                    id: m0,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(mem, Pauli::X)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m0]),
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(proc, Pauli::X)]),
                    },
                    id: m1,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(mem, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m1]),
                },
            ]
        }
        LoadStoreOp::Gate(gate) => gate_to_pbc_instructions(&gate, next_id),
    }
}

fn absorb_cliffords(circ: PauliProductCircuit) -> PauliProductCircuit {
    let mut result = PauliProductCircuit::new();
    result.next_meas_id = circ.next_meas_id;
    let mut frame = CliffordFrame::new();

    for instr in circ.instructions {
        match instr {
            PauliProductOperation::FrameReset(q) => {
                frame.reset(q);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver4,
            } => {
                frame.update(&axis);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver2,
            } => {
                frame.update(&axis);
                frame.update(&axis);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver8,
            } => {
                result.instructions.push(PauliProductOperation::Rotation {
                    axis: frame.apply(&axis),
                    angle: PiOver8,
                });
            }
            PauliProductOperation::ConditionalRotation { .. } => {
                // Dropped for now.
            }
            PauliProductOperation::Measurement { axis, id } => {
                let effective = frame.apply(&axis);
                result
                    .instructions
                    .push(PauliProductOperation::Measurement {
                        axis: effective,
                        id,
                    });
            }
        }
    }

    result
}

fn to_pauli_product_circuit(
    load_store: LoadStoreCircuit,
    simulate_corrections: bool,
) -> PauliProductCircuit {
    let mut circuit = PauliProductCircuit::new();
    for op in load_store.ops {
        let instrs = ls_circuit_op_to_pbc_op(op, &mut circuit.next_meas_id);
        circuit.instructions.extend(instrs.into_iter().filter(|i| {
            simulate_corrections || !matches!(i, PauliProductOperation::ConditionalRotation { .. })
        }));
    }
    circuit
}

fn resolve_corrections(circ: PauliProductCircuit) -> PauliProductCircuit {
    use std::collections::HashMap;

    let mut outcomes: HashMap<crate::pbc::MeasId, bool> = HashMap::new();
    for instr in &circ.instructions {
        if let PauliProductOperation::ConditionalRotation { condition: AllOf(ids), .. } = instr {
            for &id in ids {
                outcomes.entry(id).or_insert_with(rand::random);
            }
        }
    }

    let mut result = PauliProductCircuit::new();
    result.next_meas_id = circ.next_meas_id;
    for instr in circ.instructions {
        match instr {
            PauliProductOperation::ConditionalRotation { axis, angle, condition: AllOf(ref ids) } => {
                if ids.iter().all(|id| *outcomes.get(id).unwrap_or(&false)) {
                    result.instructions.push(PauliProductOperation::Rotation { axis, angle });
                }
            }
            other => result.instructions.push(other),
        }
    }
    result
}

fn to_load_store_circuit(subcircuits: Vec<Circuit>) -> LoadStoreCircuit {
    let mut ops = Vec::new();
    for sub in subcircuits {
        let mut circuit_to_processor_qubit: HashMap<Qubit, ArchitectureQubit> = HashMap::new();
        let mut loaded_count = 0;
        let mut sorted_qubits: Vec<_> = sub.qubits.iter().collect();
        sorted_qubits.sort_unstable_by_key(|q| q.0);
        for q in sorted_qubits {
            ops.push(LoadStoreOp::Load(
                ArchitectureQubit::Memory(q.0),
                ArchitectureQubit::Processor(loaded_count),
            ));
            circuit_to_processor_qubit.insert(*q, ArchitectureQubit::Processor(loaded_count));
            loaded_count += 1;
        }
        for gate in sub.gates {
            ops.push(LoadStoreOp::Gate(shift_register_hashmap(
                &gate,
                &circuit_to_processor_qubit,
            )));
        }
        let mut sorted_stores: Vec<_> = circuit_to_processor_qubit.iter().collect();
        sorted_stores.sort_unstable_by_key(|(q, _)| q.0);
        for (circ_qubit, proc_qubit) in sorted_stores {
            ops.push(LoadStoreOp::Store(
                *proc_qubit,
                ArchitectureQubit::Memory(circ_qubit.0),
            ));
        }
    }
    LoadStoreCircuit { ops }
}

pub fn compile(
    circ: Circuit,
    max_subcircuit_size: usize,
    simulate_corrections: bool,
) -> PauliProductCircuit {
    let subcircuits = get_processor_subcircuits(circ, max_subcircuit_size);
    let load_store = to_load_store_circuit(subcircuits);
    let pbc = to_pauli_product_circuit(load_store, simulate_corrections);
    let resolved = if simulate_corrections { resolve_corrections(pbc) } else { pbc };
    absorb_cliffords(resolved)
}

/// Returns (load_store, pbc_pre_clifford, pbc_final) for inspecting all intermediate stages.
pub fn compile_steps(
    circ: Circuit,
    max_subcircuit_size: usize,
    simulate_corrections: bool,
) -> (LoadStoreCircuit, PauliProductCircuit, PauliProductCircuit) {
    let subcircuits = get_processor_subcircuits(circ, max_subcircuit_size);
    let load_store = to_load_store_circuit(subcircuits);
    let load_store_saved = load_store.clone();
    let pbc = to_pauli_product_circuit(load_store, simulate_corrections);
    let resolved = if simulate_corrections { resolve_corrections(pbc.clone()) } else { pbc.clone() };
    let clifford_free = absorb_cliffords(resolved);
    (load_store_saved, pbc, clifford_free)
}
