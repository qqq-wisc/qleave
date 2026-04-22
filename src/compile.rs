use crate::circuit::{Circuit, Gate, GateId, Qubit, shift_register_hashmap};
use crate::pbc::{
    ArchitectureQubit,
    PPRAngle::{PiOver2, PiOver4, PiOver8},
    Pauli, PauliAxis, PauliProductCircuit, PauliProductOperation, PauliString, Sign,
    Sign::{NegOne, One},
    axes_commute, pauli_string_mult,
};
use std::{collections::{HashMap, HashSet}, fmt};

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
    let mut assigned = vec![false; n];
    let mut gates: Vec<Option<Gate<Qubit>>> = circ.gates.into_iter().map(Some).collect();
    let mut subcircuits: Vec<Circuit> = Vec::new();
    let mut total_assigned = 0;

    while total_assigned < n {
        let mut active_qubits: HashSet<Qubit> = HashSet::new();
        let mut subcircuit_gates: Vec<Gate<Qubit>> = Vec::new();
        let mut made_progress = true;

        while made_progress {
            made_progress = false;

            let gate_qubits = |id: &GateId| -> HashSet<Qubit> {
                gates[id.0].as_ref().unwrap().qubits().into_iter().collect()
            };
            let fits = |id: &&GateId| active_qubits.union(&gate_qubits(id)).count() <= max_size;
            let new_qubit_cost = |id: &&GateId| {
                let gq = gate_qubits(id);
                (gq.difference(&active_qubits).count(), id.0)
            };
            let best = ready
                .iter()
                .filter(|id| !assigned[id.0])
                .filter(fits)
                .min_by_key(new_qubit_cost)
                .copied();

            if let Some(id) = best {
                let gate_qubits: HashSet<Qubit> =
                    gates[id.0].as_ref().unwrap().qubits().into_iter().collect();
                active_qubits.extend(gate_qubits);
                subcircuit_gates.push(gates[id.0].take().unwrap());
                assigned[id.0] = true;
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

fn gate_to_pbc_instructions(gate: &Gate<ArchitectureQubit>) -> Vec<PauliProductOperation> {
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
        Gate::T(q) => vec![
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![
                    (*q, Pauli::Z),
                    (ArchitectureQubit::Magic(0), Pauli::X),
                ]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(ArchitectureQubit::Magic(0), Pauli::X)]),
            }),
        ],
        Gate::Tdg(q) => vec![
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![
                    (*q, Pauli::Z),
                    (ArchitectureQubit::Magic(0), Pauli::X),
                ]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(ArchitectureQubit::Magic(0), Pauli::X)]),
            }),
        ],
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
        Gate::CCZ {
            control1,
            control2,
            target,
        } => vec![
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![
                    (*control1, Pauli::Z),
                    (ArchitectureQubit::Magic(0), Pauli::X),
                ]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![
                    (*control2, Pauli::Z),
                    (ArchitectureQubit::Magic(1), Pauli::X),
                ]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![
                    (*target, Pauli::Z),
                    (ArchitectureQubit::Magic(2), Pauli::X),
                ]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(ArchitectureQubit::Magic(0), Pauli::X)]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(ArchitectureQubit::Magic(1), Pauli::X)]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(ArchitectureQubit::Magic(2), Pauli::X)]),
            }),
        ],
    }
}

fn ls_circuit_op_to_pbc_op(op: LoadStoreOp) -> Vec<PauliProductOperation> {
    match op {
        LoadStoreOp::Load(mem, proc) => {
            vec![
                PauliProductOperation::Measurement(PauliAxis {
                    sign: One,
                    pauli_string: PauliString::new(vec![(mem, Pauli::Z), (proc, Pauli::Z)]),
                }),
                PauliProductOperation::Measurement(PauliAxis {
                    sign: One,
                    pauli_string: PauliString::new(vec![(mem, Pauli::X)]),
                }),
            ]
        }
        LoadStoreOp::Store(proc, mem) => vec![
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(proc, Pauli::Z), (mem, Pauli::Z)]),
            }),
            PauliProductOperation::Measurement(PauliAxis {
                sign: One,
                pauli_string: PauliString::new(vec![(proc, Pauli::X)]),
            }),
        ],
        LoadStoreOp::Gate(gate) => gate_to_pbc_instructions(&gate),
    }
}

fn absorb_into_measurement(circ: PauliProductCircuit) -> PauliProductCircuit {
    let mut instructions = Vec::new();

    let mut pending = Vec::new();
    for instr in circ.instructions {
        match instr {
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver4,
            } => {
                pending.push(axis);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver2,
            } => {
                let copy = axis.clone();
                pending.push(axis);
                pending.push(copy);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver8,
            } => {
                eprintln!(
                    "Warning: π/8 rotations cannot be absorbed into measurements and will be emitted as separate instructions."
                );
                instructions.push(PauliProductOperation::Rotation {
                    axis,
                    angle: PiOver8,
                });
            }

            PauliProductOperation::Measurement(mut axis) => {
                // Litinski rule: for each pending rotation P against measurement Q:
                //   anti-commutes → Q becomes P·Q, rotation consumed
                //   commutes, overlaps measurement qubits → rotation consumed (passes through as trivial correction)
                //   no qubit overlap → rotation survives for the next measurement

                for to_absorb in pending.iter().rev() {
                    let p = &to_absorb.pauli_string;
                    let q = &axis.pauli_string;
                    if !axes_commute(p, q) {
                        let product_axis = pauli_string_mult(&p, &q);
                        axis.sign = axis.sign * to_absorb.sign * product_axis.sign * Sign::J;
                        axis.pauli_string = product_axis.pauli_string;
                    }
                }
                let meas_qubits: HashSet<ArchitectureQubit> =
                    axis.pauli_string.iter().map(|&(q, _)| q).collect();
                instructions.push(PauliProductOperation::Measurement(axis));
                pending.retain(|rot| {
                    let rot_qubits: HashSet<ArchitectureQubit> =
                        rot.pauli_string.iter().map(|&(q, _)| q).collect();
                    !rot_qubits.is_subset(&meas_qubits)
                });
            }
        }
    }

    PauliProductCircuit { instructions }
}

fn to_pauli_product_circuit(load_store: LoadStoreCircuit) -> PauliProductCircuit {
    let mut instructions = Vec::new();
    for op in load_store.ops {
        instructions.extend_from_slice(&ls_circuit_op_to_pbc_op(op));
    }
    PauliProductCircuit { instructions }
}

fn to_load_store_circuit(subcircuits: Vec<Circuit>) -> LoadStoreCircuit {
    let mut ops = Vec::new();
    let mut circuit_to_processor_qubit: HashMap<Qubit, ArchitectureQubit> = HashMap::new();
    for sub in subcircuits {
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
        for (circ_qubit, proc_qubit) in circuit_to_processor_qubit.iter() {
            ops.push(LoadStoreOp::Store(
                *proc_qubit,
                ArchitectureQubit::Memory(circ_qubit.0),
            ));
        }
    }
    LoadStoreCircuit { ops }
}

pub fn compile(circ: Circuit, max_subcircuit_size: usize) -> PauliProductCircuit {
    let subcircuits = get_processor_subcircuits(circ, max_subcircuit_size);
    let load_store = to_load_store_circuit(subcircuits);
    let pbc = to_pauli_product_circuit(load_store);
    absorb_into_measurement(pbc)
}

/// Returns (load_store, pbc_pre_clifford, pbc_final) for inspecting all intermediate stages.
pub fn compile_steps(
    circ: Circuit,
    max_subcircuit_size: usize,
) -> (LoadStoreCircuit, PauliProductCircuit, PauliProductCircuit) {
    let subcircuits = get_processor_subcircuits(circ, max_subcircuit_size);
    let load_store = to_load_store_circuit(subcircuits);
    let load_store_saved = load_store.clone();
    let pbc = to_pauli_product_circuit(load_store);
    let clifford_free = absorb_into_measurement(pbc.clone());
    (load_store_saved, pbc, clifford_free)
}
