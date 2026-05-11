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
        // See fig 14 in active volume paper https://arxiv.org/pdf/2211.15465
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

fn ls_circuit_op_to_pbc_op(op: &LoadStoreOp, next_id: &mut u32) -> Vec<PauliProductOperation> {
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
                PauliProductOperation::FrameReset(*proc),
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*mem, Pauli::Z), (*proc, Pauli::Z)]),
                    },
                    id: m0,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*proc, Pauli::X)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m0]),
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*mem, Pauli::X)]),
                    },
                    id: m1,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*proc, Pauli::Z)]),
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
                        pauli_string: PauliString::new(vec![(*proc, Pauli::Z), (*mem, Pauli::Z)]),
                    },
                    id: m0,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*mem, Pauli::X)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m0]),
                },
                PauliProductOperation::Measurement {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*proc, Pauli::X)]),
                    },
                    id: m1,
                },
                PauliProductOperation::ConditionalRotation {
                    axis: PauliAxis {
                        sign: One,
                        pauli_string: PauliString::new(vec![(*mem, Pauli::Z)]),
                    },
                    angle: PiOver2,
                    condition: AllOf(vec![m1]),
                },
            ]
        }
        LoadStoreOp::Gate(gate) => gate_to_pbc_instructions(gate, next_id),
    }
}

fn absorb_cliffords(circ: &PauliProductCircuit) -> PauliProductCircuit {
    let mut result = PauliProductCircuit::new();
    result.next_meas_id = circ.next_meas_id;
    let mut frame = CliffordFrame::new();

    for ref instr in &circ.instructions {
        match instr {
            PauliProductOperation::FrameReset(q) => {
                frame.reset(*q);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver4,
            } => {
                frame.update(axis);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver2,
            } => {
                frame.update(axis);
                frame.update(axis);
            }
            PauliProductOperation::Rotation {
                axis,
                angle: PiOver8,
            } => {
                result.instructions.push(PauliProductOperation::Rotation {
                    axis: frame.apply(axis),
                    angle: PiOver8,
                });
            }
            PauliProductOperation::ConditionalRotation { .. } => {
                // Dropped for now.
            }
            PauliProductOperation::Measurement { axis, id } => {
                let effective = frame.apply(axis);
                result
                    .instructions
                    .push(PauliProductOperation::Measurement {
                        axis: effective,
                        id: *id,
                    });
            }
        }
    }

    result
}

fn to_pauli_product_circuit(
    load_store: &LoadStoreCircuit,
    simulate_corrections: bool,
) -> PauliProductCircuit {
    let mut circuit = PauliProductCircuit::new();
    for op in load_store.ops.iter() {
        let instrs = ls_circuit_op_to_pbc_op(op, &mut circuit.next_meas_id);
        circuit.instructions.extend(instrs.into_iter().filter(|i| {
            simulate_corrections || !matches!(i, PauliProductOperation::ConditionalRotation { .. })
        }));
    }
    circuit
}

fn resolve_corrections(circ: &PauliProductCircuit) -> PauliProductCircuit {
    use std::collections::HashMap;

    let mut outcomes: HashMap<crate::pbc::MeasId, bool> = HashMap::new();
    for instr in &circ.instructions {
        if let PauliProductOperation::ConditionalRotation {
            condition: AllOf(ids),
            ..
        } = instr
        {
            for &id in ids {
                outcomes.entry(id).or_insert_with(rand::random);
            }
        }
    }

    let mut result = PauliProductCircuit::new();
    result.next_meas_id = circ.next_meas_id;
    for instr in &circ.instructions {
        match instr {
            PauliProductOperation::ConditionalRotation {
                axis,
                angle,
                condition: AllOf(ids),
            } => {
                if ids.iter().all(|id| *outcomes.get(id).unwrap_or(&false)) {
                    result.instructions.push(PauliProductOperation::Rotation {
                        axis: axis.clone(),
                        angle: *angle,
                    });
                }
            }
            other => result.instructions.push(other.clone()),
        }
    }
    result
}

fn to_load_store_circuit(
    subcircuits: &[Circuit],
    max_subcircuit_size: usize,
    skip_redundant: bool,
) -> LoadStoreCircuit {
    let mut ops = Vec::new();
    // Tracks which memory qubits are currently on the processor and their assigned slot.
    // When skip_redundant is false this is always empty at the start of each subcircuit.
    let mut in_flight: HashMap<Qubit, ArchitectureQubit> = HashMap::new();

    // Belady's (OPT) eviction: pre-compute the sorted list of subcircuit indices at which
    // each qubit is needed. At eviction time we look up the first index > i to find each
    // candidate's next use; we evict the one whose next use is furthest away (or never).
    let next_use: HashMap<Qubit, Vec<usize>> = if skip_redundant {
        let mut map: HashMap<Qubit, Vec<usize>> = HashMap::new();
        for (i, sub) in subcircuits.iter().enumerate() {
            for &q in &sub.qubits {
                map.entry(q).or_default().push(i);
            }
        }
        map
    } else {
        HashMap::new()
    };

    for (i, sub) in subcircuits.iter().enumerate() {
        if skip_redundant {
            // Only evict enough qubits to free slots for the ones we need to load.
            let new_count = sub
                .qubits
                .iter()
                .filter(|q| !in_flight.contains_key(q))
                .count();
            let must_evict = in_flight
                .len()
                .saturating_sub(max_subcircuit_size - new_count);
            if must_evict > 0 {
                let eviction_candidates = rank_evictable_qubits(&in_flight, &next_use, i, sub);
                for q in eviction_candidates.into_iter().take(must_evict) {
                    let proc = in_flight.remove(&q).unwrap();
                    ops.push(LoadStoreOp::Store(proc, ArchitectureQubit::Memory(q.0)));
                }
            }

            // Carried qubits hold specific processor slots; new qubits fill the gaps.
            let used_slots: HashSet<usize> = in_flight
                .values()
                .filter_map(|aq| {
                    if let ArchitectureQubit::Processor(s) = aq {
                        Some(*s)
                    } else {
                        None
                    }
                })
                .collect();
            let mut free_slots = (0usize..).filter(|s| !used_slots.contains(s));
            let mut to_load: Vec<_> = sub
                .qubits
                .iter()
                .filter(|q| !in_flight.contains_key(q))
                .collect();
            to_load.sort_unstable_by_key(|q| q.0);
            for q in to_load {
                let proc = ArchitectureQubit::Processor(free_slots.next().unwrap());
                ops.push(LoadStoreOp::Load(ArchitectureQubit::Memory(q.0), proc));
                in_flight.insert(*q, proc);
            }
        } else {
            let mut sorted_qubits: Vec<_> = sub.qubits.iter().collect();
            sorted_qubits.sort_unstable_by_key(|q| q.0);
            for (slot, q) in sorted_qubits.into_iter().enumerate() {
                let proc = ArchitectureQubit::Processor(slot);
                ops.push(LoadStoreOp::Load(ArchitectureQubit::Memory(q.0), proc));
                in_flight.insert(*q, proc);
            }
        }

        for gate in &sub.gates {
            ops.push(LoadStoreOp::Gate(shift_register_hashmap(gate, &in_flight)));
        }

        if !skip_redundant {
            let mut sorted: Vec<_> = in_flight.drain().collect();
            sorted.sort_unstable_by_key(|(q, _)| q.0);
            for (q, proc) in sorted {
                ops.push(LoadStoreOp::Store(proc, ArchitectureQubit::Memory(q.0)));
            }
        }
        // skip_redundant: leave in_flight populated so qubits stay on the processor.
    }
    LoadStoreCircuit { ops }
}

fn rank_evictable_qubits(
    in_flight: &HashMap<Qubit, ArchitectureQubit>,
    next_use: &HashMap<Qubit, Vec<usize>>,
    i: usize,
    sub: &Circuit,
) -> Vec<Qubit> {
    // next_use_after(q): the first subcircuit index strictly after i where q appears,
    // or usize::MAX if q is never used again.
    let next_use_after = |q: &Qubit| -> usize {
        next_use
            .get(q)
            .and_then(|uses| {
                let pos = uses.partition_point(|&u| u <= i);
                uses.get(pos).copied()
            })
            .unwrap_or(usize::MAX)
    };
    let mut evictable: Vec<Qubit> = in_flight
        .keys()
        .filter(|q| !sub.qubits.contains(q))
        .cloned()
        .collect();
    // Sort descending by next future use so we evict furthest-future first.
    // Break ties by qubit index for determinism.
    evictable.sort_unstable_by(|a, b| {
        next_use_after(b)
            .cmp(&next_use_after(a))
            .then(b.0.cmp(&a.0))
    });
    evictable
}

fn partition(
    circ: Circuit,
    max_subcircuit_size: usize,
    sat_mode: bool,
    sat_timeout: Option<u64>,
) -> Vec<Circuit> {
    if !sat_mode {
        return get_processor_subcircuits(circ, max_subcircuit_size);
    }
    let gate_count = circ.gates.len();
    // Run greedy first to learn max_k and a tight initial upper bound on load/stores.
    // The greedy Belady count avoids wasting the first SAT call on an unconstrained solve.
    let greedy = get_processor_subcircuits(circ.clone(), max_subcircuit_size);
    let max_k = (greedy.len() * 3 / 3).min(gate_count.max(1));

    let greedy_ls = to_load_store_circuit(&greedy, max_subcircuit_size, true);
    let initial_best_k = greedy_ls
        .ops
        .iter()
        .filter(|op| matches!(op, LoadStoreOp::Load(..) | LoadStoreOp::Store(..)))
        .count();
    // Fall back to greedy if the solver finds no feasible assignment within the timeout or proves infeasibility (which shouldn't happen since greedy is a valid solution).
    match crate::sat_partition::slice_and_optimize(
        &circ,
        max_subcircuit_size,
        max_k,
        3,
        Some(initial_best_k),
        sat_timeout,
    ) {
        Some(subcircuits) => subcircuits,
        None => {
            eprintln!("SAT solving failed, falling back on greedy solution...");

            greedy
        }
    }
}

pub fn compile(
    circ: Circuit,
    max_subcircuit_size: usize,
    simulate_corrections: bool,
    skip_redundant: bool,
    sat_mode: bool,
    sat_timeout: Option<u64>,
) -> PauliProductCircuit {
    let subcircuits = partition(circ, max_subcircuit_size, sat_mode, sat_timeout);
    let load_store = to_load_store_circuit(&subcircuits, max_subcircuit_size, skip_redundant);
    let pbc = to_pauli_product_circuit(&load_store, simulate_corrections);
    let resolved = if simulate_corrections {
        resolve_corrections(&pbc)
    } else {
        pbc
    };
    absorb_cliffords(&resolved)
}

/// Returns (load_store, pbc_pre_clifford, pbc_final) for inspecting all intermediate stages.
pub fn compile_steps(
    circ: Circuit,
    max_subcircuit_size: usize,
    simulate_corrections: bool,
    skip_redundant: bool,
    sat_mode: bool,
    sat_timeout: Option<u64>,
) -> (LoadStoreCircuit, PauliProductCircuit, PauliProductCircuit) {
    let subcircuits = partition(circ, max_subcircuit_size, sat_mode, sat_timeout);
    println!("Subcircuit count: {}", subcircuits.len());
    let load_store = to_load_store_circuit(&subcircuits, max_subcircuit_size, skip_redundant);
    let load_store_count = load_store.ops.iter().fold(0, |acc, x| match x {
        LoadStoreOp::Load(_, _) => acc + 1,
        LoadStoreOp::Store(_, _) => acc + 1,
        LoadStoreOp::Gate(_) => acc,
    });
    println!("Load store count: {load_store_count}");
    let pbc = to_pauli_product_circuit(&load_store, simulate_corrections);
    let resolved = if simulate_corrections {
        &resolve_corrections(&pbc)
    } else {
        &pbc
    };
    let clifford_free = absorb_cliffords(resolved);
    (load_store, pbc, clifford_free)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{Circuit, Gate, Qubit};
    use crate::pbc::Sign::{self, One};
    use crate::pbc::{
        MeasId, PPRAngle, Pauli, PauliAxis, PauliProductCircuit, PauliProductOperation, PauliString,
    };
    use crate::sat_partition::optimize_partition;
    use ArchitectureQubit::Processor;

    fn make_circuit(num_qubits: usize, gates: Vec<Gate<Qubit>>) -> Circuit {
        let mut c = Circuit::new(num_qubits);
        for g in gates {
            c.apply(g);
        }
        c
    }

    fn count_ls(ls: &LoadStoreCircuit) -> usize {
        ls.ops
            .iter()
            .filter(|op| matches!(op, LoadStoreOp::Load(..) | LoadStoreOp::Store(..)))
            .count()
    }

    // --- greedy partitioner tests ---

    #[test]
    fn test_greedy_capacity_respected() {
        // 3 independent H gates; proc_cap=2 forces at least 2 subcircuits
        let circ = make_circuit(
            3,
            vec![Gate::H(Qubit(0)), Gate::H(Qubit(1)), Gate::H(Qubit(2))],
        );
        let subs = get_processor_subcircuits(circ, 2);
        assert!(subs.len() >= 2, "expected at least 2 subcircuits");
        for sub in &subs {
            assert!(sub.qubits.len() <= 2, "subcircuit exceeds proc_cap=2");
        }
    }

    #[test]
    fn test_greedy_preserves_all_gates() {
        let circ = make_circuit(
            3,
            vec![Gate::H(Qubit(0)), Gate::H(Qubit(1)), Gate::H(Qubit(2))],
        );
        let subs = get_processor_subcircuits(circ, 2);
        let total_gates: usize = subs.iter().map(|s| s.gates.len()).sum();
        assert_eq!(total_gates, 3);
    }

    // --- to_load_store_circuit tests ---

    #[test]
    fn test_ls_no_skip_all_loads_stores() {
        // sub0={q0,q1}, sub1={q0}: without skip every boundary is full load+store
        let sub0 = make_circuit(2, vec![Gate::H(Qubit(0)), Gate::H(Qubit(1))]);
        let sub1 = make_circuit(2, vec![Gate::H(Qubit(0))]);
        let ls = to_load_store_circuit(&vec![sub0, sub1], 2, false);
        // sub0: 2 loads + 2 stores; sub1: 1 load + 1 store
        assert_eq!(count_ls(&ls), 6);
    }

    #[test]
    fn test_ls_skip_keeps_shared_qubits_in_flight() {
        // sub0={q0,q1}, sub1={q0,q2}, proc_cap=2
        // q0 stays; q1 is evicted (not needed in sub1); q2 is loaded → 3 loads + 1 store
        let sub0 = make_circuit(3, vec![Gate::H(Qubit(0)), Gate::H(Qubit(1))]);
        let sub1 = make_circuit(3, vec![Gate::H(Qubit(0)), Gate::H(Qubit(2))]);
        let ls = to_load_store_circuit(&vec![sub0, sub1], 2, true);
        assert_eq!(count_ls(&ls), 4);
    }

    #[test]
    fn test_ls_belady_evicts_furthest_future_qubit() {
        // sub0={q0,q1}, sub1={q2}, sub2={q0,q2}, proc_cap=2
        // At sub1: Belady evicts q1 (next use = never) over q0 (next use = sub2).
        // So q0 stays in flight into sub2, avoiding a reload → 3 loads + 1 store = 4 ops.
        // A naive eviction (remove q0) would require an extra load+store → 5+ ops.
        let sub0 = make_circuit(3, vec![Gate::H(Qubit(0)), Gate::H(Qubit(1))]);
        let sub1 = make_circuit(3, vec![Gate::H(Qubit(2))]);
        let sub2 = make_circuit(3, vec![Gate::H(Qubit(0)), Gate::H(Qubit(2))]);
        let ls = to_load_store_circuit(&vec![sub0, sub1, sub2], 2, true);
        assert_eq!(count_ls(&ls), 4);
    }

    // --- SAT partitioner tests ---

    fn four_qubit_circuit() -> Circuit {
        // H(q0), H(q1), H(q2), H(q3), CNOT(q0,q2), CNOT(q1,q3)
        // DAG: H(qi) -> CNOT that uses qi
        // Greedy with proc_cap=2 yields 4 subcircuits, 12 load/store ops (with Belady skip).
        // SAT optimum rearranges into {q0,q2} / {q0,q2} / {q1,q3} / {q1,q3}, giving 6 ops.
        make_circuit(
            4,
            vec![
                Gate::H(Qubit(0)),
                Gate::H(Qubit(1)),
                Gate::H(Qubit(2)),
                Gate::H(Qubit(3)),
                Gate::CNOT {
                    control: Qubit(0),
                    target: Qubit(2),
                },
                Gate::CNOT {
                    control: Qubit(1),
                    target: Qubit(3),
                },
            ],
        )
    }

    #[test]
    fn test_sat_respects_capacity() {
        let circ = four_qubit_circuit();
        let subs = optimize_partition(&circ, 2, 8, Some(12), None, &HashMap::new())
            .unwrap()
            .0;
        for sub in &subs {
            assert!(sub.qubits.len() <= 2, "SAT subcircuit exceeds proc_cap=2");
        }
    }

    #[test]
    fn test_sat_finds_better_partition() {
        let circ = four_qubit_circuit();
        let greedy_subs = get_processor_subcircuits(circ.clone(), 2);
        let max_k = greedy_subs.len() * 2;
        let greedy_count = count_ls(&to_load_store_circuit(&greedy_subs, 2, true));
        let sat_subs =
            optimize_partition(&circ, 2, max_k, Some(greedy_count), None, &HashMap::new())
                .unwrap()
                .0;
        let sat_count = count_ls(&to_load_store_circuit(&sat_subs, 2, true));
        assert!(
            sat_count < greedy_count,
            "SAT ({sat_count}) should beat greedy ({greedy_count})"
        );
        assert_eq!(sat_count, 6, "expected optimal 4 loads + 2 stores = 6");
    }

    fn single(sign: Sign, q: ArchitectureQubit, p: Pauli) -> PauliAxis {
        PauliAxis {
            sign,
            pauli_string: PauliString::new(vec![(q, p)]),
        }
    }

    fn rot(axis: PauliAxis, angle: PPRAngle) -> PauliProductOperation {
        PauliProductOperation::Rotation { axis, angle }
    }

    fn meas(axis: PauliAxis, id: u32) -> PauliProductOperation {
        PauliProductOperation::Measurement {
            axis,
            id: MeasId(id),
        }
    }

    fn absorbed_meas(instrs: Vec<PauliProductOperation>) -> PauliAxis {
        let mut circ = PauliProductCircuit::new();
        circ.instructions = instrs;
        let out = absorb_cliffords(&circ);
        assert_eq!(
            out.instructions.len(),
            1,
            "expected exactly one output instruction"
        );
        match out.instructions.into_iter().next().unwrap() {
            PauliProductOperation::Measurement { axis, .. } => axis,
            other => panic!("expected Measurement, got {:?}", other),
        }
    }

    #[test]
    fn x_pi4_z_pi4_then_z_meas() {
        let q = Processor(0);
        let result = absorbed_meas(vec![
            rot(single(One, q, Pauli::X), PPRAngle::PiOver4),
            rot(single(One, q, Pauli::Z), PPRAngle::PiOver4),
            meas(single(One, q, Pauli::Z), 0),
        ]);

        let expected_sign = One;
        let expected_pauli = Pauli::Y;
        assert_eq!(result.sign, expected_sign);
        assert_eq!(&*result.pauli_string, &[(q, expected_pauli)]);
    }
    #[test]
    fn z_pi4_x_pi4_then_z_meas() {
        let q = Processor(0);
        let result = absorbed_meas(vec![
            rot(single(One, q, Pauli::Z), PPRAngle::PiOver4),
            rot(single(One, q, Pauli::X), PPRAngle::PiOver4),
            meas(single(One, q, Pauli::Z), 0),
        ]);

        let expected_sign = One;
        let expected_pauli = Pauli::X;
        assert_eq!(result.sign, expected_sign);
        assert_eq!(&*result.pauli_string, &[(q, expected_pauli)]);
    }
}
