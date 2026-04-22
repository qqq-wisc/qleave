use std::{
    collections::{HashMap, HashSet},
    fmt::{self, Debug},
    hash::Hash,
};

#[derive(Debug)]
pub struct Circuit {
    pub gates: Vec<Gate<Qubit>>,
    pub qubits: HashSet<Qubit>,
    pub num_qubits: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GateId(pub usize);

pub struct Dag {
    // n = number of gates
    pub n: usize,
    pub successors: Vec<Vec<GateId>>,
    // in_degree[i] = number of predecessors of gate i
    pub in_degree: Vec<usize>,
}

impl Circuit {
    pub fn new(num_qubits: usize) -> Self {
        Circuit {
            gates: Vec::new(),
            qubits: HashSet::new(),
            num_qubits,
        }
    }

    pub fn apply(&mut self, gate: Gate<Qubit>) {
        for q in gate.qubits() {
            self.qubits.insert(q);
        }
        self.gates.push(gate);
    }

    pub fn to_dag(&self) -> Dag {
        let n = self.gates.len();
        let mut successors: Vec<Vec<GateId>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];
        let mut last_on_qubit: HashMap<Qubit, GateId> = HashMap::new();
        let mut seen_preds: Vec<HashSet<GateId>> = vec![HashSet::new(); n];

        for (i, gate) in self.gates.iter().enumerate() {
            let id = GateId(i);
            for q in gate.qubits() {
                if let Some(pred) = last_on_qubit.get(&q).copied() {
                    if seen_preds[i].insert(pred) {
                        successors[pred.0].push(id);
                        in_degree[i] += 1;
                    }
                }
                last_on_qubit.insert(q, id);
            }
        }

        Dag {
            n,
            successors,
            in_degree,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Qubit(pub usize);

#[derive(Clone, Debug)]
pub enum Gate<A: Copy> {
    H(A),
    CNOT { control: A, target: A },
    X(A),
    Y(A),
    Z(A),
    CCZ { control1: A, control2: A, target: A },
    S(A),
    Sdg(A),
    T(A),
    Tdg(A),
}

impl<A: Copy> Gate<A> {
    pub fn qubits(&self) -> Vec<A> {
        match self {
            Gate::H(q)
            | Gate::X(q)
            | Gate::Y(q)
            | Gate::Z(q)
            | Gate::T(q)
            | Gate::S(q)
            | Gate::Sdg(q) => vec![*q],
            Gate::CNOT { control, target } => vec![*control, *target],
            Gate::CCZ {
                control1,
                control2,
                target,
            } => vec![*control1, *control2, *target],
            Gate::Tdg(q) => vec![*q],
        }
    }
}

impl<A: Copy + fmt::Display> fmt::Display for Gate<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gate::H(q) => write!(f, "H({q})"),
            Gate::X(q) => write!(f, "X({q})"),
            Gate::Y(q) => write!(f, "Y({q})"),
            Gate::Z(q) => write!(f, "Z({q})"),
            Gate::S(q) => write!(f, "S({q})"),
            Gate::Sdg(q) => write!(f, "Sdg({q})"),
            Gate::T(q) => write!(f, "T({q})"),
            Gate::Tdg(q) => write!(f, "Tdg({q})"),
            Gate::CNOT { control, target } => write!(f, "CNOT({control}, {target})"),
            Gate::CCZ { control1, control2, target } => {
                write!(f, "CCZ({control1}, {control2}, {target})")
            }
        }
    }
}

fn shift_register<T: Copy, R: Copy>(gate: &Gate<T>, f: impl Fn(T) -> R) -> Gate<R> {
    match gate {
        Gate::H(q) => Gate::H(f(*q)),
        Gate::X(q) => Gate::X(f(*q)),
        Gate::Y(q) => Gate::Y(f(*q)),
        Gate::Z(q) => Gate::Z(f(*q)),
        Gate::T(q) => Gate::T(f(*q)),
        Gate::S(q) => Gate::S(f(*q)),
        Gate::Sdg(q) => Gate::Sdg(f(*q)),
        Gate::CNOT { control, target } => Gate::CNOT {
            control: f(*control),
            target: f(*target),
        },
        Gate::CCZ {
            control1,
            control2,
            target,
        } => Gate::CCZ {
            control1: f(*control1),
            control2: f(*control2),
            target: f(*target),
        },
        Gate::Tdg(q) => Gate::Tdg(f(*q)),
    }
}

pub fn shift_register_hashmap<T: Copy + Hash + Eq + Debug, R: Copy>(
    gate: &Gate<T>,
    m: &HashMap<T, R>,
) -> Gate<R> {
    shift_register(gate, |q| {
        m.get(&q)
            .copied()
            .unwrap_or_else(|| panic!("Qubit not found in mapping: {:?}", q))
    })
}
