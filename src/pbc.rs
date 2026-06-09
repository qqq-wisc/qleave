use std::fmt;

/// A stable identifier for a measurement outcome (classical bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeasId(pub u32);

/// A conjunction of measurement outcomes: satisfied when ALL listed bits are 1.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AllOf(pub Vec<MeasId>); // kept sorted

impl AllOf {
    pub fn single(id: MeasId) -> Self {
        Self(vec![id])
    }

    pub fn and(mut self, id: MeasId) -> Self {
        if let Err(pos) = self.0.binary_search(&id) {
            self.0.insert(pos, id);
        }
        self
    }
}

#[derive(Clone)]
pub struct PauliProductCircuit {
    pub instructions: Vec<PauliProductOperation>,
    pub(crate) next_meas_id: u32,
}

impl Default for PauliProductCircuit {
    fn default() -> Self {
        Self::new()
    }
}

impl PauliProductCircuit {
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            next_meas_id: 0,
        }
    }

    pub fn allocate_meas_id(&mut self) -> MeasId {
        let id = MeasId(self.next_meas_id);
        self.next_meas_id += 1;
        id
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Copy, PartialOrd, Ord, Debug)]
pub enum ArchitectureQubit {
    Memory(usize),
    Processor(usize),
    Magic(usize),
}

impl PauliStringIndex for ArchitectureQubit {}

#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub enum Sign {
    One,
    NegOne,
    J,
    NegJ,
}
impl std::ops::Mul for Sign {
    type Output = Sign;

    fn mul(self, rhs: Self) -> Self::Output {
        use Sign::*;
        match (self, rhs) {
            (One, x) | (x, One) => x,
            (NegOne, NegOne) => One,
            (J, NegJ) | (NegJ, J) => One,
            (J, J) => NegOne,
            (NegJ, NegJ) => NegOne,
            (NegOne, J) | (J, NegOne) => NegJ,
            (NegOne, NegJ) | (NegJ, NegOne) => J,
        }
    }
}
#[derive(Clone, Debug)]
pub struct PauliString<A>(Vec<(A, Pauli)>);

pub trait PauliStringIndex: Copy + std::cmp::Ord {}

impl<A: PauliStringIndex> PauliString<A> {
    pub fn new(mut pairs: Vec<(A, Pauli)>) -> Self {
        pairs.sort_unstable_by_key(|&(q, _)| q);
        Self(pairs)
    }
}

impl<A: PauliStringIndex> std::ops::Deref for PauliString<A> {
    type Target = [(A, Pauli)];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn pauli_mul(a: Pauli, b: Pauli) -> (Sign, Pauli) {
    match (a, b) {
        (Pauli::I, x) | (x, Pauli::I) => (Sign::One, x),
        (Pauli::X, Pauli::X) | (Pauli::Y, Pauli::Y) | (Pauli::Z, Pauli::Z) => (Sign::One, Pauli::I),
        (Pauli::X, Pauli::Y) => (Sign::J, Pauli::Z),
        (Pauli::Y, Pauli::X) => (Sign::NegJ, Pauli::Z),
        (Pauli::Y, Pauli::Z) => (Sign::J, Pauli::X),
        (Pauli::Z, Pauli::Y) => (Sign::NegJ, Pauli::X),
        (Pauli::Z, Pauli::X) => (Sign::J, Pauli::Y),
        (Pauli::X, Pauli::Z) => (Sign::NegJ, Pauli::Y),
    }
}

pub fn pauli_string_mult<A: PauliStringIndex>(a: &PauliString<A>, b: &PauliString<A>) -> PauliAxis<A> {
    let mut sign = Sign::One;
    let mut out = Vec::new();
    let mut ai = a.iter().peekable();
    let mut bi = b.iter().peekable();
    loop {
        match (ai.peek(), bi.peek()) {
            (None, None) => break,
            (Some(_), None) => {
                let &(q, p) = ai.next().unwrap();
                out.push((q, p));
            }
            (None, Some(_)) => {
                let &(q, p) = bi.next().unwrap();
                out.push((q, p));
            }
            (Some(&(qa, _)), Some(&(qb, _))) => match qa.cmp(qb) {
                std::cmp::Ordering::Less => {
                    let &(q, p) = ai.next().unwrap();
                    out.push((q, p));
                }
                std::cmp::Ordering::Greater => {
                    let &(q, p) = bi.next().unwrap();
                    out.push((q, p));
                }
                std::cmp::Ordering::Equal => {
                    let &(q, pa) = ai.next().unwrap();
                    let &(_, pb) = bi.next().unwrap();
                    let (s, p) = pauli_mul(pa, pb);
                    sign = sign * s;
                    if p != Pauli::I {
                        out.push((q, p));
                    }
                }
            },
        }
    }
    PauliAxis {
        sign,
        pauli_string: PauliString(out),
    }
}

pub fn axes_commute<A: PauliStringIndex>(a: &PauliString<A>, b: &PauliString<A>) -> bool {
    // Two Pauli products commute iff an even number of qubit sites anti-commute.
    let mut anti = 0usize;
    let mut ai = a.iter().peekable();
    let mut bi = b.iter().peekable();
    loop {
        match (ai.peek(), bi.peek()) {
            (None, _) | (_, None) => break,
            (Some(&(qa, _)), Some(&(qb, _))) => match qa.cmp(qb) {
                std::cmp::Ordering::Less => {
                    ai.next();
                }
                std::cmp::Ordering::Greater => {
                    bi.next();
                }
                std::cmp::Ordering::Equal => {
                    let &(_, pa) = ai.next().unwrap();
                    let &(_, pb) = bi.next().unwrap();
                    if !matches!(
                        (pa, pb),
                        (Pauli::I, _)
                            | (_, Pauli::I)
                            | (Pauli::X, Pauli::X)
                            | (Pauli::Y, Pauli::Y)
                            | (Pauli::Z, Pauli::Z)
                    ) {
                        anti += 1;
                    }
                }
            },
        }
    }
    anti.is_multiple_of(2)
}
#[derive(Clone, Debug)]
pub struct PauliAxis<A> {
    pub sign: Sign,
    pub pauli_string: PauliString<A>,
}
#[derive(Clone, Debug)]
pub enum PauliProductOperation {
    Rotation {
        axis: PauliAxis<ArchitectureQubit>,
        angle: PPRAngle,
    },
    /// Classically-controlled Clifford correction: apply iff ALL bits in condition are 1.
    /// Only PiOver4 (Clifford) angles are meaningful here; PiOver2 corrections are software-only.
    ConditionalRotation {
        axis: PauliAxis<ArchitectureQubit>,
        angle: PPRAngle,
        condition: AllOf,
    },
    Measurement {
        axis: PauliAxis<ArchitectureQubit>,
        id: MeasId,
    },
    /// Internal compiler marker: clear the Clifford frame for this qubit.
    /// Emitted before Load operations; consumed by apply_clifford_frame, never in final output.
    FrameReset(ArchitectureQubit),
}

#[derive(Clone, Copy, Debug)]
pub enum PPRAngle {
    PiOver8,
    PiOver4,
    PiOver2,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pauli {
    X,
    Y,
    Z,
    I,
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pauli::X => write!(f, "X"),
            Pauli::Y => write!(f, "Y"),
            Pauli::Z => write!(f, "Z"),
            Pauli::I => write!(f, "I"),
        }
    }
}

impl fmt::Display for Sign {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Sign::One => write!(f, "+"),
            Sign::NegOne => write!(f, "-"),
            Sign::J => write!(f, "+i"),
            Sign::NegJ => write!(f, "-i"),
        }
    }
}

impl fmt::Display for ArchitectureQubit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchitectureQubit::Memory(n) => write!(f, "Mem{n}"),
            ArchitectureQubit::Processor(n) => write!(f, "Proc{n}"),
            ArchitectureQubit::Magic(n) => write!(f, "Magic{n}"),
        }
    }
}

impl<A : fmt::Display> fmt::Display for PauliString<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (q, p)) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, "⊗")?;
            }
            write!(f, "{p}[{q}]")?;
        }
        Ok(())
    }
}

impl<A : fmt::Display> fmt::Display for PauliAxis<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.sign, self.pauli_string)
    }
}

impl fmt::Display for PPRAngle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PPRAngle::PiOver8 => write!(f, "π/8"),
            PPRAngle::PiOver4 => write!(f, "π/4"),
            PPRAngle::PiOver2 => write!(f, "π/2"),
        }
    }
}

impl fmt::Display for MeasId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "m{}", self.0)
    }
}

impl fmt::Display for AllOf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0.as_slice() {
            [] => write!(f, "never"),
            [id] => write!(f, "{id}"),
            ids => {
                write!(f, "(")?;
                for (i, id) in ids.iter().enumerate() {
                    if i > 0 {
                        write!(f, "∧")?;
                    }
                    write!(f, "{id}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for PauliProductOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PauliProductOperation::Rotation { axis, angle } => write!(f, "R({angle}, {axis})"),
            PauliProductOperation::ConditionalRotation {
                axis,
                angle,
                condition,
            } => {
                write!(f, "R({angle}, {axis}) if {condition}")
            }
            PauliProductOperation::Measurement { axis, id } => write!(f, "[{id}] Meas({axis})"),
            PauliProductOperation::FrameReset(q) => write!(f, "FrameReset({q})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Clifford frame
// ---------------------------------------------------------------------------

/// Tracks where X[q] and Z[q] map to under accumulated unconditional Clifford corrections.
/// Absent qubits are identity (X[q]→X[q], Z[q]→Z[q]).
#[derive(Debug)]
pub struct CliffordFrame {
    x: std::collections::HashMap<ArchitectureQubit, PauliAxis<ArchitectureQubit>>,
    z: std::collections::HashMap<ArchitectureQubit, PauliAxis<ArchitectureQubit>>,
}

impl Default for CliffordFrame {
    fn default() -> Self {
        Self::new()
    }
}

impl CliffordFrame {
    pub fn new() -> Self {
        Self {
            x: std::collections::HashMap::new(),
            z: std::collections::HashMap::new(),
        }
    }

    fn default_x(q: ArchitectureQubit) -> PauliAxis<ArchitectureQubit> {
        PauliAxis {
            sign: Sign::One,
            pauli_string: PauliString::new(vec![(q, Pauli::X)]),
        }
    }

    fn default_z(q: ArchitectureQubit) -> PauliAxis<ArchitectureQubit> {
        PauliAxis {
            sign: Sign::One,
            pauli_string: PauliString::new(vec![(q, Pauli::Z)]),
        }
    }

    /// Returns the image of basis element `p` on qubit `q` under the frame.
    pub fn image(&self, q: ArchitectureQubit, p: Pauli) -> PauliAxis<ArchitectureQubit> {
        match p {
            Pauli::X => self
                .x
                .get(&q)
                .cloned()
                .unwrap_or_else(|| Self::default_x(q)),
            Pauli::Z => self
                .z
                .get(&q)
                .cloned()
                .unwrap_or_else(|| Self::default_z(q)),
            Pauli::Y => {
                // Y = iXZ  →  frame(Y[q]) = i · frame(X[q]) · frame(Z[q])
                let xi = self.image(q, Pauli::X);
                let zi = self.image(q, Pauli::Z);
                let prod = pauli_string_mult(&xi.pauli_string, &zi.pauli_string);
                PauliAxis {
                    sign: xi.sign * zi.sign * prod.sign * Sign::J,
                    pauli_string: prod.pauli_string,
                }
            }
            Pauli::I => PauliAxis {
                sign: Sign::One,
                pauli_string: PauliString::new(vec![]),
            },
        }
    }

    /// Conjugates `axis` by the frame: replaces each (q, p) factor with its frame image.
    pub fn apply(&self, axis: &PauliAxis<ArchitectureQubit>) -> PauliAxis<ArchitectureQubit> {
        let mut acc = PauliAxis {
            sign: axis.sign,
            pauli_string: PauliString::new(vec![]),
        };
        for &(q, p) in axis.pauli_string.iter() {
            let img = self.image(q, p);
            let prod = pauli_string_mult(&acc.pauli_string, &img.pauli_string);
            acc = PauliAxis {
                sign: acc.sign * img.sign * prod.sign,
                pauli_string: prod.pauli_string,
            };
        }
        acc
    }

    /// Updates the frame by composing a PiOver4 rotation about `rotation` on the left.
    /// Also initializes frame entries for qubits in `rotation` whose default basis
    /// element anti-commutes with the rotation.
    pub fn update(&mut self, rotation: &PauliAxis<ArchitectureQubit>) {
        let rot_qubits: Vec<_> = rotation.pauli_string.iter().map(|&(q, _)| q).collect();

        let x_qubits: Vec<_> = self
            .x
            .keys()
            .cloned()
            .chain(rot_qubits.iter().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let z_qubits: Vec<_> = self
            .z
            .keys()
            .cloned()
            .chain(rot_qubits.into_iter())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let new_x: Vec<_> = x_qubits
            .into_iter()
            .filter_map(|q| {
                let row = PauliAxis {
                    sign: Sign::One,
                    pauli_string: PauliString::new(vec![(q, Pauli::X)]),
                };
                if axes_commute(&row.pauli_string, &rotation.pauli_string) {
                    return None;
                }
                let row_image = self.apply(&row); // should be self.apply(&row)
                let rotation_image = self.apply(&rotation);
                let prod = pauli_string_mult(&rotation_image.pauli_string, &row_image.pauli_string);
                let sign = Sign::J * prod.sign * row_image.sign * rotation_image.sign;
                // let new_axis = self.apply(&PauliAxis { sign, pauli_string: prod.pauli_string });
                let new_axis = PauliAxis { sign, pauli_string: prod.pauli_string };
                Some((q, new_axis))
            })
            .collect();

        let new_z: Vec<_> = z_qubits
            .into_iter()
            .filter_map(|q| {
                let row = PauliAxis {
                    sign: Sign::One,
                    pauli_string: PauliString::new(vec![(q, Pauli::Z)]),
                };
                if axes_commute(&row.pauli_string, &rotation.pauli_string) {
                    return None;
                }
                let row_image = self.apply(&row);
                let rotation_image = self.apply(&rotation);
                let prod = pauli_string_mult(&rotation_image.pauli_string, &row_image.pauli_string);
                let sign = Sign::J * prod.sign * row_image.sign * rotation_image.sign;
                // let new_axis = self.apply(&PauliAxis { sign, pauli_string: prod.pauli_string });
                let new_axis = PauliAxis { sign, pauli_string: prod.pauli_string };
                Some((q, new_axis))
            })
            .collect();

        self.x.extend(new_x);
        self.z.extend(new_z);
    }

    /// Clears the frame entries for `q` (called when `q` is loaded from memory).
    pub fn reset(&mut self, q: ArchitectureQubit) {
        self.x.remove(&q);
        self.z.remove(&q);
    }
}

impl fmt::Display for PauliProductCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, instr) in self.instructions.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{instr}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for PauliProductCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
