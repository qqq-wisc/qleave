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
pub struct PauliString(Vec<(ArchitectureQubit, Pauli)>);

impl PauliString {
    pub fn new(mut pairs: Vec<(ArchitectureQubit, Pauli)>) -> Self {
        pairs.sort_unstable_by_key(|&(q, _)| q);
        Self(pairs)
    }
}

impl std::ops::Deref for PauliString {
    type Target = [(ArchitectureQubit, Pauli)];
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

pub fn pauli_string_mult(a: &PauliString, b: &PauliString) -> PauliAxis {
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
            (Some(&(qa, _)), Some(&(qb, _))) => match qa.cmp(&qb) {
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

pub fn axes_commute(a: &PauliString, b: &PauliString) -> bool {
    // Two Pauli products commute iff an even number of qubit sites anti-commute.
    let mut anti = 0usize;
    let mut ai = a.iter().peekable();
    let mut bi = b.iter().peekable();
    loop {
        match (ai.peek(), bi.peek()) {
            (None, _) | (_, None) => break,
            (Some(&(qa, _)), Some(&(qb, _))) => match qa.cmp(&qb) {
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
    anti % 2 == 0
}
#[derive(Clone, Debug)]
pub struct PauliAxis {
    pub sign: Sign,
    pub pauli_string: PauliString,
}
#[derive(Clone, Debug)]
pub enum PauliProductOperation {
    Rotation { axis: PauliAxis, angle: PPRAngle },
    /// Classically-controlled Clifford correction: apply iff ALL bits in condition are 1.
    /// Only PiOver4 (Clifford) angles are meaningful here; PiOver2 corrections are software-only.
    ConditionalRotation { axis: PauliAxis, angle: PPRAngle, condition: AllOf },
    Measurement { axis: PauliAxis, id: MeasId },
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

impl fmt::Display for PauliString {
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

impl fmt::Display for PauliAxis {
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
            PauliProductOperation::ConditionalRotation { axis, angle, condition } => {
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
pub struct CliffordFrame {
    x: std::collections::HashMap<ArchitectureQubit, PauliAxis>,
    z: std::collections::HashMap<ArchitectureQubit, PauliAxis>,
}

impl CliffordFrame {
    pub fn new() -> Self {
        Self { x: std::collections::HashMap::new(), z: std::collections::HashMap::new() }
    }

    fn default_x(q: ArchitectureQubit) -> PauliAxis {
        PauliAxis { sign: Sign::One, pauli_string: PauliString::new(vec![(q, Pauli::X)]) }
    }

    fn default_z(q: ArchitectureQubit) -> PauliAxis {
        PauliAxis { sign: Sign::One, pauli_string: PauliString::new(vec![(q, Pauli::Z)]) }
    }

    /// Returns the image of basis element `p` on qubit `q` under the frame.
    pub fn image(&self, q: ArchitectureQubit, p: Pauli) -> PauliAxis {
        match p {
            Pauli::X => self.x.get(&q).cloned().unwrap_or_else(|| Self::default_x(q)),
            Pauli::Z => self.z.get(&q).cloned().unwrap_or_else(|| Self::default_z(q)),
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
            Pauli::I => PauliAxis { sign: Sign::One, pauli_string: PauliString::new(vec![]) },
        }
    }

    /// Conjugates `axis` by the frame: replaces each (q, p) factor with its frame image.
    pub fn apply(&self, axis: &PauliAxis) -> PauliAxis {
        let mut acc = PauliAxis { sign: axis.sign, pauli_string: PauliString::new(vec![]) };
        for &(q, p) in axis.pauli_string.iter() {
            let img = self.image(q, p);
            let prod = pauli_string_mult(&acc.pauli_string, &img.pauli_string);
            acc = PauliAxis { sign: acc.sign * img.sign * prod.sign, pauli_string: prod.pauli_string };
        }
        acc
    }

    /// Updates the frame by composing a PiOver4 rotation about `rotation` on the left.
    /// Also initializes frame entries for qubits in `rotation` whose default basis
    /// element anti-commutes with the rotation.
    pub fn update(&mut self, rotation: &PauliAxis) {
        let apply_rotation = |img: &PauliAxis, rot: &PauliAxis| -> Option<PauliAxis> {
            if axes_commute(&rot.pauli_string, &img.pauli_string) {
                None // no change
            } else {
                let prod = pauli_string_mult(&rot.pauli_string, &img.pauli_string);
                Some(PauliAxis {
                    sign: img.sign * rot.sign * prod.sign * Sign::J,
                    pauli_string: prod.pauli_string,
                })
            }
        };

        // Update existing frame entries. Iterate maps directly to avoid
        // a Vec allocation and the double-update bug that occurs when a
        // qubit is present in both maps (chaining keys would visit it twice).
        for xi in self.x.values_mut() {
            if let Some(new_xi) = apply_rotation(xi, rotation) {
                *xi = new_xi;
            }
        }
        for zi in self.z.values_mut() {
            if let Some(new_zi) = apply_rotation(zi, rotation) {
                *zi = new_zi;
            }
        }

        // Initialize entries for qubits in the rotation not yet in the frame.
        for &(q, _) in rotation.pauli_string.iter() {
            if !self.x.contains_key(&q) {
                let default = Self::default_x(q);
                if let Some(new_xi) = apply_rotation(&default, rotation) {
                    self.x.insert(q, new_xi);
                }
            }
            if !self.z.contains_key(&q) {
                let default = Self::default_z(q);
                if let Some(new_zi) = apply_rotation(&default, rotation) {
                    self.z.insert(q, new_zi);
                }
            }
        }
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
