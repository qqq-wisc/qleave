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

/// Pre-computed truth table of measurement axes indexed by measurement outcome patterns.
///
/// `conditions` lists all distinct MeasIds that influence this measurement.
/// `axes[mask]` is the physical gadget (axis) to use when bit `i` of `mask` equals whether
/// `conditions[i]` is 1.  An unconditional measurement has `conditions = []`, `axes = [base]`.
#[derive(Clone, Debug)]
pub struct ConditionedAxis {
    pub conditions: Vec<MeasId>, // sorted; one per bit dimension of the truth table
    pub axes: Vec<PauliAxis>,    // len == 1 << conditions.len()
}

impl ConditionedAxis {
    pub fn unconditional(base: PauliAxis) -> Self {
        Self {
            conditions: vec![],
            axes: vec![base],
        }
    }
}

/// Build the truth table for a measurement axis given a base and a list of conditional
/// Clifford corrections.  Each correction `(AllOf(bits), factor)` is applied (multiplied
/// into the axis) for every truth-table row where ALL bits in `bits` are set.
pub fn build_conditioned_axis(base: PauliAxis, corrections: &[(AllOf, PauliAxis)]) -> ConditionedAxis {
    if corrections.is_empty() {
        return ConditionedAxis::unconditional(base);
    }

    // Collect all distinct MeasIds referenced across all conditions.
    let mut all_ids: Vec<MeasId> = corrections
        .iter()
        .flat_map(|(allof, _)| allof.0.iter().copied())
        .collect();
    all_ids.sort_unstable();
    all_ids.dedup();

    let k = all_ids.len();
    let n = 1usize << k;

    // Initialize all rows with the base axis.
    let mut axes: Vec<PauliAxis> = (0..n).map(|_| base.clone()).collect();

    // For each row (bit assignment), apply corrections whose conditions are fully satisfied.
    for mask in 0..n {
        for (allof, factor) in corrections {
            let all_set = allof.0.iter().all(|id| {
                let bit = all_ids.binary_search(id).unwrap();
                (mask >> bit) & 1 == 1
            });
            if all_set {
                let ax = &axes[mask];
                let product = pauli_string_mult(&factor.pauli_string, &ax.pauli_string);
                axes[mask] = PauliAxis {
                    sign: factor.sign * ax.sign * product.sign * Sign::J,
                    pauli_string: product.pauli_string,
                };
            }
        }
    }

    ConditionedAxis {
        conditions: all_ids,
        axes,
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
    Measurement { axis: ConditionedAxis, id: MeasId },
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

impl fmt::Display for ConditionedAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.conditions.is_empty() {
            // Unconditional: just show the axis.
            write!(f, "{}", self.axes[0])
        } else {
            // Show the truth table: "base | mask→axis ..."
            write!(f, "{}", self.axes[0])?;
            let k = self.conditions.len();
            for mask in 1..(1usize << k) {
                // Describe which condition bits are set.
                write!(f, " | ")?;
                let mut first = true;
                for i in 0..k {
                    if (mask >> i) & 1 == 1 {
                        if !first {
                            write!(f, "∧")?;
                        }
                        write!(f, "{}", self.conditions[i])?;
                        first = false;
                    }
                }
                write!(f, "→{}", self.axes[mask])?;
            }
            Ok(())
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
        }
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
