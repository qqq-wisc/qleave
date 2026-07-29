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

impl Default for Sign {
    fn default() -> Self {
        Sign::One
    }
}

// `PartialEq`/`Hash` are structural but canonical: `new` sorts by qubit index,
// so equal supports compare equal (used as a memoization key in
// `circuit_to_checks`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PauliString<A>(Vec<(A, Pauli)>);

pub trait PauliStringIndex: Copy + std::cmp::Ord {}

/// Flat integer qubit index, used by the stim-lowered physical circuit (see
/// [`crate::checks_to_physical_circuit::PhysicalCircuit::flatten`]).
impl PauliStringIndex for usize {}

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

impl<A: PauliStringIndex> PauliString<A> {
    /// Relabel every qubit index by `f`, keeping the Pauli at each site. The
    /// result is re-sorted by [`PauliString::new`], so `f` need not be monotone.
    /// Mirrors [`crate::circuit::shift_register`] for gates.
    pub fn map_index<B: PauliStringIndex>(&self, mut f: impl FnMut(A) -> B) -> PauliString<B> {
        PauliString::new(self.0.iter().map(|&(q, p)| (f(q), p)).collect())
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

impl<A: PauliStringIndex> PauliAxis<A> {
    /// Relabel every qubit index by `f`, preserving the sign. See
    /// [`PauliString::map_index`].
    pub fn map_index<B: PauliStringIndex>(&self, f: impl FnMut(A) -> B) -> PauliAxis<B> {
        PauliAxis {
            sign: self.sign,
            pauli_string: self.pauli_string.map_index(f),
        }
    }
}

// ---------------------------------------------------------------------------
// Qubit index types
//
// The qubit identifiers a Pauli string can range over, in pipeline order: a
// physical qubit of one block, a code qubit of a (bridged) merged code, and a
// qubit of the surgery code built over a measurement graph. Each implements
// [`PauliStringIndex`] so it can key a [`PauliString`] / [`PauliAxis`]; the
// `…PauliString` / `…Pauli` aliases name the corresponding operator types.
// ---------------------------------------------------------------------------

/// A physical qubit of a single code block, identified by its within-block index.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PhysicalQubit(pub usize);
impl PauliStringIndex for PhysicalQubit {}

/// A Pauli string over the physical qubits of one code block.
pub type PhysicalPauliString = PauliString<PhysicalQubit>;

/// A code qubit of a (possibly bridged) merged code: a within-block physical qubit
/// `index` together with the `block` it belongs to. Carried as the surgery graph's
/// node weight on the port vertices — `Some(CodeQubit { .. })` on a port, `None`
/// on every other (ancilla / check) vertex. Recording the (block, index) pairing
/// at construction means bridging carries it verbatim, so downstream code never
/// has to undo the vertex-id offset a bridge introduces.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CodeQubit<K> {
    pub block: K,
    pub index: usize,
}
impl<K: Ord + Copy> PauliStringIndex for CodeQubit<K> {}

/// A Pauli operator over a single (possibly bridged) code's code qubits.
pub type CodePauli<K> = PauliAxis<CodeQubit<K>>;

/// A qubit of the merged surgery code: either an `EdgeQubit` (ancilla / edge
/// qubit, keyed by edge id) or an original `CodeQubit` of a code block.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum MergedCodeQubit<K> {
    EdgeQubit(usize),
    CodeQubit { block: K, index: usize },
}
impl<K: Ord + Copy> PauliStringIndex for MergedCodeQubit<K> {}

impl<K> From<CodeQubit<K>> for MergedCodeQubit<K> {
    fn from(q: CodeQubit<K>) -> Self {
        MergedCodeQubit::CodeQubit {
            block: q.block,
            index: q.index,
        }
    }
}

/// Lift a single code's Pauli operator into the merged surgery code by tagging
/// each `CodeQubit` as a `MergedCodeQubit::CodeQubit`.
pub fn lift_code_to_merged<K: Ord + Copy>(code_pauli: &CodePauli<K>) -> GraphPauli<K> {
    code_pauli.map_index(MergedCodeQubit::from)
}

/// A Pauli operator over the merged surgery code's qubits.
pub type GraphPauli<K> = PauliAxis<MergedCodeQubit<K>>;

/// A *logical* qubit of the target architecture, tagged by the region it lives in
/// (the memory / processor / magic-state blocks). The index type final compiled
/// Pauli-product operations range over.
#[derive(Clone, PartialEq, Eq, Hash, Copy, PartialOrd, Ord, Debug)]
pub enum ArchitectureQubit {
    Memory(usize),
    Processor(usize),
    Magic(usize),
}
impl PauliStringIndex for ArchitectureQubit {}
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PPRAngle {
    PiOver8,
    PiOver4,
    PiOver2,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
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

impl<K: fmt::Display> fmt::Display for MergedCodeQubit<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MergedCodeQubit::EdgeQubit(n) => write!(f, "Edge{n}"),
            MergedCodeQubit::CodeQubit { block, index } => write!(f, "{block}{index}"),
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
