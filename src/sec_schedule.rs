//! Left–right / staggered syndrome-extraction scheduling (arXiv:2603.05481,
//! Strikis–Browne–Beverland).
//!
//! Replaces the legacy check-colored schedule of
//! [`crate::checks_to_physical_circuit`] — which serializes *any* two checks
//! sharing a data qubit — with schedules that only serialize where correctness
//! demands it.
//!
//! # Correctness theory
//!
//! The measurement gadget is `RX a; C_{P_i}(a, q_i)…; MX a` (ancilla = control).
//! Swapping two controlled-Paulis on the same data qubit with *anticommuting*
//! site Paulis creates a `CZ` between the two ancillas; commuting site Paulis
//! swap freely. Two commuting checks anticommute sitewise on an **even** number
//! of shared qubits, so the spurious `CZ` hooks cancel iff, for every such pair
//! (A, B), the time order is **consistent**: A's gate precedes B's at *all*
//! shared anticommuting sites, or follows at all of them (an even number of
//! crossings ⇒ `CZ² = I`). A mixed one-each ordering leaves an odd `CZ` count
//! and corrupts both outcomes.
//!
//! Define the **conflict graph** on a round's checks: an edge iff two checks
//! share at least one qubit with anticommuting site Paulis. Checks in one
//! conflict-free class commute sitewise everywhere, so their gates interleave
//! freely; only the physical one-gate-per-qubit-per-TICK constraint applies,
//! which a bipartite **edge coloring** of the (check × qubit) gate graph
//! satisfies with Δ colors (König). Same-type CSS checks sharing qubits are
//! *not* conflicting — the key relaxation over the legacy schedule.
//!
//! Two schedules are emitted:
//!
//! * **Staggered left–right** (conflict graph 2-colorable, ≥ 2 rounds): data
//!   qubits are split into left/right sets; round `n` runs phase 1 =
//!   `[L-gates of G1(n) ∥ R-gates of G2(n−1)]`, measures `G2(n−1)`, resets
//!   `G2(n)`, runs phase 2 = `[R-gates of G1(n) ∥ L-gates of G2(n)]`, and
//!   measures `G1(n)`. Every conflicting pair is consistently ordered for *any*
//!   partition and any Pauli content: relative to `G1(n)`, the `G2(n−1)`
//!   instance is entirely earlier (left gates in round n−1's phase 2, right
//!   gates in round n's phase 1) and the `G2(n)` instance entirely later.
//!   Amortized depth ≈ `max(|C(L₁)|,|C(R₂)|) + max(|C(L₂)|,|C(R₁)|)` + O(1)
//!   per round.
//!
//! * **Sequential conflict phases** (fallback for χ > 2 — possible in
//!   mixed-operator surgery rounds — or single-round segments): the conflict
//!   classes run one after another, each edge-colored internally. Depth =
//!   Σ_c Δ(class c) + 2 per round; record order is preserved exactly.
//!
//! # Record bookkeeping
//!
//! Downstream `rec[-k]` offsets (round detectors, split `XCorrection`s, the
//! frame-solver observables and detectors) assume one record per `MPP` *in MPP
//! order*. The staggered schedule permutes records — each steady round emits
//! `[G2 checks of round n−1] then [G1 checks of round n]` — so the expansion
//! maintains an old→new record permutation and rewrites every rec-consumer
//! outside `REPEAT` bodies. Round detectors inside bodies are re-emitted
//! structurally with iteration-invariant offsets (see [`StaggeredSeg::emit`]).

use std::collections::BTreeMap;

use crate::checks_to_physical_circuit::{GateSink, PhysicalGate};
use crate::pbc::{Pauli, PauliAxis, Sign};

/// Which syndrome-extraction schedule
/// [`crate::checks_to_physical_circuit::PhysicalCircuit::expand_mpp_to_sec`]
/// lowers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SecSchedule {
    /// Staggered left–right circuits (this module).
    Lrc,
    /// The original check-colored, lockstep schedule (kept for A/B comparison).
    Legacy,
}

type Gate = PhysicalGate<usize>;
/// One controlled-Pauli of the gadget: (check index within the round, data
/// qubit, site Pauli).
type Site = (usize, usize, Pauli);

/// The non-identity sites of a check.
fn sites(check: &PauliAxis<usize>) -> impl Iterator<Item = (usize, Pauli)> + '_ {
    check.pauli_string.iter().copied().filter(|&(_, p)| p != Pauli::I)
}

/// Conflict-class assignment for a round's checks: two checks conflict iff they
/// share a qubit with anticommuting (distinct, non-identity) site Paulis. Tries
/// a BFS 2-coloring of the conflict graph first (the CSS case is bipartite:
/// X checks vs Z checks); on an odd cycle falls back to greedy coloring.
fn conflict_classes(checks: &[&PauliAxis<usize>]) -> Vec<usize> {
    let m = checks.len();
    // Adjacency via per-qubit incidence lists (LDPC ⇒ small qubit degree).
    let mut by_qubit: BTreeMap<usize, Vec<(usize, Pauli)>> = BTreeMap::new();
    for (j, check) in checks.iter().enumerate() {
        for (q, p) in sites(check) {
            by_qubit.entry(q).or_default().push((j, p));
        }
    }
    let mut adj: Vec<std::collections::BTreeSet<usize>> = vec![Default::default(); m];
    for incidences in by_qubit.values() {
        for (i, &(ja, pa)) in incidences.iter().enumerate() {
            for &(jb, pb) in &incidences[i + 1..] {
                if pa != pb && ja != jb {
                    adj[ja].insert(jb);
                    adj[jb].insert(ja);
                }
            }
        }
    }

    // BFS 2-coloring attempt.
    let mut color = vec![usize::MAX; m];
    let mut bipartite = true;
    'outer: for start in 0..m {
        if color[start] != usize::MAX {
            continue;
        }
        color[start] = 0;
        let mut queue = std::collections::VecDeque::from([start]);
        while let Some(j) = queue.pop_front() {
            for &k in &adj[j] {
                if color[k] == usize::MAX {
                    color[k] = 1 - color[j];
                    queue.push_back(k);
                } else if color[k] == color[j] {
                    bipartite = false;
                    break 'outer;
                }
            }
        }
    }
    if bipartite {
        return color;
    }

    // Greedy fallback: smallest color unused by already-colored neighbors.
    let mut color = vec![usize::MAX; m];
    for j in 0..m {
        let used: std::collections::BTreeSet<usize> =
            adj[j].iter().map(|&k| color[k]).filter(|&c| c != usize::MAX).collect();
        color[j] = (0..).find(|c| !used.contains(c)).unwrap();
    }
    color
}

/// Proper edge coloring of a bipartite graph with exactly Δ colors (König).
/// Vertices are dense ids `0..n_vertices` with the two sides disjoint (here:
/// checks and qubits); `edges` are (u, v) pairs. For each new edge, take the
/// smallest color `α` free at `u`; if occupied at `v`, flip the maximal
/// α/β-alternating path starting at `v` (β = smallest free at `v`). In a
/// bipartite graph that path cannot reach `u` — it would have to arrive on an
/// α edge (odd distance from `v`), but α is free at `u` — so after the flip α
/// is free at both endpoints.
fn bipartite_edge_coloring(n_vertices: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut at: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); n_vertices];
    let mut color = vec![usize::MAX; edges.len()];
    let free = |at: &[BTreeMap<usize, usize>], x: usize| -> usize {
        (0..).find(|c| !at[x].contains_key(c)).unwrap()
    };
    for (e, &(u, v)) in edges.iter().enumerate() {
        let alpha = free(&at, u);
        if at[v].contains_key(&alpha) {
            let beta = free(&at, v);
            // Collect the maximal alternating path from `v`: α, β, α, …
            let (mut cur, mut c, mut path) = (v, alpha, Vec::new());
            while let Some(&eidx) = at[cur].get(&c) {
                path.push(eidx);
                let (x, y) = edges[eidx];
                cur = if x == cur { y } else { x };
                c = if c == alpha { beta } else { alpha };
            }
            // Flip: detach the path, then re-attach with swapped colors.
            for &eidx in &path {
                let (x, y) = edges[eidx];
                at[x].remove(&color[eidx]);
                at[y].remove(&color[eidx]);
            }
            for &eidx in &path {
                let swapped = if color[eidx] == alpha { beta } else { alpha };
                color[eidx] = swapped;
                let (x, y) = edges[eidx];
                at[x].insert(swapped, eidx);
                at[y].insert(swapped, eidx);
            }
        }
        color[e] = alpha;
        at[u].insert(alpha, e);
        at[v].insert(alpha, e);
    }
    color
}

/// Greedy left/right data-qubit partition for the staggered schedule (classes
/// `0` = G1, `1` = G2): qubits in descending total degree, each assigned the
/// side minimizing the resulting depth proxy
/// `max(Δ(L₁), Δ(R₂)) + max(Δ(L₂), Δ(R₁))`, where Δ of each phase subgraph is
/// the larger of its max qubit degree and max per-check side weight. Ties go
/// left; deterministic.
fn partition_left(checks: &[&PauliAxis<usize>], classes: &[usize]) -> std::collections::BTreeSet<usize> {
    // Per-qubit class degrees.
    let mut deg: BTreeMap<usize, [usize; 2]> = BTreeMap::new();
    for (j, check) in checks.iter().enumerate() {
        for (q, _) in sites(check) {
            deg.entry(q).or_default()[classes[j]] += 1;
        }
    }
    let mut order: Vec<usize> = deg.keys().copied().collect();
    order.sort_by_key(|q| (std::cmp::Reverse(deg[q][0] + deg[q][1]), *q));

    // side_weight[j][side]: sites of check j assigned to that side so far.
    let mut side_weight = vec![[0usize; 2]; checks.len()];
    let mut qubit_side: BTreeMap<usize, usize> = BTreeMap::new(); // 0 = left, 1 = right
    // Max qubit degree per (side, class) among assigned qubits.
    let mut max_qdeg = [[0usize; 2]; 2]; // [side][class]
    // Running max side-weight over already-committed qubits, per (side, class).
    // Maintained incrementally: assigning a qubit only bumps the checks incident
    // to it, so there is no need to rescan all `m` checks per candidate.
    let mut max_check = [[0usize; 2]; 2]; // [side][class]

    let mut incident: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (j, check) in checks.iter().enumerate() {
        for (q, _) in sites(check) {
            incident.entry(q).or_default().push(j);
        }
    }

    for &q in &order {
        let incident_q = &incident[&q];
        let mut best = (usize::MAX, 0usize);
        for side in [0usize, 1] {
            // Δ per phase subgraph after assigning q to `side`. Only q's incident
            // checks change, and only on `side`; every other (s, c) max is the
            // committed running max unchanged.
            let mut max_q = max_qdeg;
            for class in 0..2 {
                max_q[side][class] = max_q[side][class].max(deg[&q][class]);
            }
            let mut mc = max_check;
            for &j in incident_q {
                let c = classes[j];
                mc[side][c] = mc[side][c].max(side_weight[j][side] + 1);
            }
            // Phase 1 = (left, G1) ∥ (right, G2); phase 2 = (left, G2) ∥ (right, G1).
            let delta = |s: usize, c: usize| max_q[s][c].max(mc[s][c]);
            let cost = delta(0, 0).max(delta(1, 1)) + delta(0, 1).max(delta(1, 0));
            if cost < best.0 {
                best = (cost, side);
            }
        }
        let side = best.1;
        qubit_side.insert(q, side);
        for class in 0..2 {
            max_qdeg[side][class] = max_qdeg[side][class].max(deg[&q][class]);
        }
        for &j in incident_q {
            side_weight[j][side] += 1;
            let c = classes[j];
            max_check[side][c] = max_check[side][c].max(side_weight[j][side]);
        }
    }
    qubit_side.iter().filter(|&(_, &s)| s == 0).map(|(&q, _)| q).collect()
}

/// A staggered left–right segment: one check set measured for `r` consecutive
/// rounds (a reference round plus a `REPEAT` of identical rounds).
struct StaggeredSeg<'a> {
    checks: &'a [&'a PauliAxis<usize>],
    base: usize,
    m: usize,
    /// Class sizes: `a = |G1|`, `b = |G2|`.
    a: usize,
    b: usize,
    class_of: Vec<usize>,
    /// Rank of each check within its class (ascending check index).
    subpos: Vec<usize>,
    g1: Vec<usize>,
    g2: Vec<usize>,
    /// Controlled-Pauli buckets by edge color, per phase subgraph: `l1`/`r2`
    /// run in phase 1, `l2`/`r1` in phase 2.
    l1: Vec<Vec<Site>>,
    r2: Vec<Vec<Site>>,
    l2: Vec<Vec<Site>>,
    r1: Vec<Vec<Site>>,
}

impl<'a> StaggeredSeg<'a> {
    fn new(checks: &'a [&'a PauliAxis<usize>], classes: Vec<usize>, base: usize) -> Self {
        let m = checks.len();
        let left = partition_left(checks, &classes);
        let mut subpos = vec![0usize; m];
        let (mut g1, mut g2) = (Vec::new(), Vec::new());
        for j in 0..m {
            let group = if classes[j] == 0 { &mut g1 } else { &mut g2 };
            subpos[j] = group.len();
            group.push(j);
        }

        // Split the gate list into the four phase subgraphs and edge-color each.
        let mut parts: [Vec<Site>; 4] = Default::default(); // l1, r2, l2, r1
        for (j, check) in checks.iter().enumerate() {
            for (q, p) in sites(check) {
                let idx = match (left.contains(&q), classes[j]) {
                    (true, 0) => 0,  // l1
                    (false, 1) => 1, // r2
                    (true, 1) => 2,  // l2
                    (false, 0) => 3, // r1
                    _ => unreachable!("staggered segments have exactly two conflict classes"),
                };
                parts[idx].push((j, q, p));
            }
        }
        let color_part = |part: &Vec<Site>| -> Vec<Vec<Site>> {
            // Dense vertex ids: checks 0..m, then qubits.
            let mut qid: BTreeMap<usize, usize> = BTreeMap::new();
            for &(_, q, _) in part {
                let next = m + qid.len();
                qid.entry(q).or_insert(next);
            }
            let edges: Vec<(usize, usize)> = part.iter().map(|&(j, q, _)| (j, qid[&q])).collect();
            let colors = bipartite_edge_coloring(m + qid.len(), &edges);
            let n_colors = colors.iter().copied().max().map_or(0, |c| c + 1);
            let mut buckets = vec![Vec::new(); n_colors];
            for (i, &site) in part.iter().enumerate() {
                buckets[colors[i]].push(site);
            }
            buckets
        };
        let [l1, r2, l2, r1] = [
            color_part(&parts[0]),
            color_part(&parts[1]),
            color_part(&parts[2]),
            color_part(&parts[3]),
        ];
        StaggeredSeg {
            checks,
            base,
            m,
            a: g1.len(),
            b: g2.len(),
            class_of: classes,
            subpos,
            g1,
            g2,
            l1,
            r2,
            l2,
            r1,
        }
    }

    fn cnots<'b>(&self, bucket: &'b [Site]) -> impl Iterator<Item = Gate> + 'b {
        let base = self.base;
        bucket.iter().map(move |&(j, q, p)| PhysicalGate::Controlled(p, base + j, q))
    }

    fn measure(&self, group: &[usize]) -> Vec<Gate> {
        group
            .iter()
            .map(|&j| {
                debug_assert!(
                    matches!(self.checks[j].sign, Sign::One | Sign::NegOne),
                    "non-Hermitian sign on a syndrome MPP"
                );
                PhysicalGate::Measure(Pauli::X, self.base + j, self.checks[j].sign == Sign::NegOne)
            })
            .collect()
    }

    /// The TICK layers of one round. `prologue` = round 1: no previous-round G2
    /// instance exists, so phase 1 omits `r2` and phase 2 omits the G2 measure.
    ///
    /// Steady layout: `[RX G1]`, phase-1 layers `[l1ᵏ ∥ r2ᵏ]`, then phase 2 with
    /// the G2 handover folded into its first two layers — `[MX G2(n−1) ∥ r1⁰]`,
    /// `[RX G2(n) ∥ r1¹]`, `[l2ᵏ ∥ r1ᵏ⁺²]`… — and finally `[MX G1(n)]`. The
    /// folds are qubit-disjoint (G2 ancillas vs right data + G1 ancillas), and
    /// G2(n)'s left CNOTs start only after its reset.
    fn round_layers(&self, prologue: bool) -> Vec<Vec<Gate>> {
        let mut layers: Vec<Vec<Gate>> = Vec::new();
        layers.push(self.g1.iter().map(|&j| PhysicalGate::Reset(Pauli::X, self.base + j)).collect());
        let t1 = if prologue { self.l1.len() } else { self.l1.len().max(self.r2.len()) };
        for k in 0..t1 {
            let mut layer = Vec::new();
            if let Some(bucket) = self.l1.get(k) {
                layer.extend(self.cnots(bucket));
            }
            if !prologue {
                if let Some(bucket) = self.r2.get(k) {
                    layer.extend(self.cnots(bucket));
                }
            }
            layers.push(layer);
        }
        let t2 = (self.l2.len() + 2).max(self.r1.len()).max(2);
        for k in 0..t2 {
            let mut layer = Vec::new();
            if let Some(bucket) = self.r1.get(k) {
                layer.extend(self.cnots(bucket));
            }
            if k == 0 && !prologue {
                layer.extend(self.measure(&self.g2));
            }
            if k == 1 {
                layer.extend(
                    self.g2.iter().map(|&j| PhysicalGate::Reset(Pauli::X, self.base + j)),
                );
            }
            if k >= 2 {
                if let Some(bucket) = self.l2.get(k - 2) {
                    layer.extend(self.cnots(bucket));
                }
            }
            layers.push(layer);
        }
        layers.push(self.measure(&self.g1));
        layers
    }

    /// The trailing flush: the last G2 instance still owes its right-side CNOTs
    /// (they would run in the never-emitted next round's phase 1) and its
    /// measurement.
    fn epilogue_layers(&self) -> Vec<Vec<Gate>> {
        let mut layers: Vec<Vec<Gate>> =
            self.r2.iter().map(|bucket| self.cnots(bucket).collect()).collect();
        layers.push(self.measure(&self.g2));
        layers
    }

    /// Emit the whole `r`-round segment (prologue, peeled round 2, a
    /// `REPEAT(r−2)` steady body, epilogue) plus round detectors, and extend the
    /// old→new record permutation.
    ///
    /// Record layout (`start` = records before the segment; check j has class
    /// subposition `p`/`q`): round-n G1 records land at `start + p` (n = 1) or
    /// `start + a + (n−2)m + b + p`; round-n G2 records land at
    /// `start + a + (n−1)m + q` — one round late, in the next round's phase-2
    /// handover. Totals per emission unit: prologue `a`, each round `m`,
    /// epilogue `b`; segment total `rm`, matching the MPP stream.
    ///
    /// Detector offsets are iteration-invariant, including the first steady
    /// iteration against the prologue, exactly because the prologue omits the
    /// `b` G2 records: a G1 detector reads `rec[-(a−p)]` / `rec[-(m+a−p)]`, a
    /// G2 detector (comparing rounds n−1 and n−2, one round late) reads
    /// `rec[-(m−q)]` / `rec[-(2m−q)]`, and the epilogue closes G2's last
    /// comparison with `rec[-(b−q)]` / `rec[-(m+b−q)]`. Each check gets r−1
    /// detectors, as in the MPP stream.
    fn emit<S: GateSink>(&self, r: usize, out: &mut S, map: &mut Vec<usize>) {
        let (m, a, b) = (self.m, self.a, self.b);
        let g1_detectors = || -> Vec<Gate> {
            (0..a).map(|p| PhysicalGate::DeclareDetector(vec![a - p, m + a - p])).collect()
        };

        push_layers(out, self.round_layers(true));
        if r >= 2 {
            push_layers(out, self.round_layers(false));
            out.extend_gates(g1_detectors());
            if r >= 3 {
                let mut body = Vec::new();
                push_layers(&mut body, self.round_layers(false));
                body.extend(g1_detectors());
                body.extend((0..b).map(|q| PhysicalGate::DeclareDetector(vec![m - q, 2 * m - q])));
                out.push(PhysicalGate::Repeat(r - 2, body));
            }
        }
        push_layers(out, self.epilogue_layers());
        if r >= 2 {
            out.extend_gates((0..b).map(|q| PhysicalGate::DeclareDetector(vec![b - q, m + b - q])));
        }

        let start = map.len();
        for n in 1..=r {
            for j in 0..m {
                let p = self.subpos[j];
                map.push(if self.class_of[j] == 0 {
                    if n == 1 { start + p } else { start + a + (n - 2) * m + b + p }
                } else {
                    start + a + (n - 1) * m + p
                });
            }
        }
    }
}

/// Append `layers` as gates delimited by `TICK`s, dropping empty layers.
fn push_layers<S: GateSink>(out: &mut S, layers: Vec<Vec<Gate>>) {
    for layer in layers {
        if layer.is_empty() {
            continue;
        }
        out.extend_gates(layer);
        out.push(PhysicalGate::Tick);
    }
}

/// Emit one syndrome round under the sequential conflict-phase schedule: a
/// reset layer, each conflict class's edge-colored CNOT layers in class order
/// (consistent ordering for every conflicting pair), and a measure layer in
/// check order — so the record stream matches the MPP stream exactly.
fn emit_round_sequential<S: GateSink>(checks: &[&PauliAxis<usize>], base: usize, out: &mut S) {
    let m = checks.len();
    let classes = conflict_classes(checks);
    let n_classes = classes.iter().copied().max().map_or(0, |c| c + 1);

    let mut layers: Vec<Vec<Gate>> = vec![(0..m)
        .map(|j| PhysicalGate::Reset(Pauli::X, base + j))
        .collect()];
    for c in 0..n_classes {
        let part: Vec<Site> = checks
            .iter()
            .enumerate()
            .filter(|&(j, _)| classes[j] == c)
            .flat_map(|(j, check)| sites(check).map(move |(q, p)| (j, q, p)))
            .collect();
        let mut qid: BTreeMap<usize, usize> = BTreeMap::new();
        for &(_, q, _) in &part {
            let next = m + qid.len();
            qid.entry(q).or_insert(next);
        }
        let edges: Vec<(usize, usize)> = part.iter().map(|&(j, q, _)| (j, qid[&q])).collect();
        let colors = bipartite_edge_coloring(m + qid.len(), &edges);
        let n_colors = colors.iter().copied().max().map_or(0, |k| k + 1);
        let mut buckets = vec![Vec::new(); n_colors];
        for (i, &(j, q, p)) in part.iter().enumerate() {
            buckets[colors[i]].push(PhysicalGate::Controlled(p, base + j, q));
        }
        layers.extend(buckets);
    }
    layers.push(
        checks
            .iter()
            .enumerate()
            .map(|(j, check)| {
                debug_assert!(
                    matches!(check.sign, Sign::One | Sign::NegOne),
                    "non-Hermitian sign on a syndrome MPP"
                );
                PhysicalGate::Measure(Pauli::X, base + j, check.sign == Sign::NegOne)
            })
            .collect(),
    );
    push_layers(out, layers);
}

/// Whether a `REPEAT` body is the canonical `d_rounds` shape for `checks`: the
/// same `m` MPPs in order, then exactly the `m` round-over-round detectors
/// `[j+1, m+j+1]`.
fn repeat_matches(body: &[Gate], checks: &[&PauliAxis<usize>]) -> bool {
    let m = checks.len();
    if body.len() != 2 * m {
        return false;
    }
    for (j, gate) in body.iter().enumerate() {
        let ok = if j < m {
            matches!(gate, PhysicalGate::MPP(p)
                if p.sign == checks[j].sign && p.pauli_string == checks[j].pauli_string)
        } else {
            let i = j - m;
            matches!(gate, PhysicalGate::DeclareDetector(recs) if *recs == vec![i + 1, m + i + 1])
        };
        if !ok {
            return false;
        }
    }
    true
}

fn extend_identity(map: &mut Vec<usize>, n: usize) {
    let s = map.len();
    map.extend(s..s + n);
}

/// Number of measurement records a gate sequence produces (`REPEAT` bodies
/// counted by their repeat factor).
fn records_in(gates: &[Gate]) -> usize {
    gates
        .iter()
        .map(|g| match g {
            PhysicalGate::MPP(_) | PhysicalGate::Measure(_, _, _) => 1,
            PhysicalGate::Repeat(n, body) => n * records_in(body),
            _ => 0,
        })
        .sum()
}

/// Expand a gate stream that must keep its record order (an unmatched `REPEAT`
/// body): rounds lower via the sequential schedule, everything else — including
/// rec-consumers, whose offsets stay valid — passes through unchanged.
fn expand_sequential_only(gates: &[Gate], base: usize) -> Vec<Gate> {
    let mut out = Vec::new();
    let mut round: Vec<&PauliAxis<usize>> = Vec::new();
    for g in gates {
        match g {
            PhysicalGate::MPP(p) => round.push(p),
            other => {
                if !round.is_empty() {
                    emit_round_sequential(&round, base, &mut out);
                    round.clear();
                }
                match other {
                    PhysicalGate::Repeat(n, body) => {
                        out.push(PhysicalGate::Repeat(*n, expand_sequential_only(body, base)));
                    }
                    g => out.push(g.clone()),
                }
            }
        }
    }
    if !round.is_empty() {
        emit_round_sequential(&round, base, &mut out);
    }
    out
}

/// Lower every MPP syndrome round of `gates` to the left–right /
/// conflict-phase schedules, rewriting downstream `rec[-k]` offsets through
/// the resulting record permutation. `base` is the first ancilla id (one past
/// the largest data-qubit id); the j-th check of every round uses ancilla
/// `base + j`, reset and reused round over round — the same pool as the legacy
/// expansion, so `sec_ancilla_pool` stays valid.
/// Emitted into a [`GateSink`] rather than returned: the expansion is produced
/// strictly in order and never read back, so the writing path can stream it to
/// disk instead of holding it (see [`crate::checks_to_physical_circuit::StimWriter`]).
pub(crate) fn expand_gates_lrc_into<S: GateSink>(gates: &[Gate], base: usize, out: &mut S) {
    let mut map: Vec<usize> = Vec::new();
    expand_gates_lrc_with_map(gates, base, out, &mut map);
}

/// [`expand_gates_lrc_into`] over a caller-owned record map, so a circuit can be
/// expanded in consecutive chunks instead of all at once. `map` carries the
/// old→new record correspondence across calls, which is what lets a later chunk's
/// `rec[-k]` still resolve against records emitted by an earlier one. Splitting is
/// only sound where no `MPP` run straddles the boundary — the streaming lowerer
/// cuts between deformations, each of which ends in the split's `M`/`CNOT` gates.
pub(crate) fn expand_gates_lrc_with_map<S: GateSink>(
    gates: &[Gate],
    base: usize,
    out: &mut S,
    map: &mut Vec<usize>,
) {
    // Old→new record permutation; `map.len()` doubles as the running old-record
    // count, which equals the running new-record count everywhere outside a
    // segment (both schedules preserve per-segment record totals).
    let rewrite = |map: &[usize], recs: &[usize]| -> Vec<usize> {
        recs.iter().map(|&k| map.len() - map[map.len() - k]).collect()
    };

    let mut i = 0;
    while i < gates.len() {
        match &gates[i] {
            PhysicalGate::MPP(_) => {
                let mut checks: Vec<&PauliAxis<usize>> = Vec::new();
                let mut j = i;
                while let Some(PhysicalGate::MPP(p)) = gates.get(j) {
                    checks.push(p);
                    j += 1;
                }
                let m = checks.len();
                let matched_repeat = match gates.get(j) {
                    Some(PhysicalGate::Repeat(n, body)) if repeat_matches(body, &checks) => {
                        Some(*n)
                    }
                    _ => None,
                };
                let r = 1 + matched_repeat.unwrap_or(0);
                let classes = conflict_classes(&checks);
                let n_classes = classes.iter().copied().max().map_or(0, |c| c + 1);
                if std::env::var("SEC_DEBUG").is_ok() {
                    let mixed = checks
                        .iter()
                        .filter(|c| {
                            let mut ps = sites(c).map(|(_, p)| p);
                            let first = ps.next();
                            ps.any(|p| Some(p) != first)
                        })
                        .count();
                    eprintln!(
                        "[sec] segment: m={m} r={r} chi={n_classes} mixed_checks={mixed} -> {}",
                        if n_classes == 2 && r >= 2 { "staggered" } else { "sequential" }
                    );
                }
                if n_classes == 2 && r >= 2 {
                    StaggeredSeg::new(&checks, classes, base).emit(r, out, map);
                } else {
                    // Sequential: reference round, then (if matched) a body of
                    // the same round with the original detectors — offsets are
                    // preserved because record order is.
                    emit_round_sequential(&checks, base, out);
                    if let Some(n) = matched_repeat {
                        let mut body = Vec::new();
                        emit_round_sequential(&checks, base, &mut body);
                        body.extend(
                            (0..m).map(|k| PhysicalGate::DeclareDetector(vec![k + 1, m + k + 1])),
                        );
                        out.push(PhysicalGate::Repeat(n, body));
                    }
                    extend_identity(map, r * m);
                }
                i = j + matched_repeat.map_or(0, |_| 1);
            }
            PhysicalGate::Measure(basis, q, invert) => {
                extend_identity(map, 1);
                out.push(PhysicalGate::Measure(*basis, *q, *invert));
                i += 1;
            }
            PhysicalGate::XCorrection(k, q) => {
                out.push(PhysicalGate::XCorrection(rewrite(map, &[*k])[0], *q));
                i += 1;
            }
            PhysicalGate::DeclareDetector(recs) => {
                out.push(PhysicalGate::DeclareDetector(rewrite(map, recs)));
                i += 1;
            }
            PhysicalGate::DeclareObservable(idx, recs) => {
                out.push(PhysicalGate::DeclareObservable(*idx, rewrite(map, recs)));
                i += 1;
            }
            PhysicalGate::Repeat(n, body) => {
                // A REPEAT not attached to a reference round (does not occur
                // today): lower it order-preservingly so in-body offsets and
                // the identity record map stay valid.
                extend_identity(map, n * records_in(body));
                out.push(PhysicalGate::Repeat(*n, expand_sequential_only(body, base)));
                i += 1;
            }
            g => {
                out.push(g.clone());
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::PauliString;

    fn axis(sites: &[(usize, Pauli)]) -> PauliAxis<usize> {
        PauliAxis { sign: Sign::One, pauli_string: PauliString::new(sites.to_vec()) }
    }

    /// The expansion collected into a `Vec`, which these tests inspect.
    fn expand_gates_lrc(gates: &[Gate], base: usize) -> Vec<Gate> {
        let mut out = Vec::new();
        expand_gates_lrc_into(gates, base, &mut out);
        out
    }

    /// `[MPP×m, REPEAT(r−1, [MPP×m, round detectors])]` — the exact `d_rounds`
    /// shape the walker matches.
    fn round_stream(checks: &[PauliAxis<usize>], r: usize) -> Vec<Gate> {
        let m = checks.len();
        let mpps: Vec<Gate> = checks.iter().cloned().map(PhysicalGate::MPP).collect();
        let mut out = mpps.clone();
        if r > 1 {
            let mut body = mpps;
            body.extend((0..m).map(|j| PhysicalGate::DeclareDetector(vec![j + 1, m + j + 1])));
            out.push(PhysicalGate::Repeat(r - 1, body));
        }
        out
    }

    fn unroll(gates: &[Gate]) -> Vec<Gate> {
        let mut out = Vec::new();
        for g in gates {
            match g {
                PhysicalGate::Repeat(n, body) => {
                    let flat = unroll(body);
                    for _ in 0..*n {
                        out.extend(flat.iter().cloned());
                    }
                }
                g => out.push(g.clone()),
            }
        }
        out
    }

    /// Replay an unrolled SEC circuit. Returns per-measurement (ancilla,
    /// instance#) in record order, per-ancilla-instance controlled-Pauli events
    /// (data qubit, pauli, time), and resolved detector references.
    struct Replay {
        /// Record stream: (ancilla qubit, 0-based instance number).
        records: Vec<(usize, usize)>,
        /// (ancilla, instance) -> gate events (qubit, pauli, global time).
        gates_of: BTreeMap<(usize, usize), Vec<(usize, Pauli, usize)>>,
        /// Each detector's referenced absolute record indices.
        detectors: Vec<Vec<usize>>,
    }

    fn replay(gates: &[Gate]) -> Replay {
        let mut time = 0usize;
        let mut instance: BTreeMap<usize, usize> = BTreeMap::new();
        let mut records = Vec::new();
        let mut gates_of: BTreeMap<(usize, usize), Vec<(usize, Pauli, usize)>> = BTreeMap::new();
        let mut detectors = Vec::new();
        let mut busy: std::collections::BTreeSet<usize> = Default::default();
        for g in gates {
            match g {
                PhysicalGate::Tick => {
                    time += 1;
                    busy.clear();
                }
                PhysicalGate::Reset(_, q) => {
                    assert!(busy.insert(*q), "qubit {q} touched twice in layer {time}");
                    *instance.entry(*q).or_insert(0) += 1;
                }
                PhysicalGate::Controlled(p, a, q) => {
                    assert!(busy.insert(*a), "ancilla {a} touched twice in layer {time}");
                    assert!(busy.insert(*q), "qubit {q} touched twice in layer {time}");
                    gates_of.entry((*a, instance[a] - 1)).or_default().push((*q, *p, time));
                }
                PhysicalGate::Measure(_, q, _) => {
                    assert!(busy.insert(*q), "qubit {q} touched twice in layer {time}");
                    records.push((*q, instance[q] - 1));
                }
                PhysicalGate::DeclareDetector(recs) => {
                    detectors
                        .push(recs.iter().map(|&k| records.len() - k).collect());
                }
                _ => {}
            }
        }
        Replay { records, gates_of, detectors }
    }

    /// The exact hook-cancellation condition: for every pair of ancilla
    /// instances whose checks share anticommuting site Paulis, the number of
    /// shared sites where the two gates occur in "crossed" order must be even.
    fn assert_hooks_cancel(replay: &Replay, base: usize, checks: &[PauliAxis<usize>]) {
        let keys: Vec<(usize, usize)> = replay.gates_of.keys().copied().collect();
        for (i, &ka) in keys.iter().enumerate() {
            for &kb in &keys[i + 1..] {
                if ka.0 == kb.0 {
                    continue; // same check slot: identical Paulis, no hooks
                }
                let (ga, gb) = (&replay.gates_of[&ka], &replay.gates_of[&kb]);
                let mut crossings = 0usize;
                let mut shared = 0usize;
                for &(qa, pa, ta) in ga {
                    for &(qb, pb, tb) in gb {
                        if qa == qb && pa != pb {
                            shared += 1;
                            assert_ne!(ta, tb, "anticommuting gates share a layer");
                            if ta > tb {
                                crossings += 1;
                            }
                        }
                    }
                }
                assert!(
                    shared % 2 == 0,
                    "checks {} and {} anticommute (invalid stabilizer input)",
                    checks[ka.0 - base].pauli_string.len(),
                    checks[kb.0 - base].pauli_string.len(),
                );
                assert!(
                    crossings % 2 == 0,
                    "uncancelled CZ hook between ancilla instances {ka:?} and {kb:?}",
                );
            }
        }
    }

    fn assert_round_structure(
        replay: &Replay,
        base: usize,
        m: usize,
        r: usize,
        expect_check_order: bool,
    ) {
        // One record per (check, round); every check measured r times.
        assert_eq!(replay.records.len(), r * m);
        for j in 0..m {
            let n = replay.records.iter().filter(|&&(q, _)| q == base + j).count();
            assert_eq!(n, r, "check {j} measured {n} times, expected {r}");
        }
        if expect_check_order {
            let expected: Vec<(usize, usize)> =
                (0..r).flat_map(|n| (0..m).map(move |j| (base + j, n))).collect();
            assert_eq!(replay.records, expected, "sequential schedule permuted records");
        }
        // Every round detector compares consecutive instances of one check.
        assert_eq!(replay.detectors.len(), (r - 1) * m);
        let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
        for det in &replay.detectors {
            assert_eq!(det.len(), 2);
            let (a, b) = (replay.records[det[0]], replay.records[det[1]]);
            assert_eq!(a.0, b.0, "detector mixes checks: {a:?} vs {b:?}");
            assert_eq!(
                a.1.abs_diff(b.1),
                1,
                "detector compares non-consecutive rounds: {a:?} vs {b:?}"
            );
            *seen.entry(a.0).or_insert(0) += 1;
        }
        for j in 0..m {
            assert_eq!(seen[&(base + j)], r - 1);
        }
    }

    /// A commuting CSS-style round: one weight-4 X check conflicting with two
    /// weight-2 Z checks (bipartite conflict graph ⇒ staggered path).
    fn css_checks() -> Vec<PauliAxis<usize>> {
        vec![
            axis(&[(0, Pauli::X), (1, Pauli::X), (2, Pauli::X), (3, Pauli::X)]),
            axis(&[(0, Pauli::Z), (1, Pauli::Z)]),
            axis(&[(2, Pauli::Z), (3, Pauli::Z)]),
        ]
    }

    /// A pairwise-commuting conflict triangle (χ = 3 ⇒ sequential fallback):
    /// X⊗X, Z⊗Z, Y⊗Y on the same two qubits.
    fn triangle_checks() -> Vec<PauliAxis<usize>> {
        vec![
            axis(&[(0, Pauli::X), (1, Pauli::X)]),
            axis(&[(0, Pauli::Z), (1, Pauli::Z)]),
            axis(&[(0, Pauli::Y), (1, Pauli::Y)]),
        ]
    }

    #[test]
    fn edge_coloring_is_proper_and_minimal() {
        // Complete bipartite K_{4,5} plus a pendant edge: Δ = 5.
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for u in 0..4 {
            for v in 0..5 {
                edges.push((u, 4 + v));
            }
        }
        edges.push((9, 4));
        let colors = bipartite_edge_coloring(10, &edges);
        let delta = 5;
        assert!(colors.iter().all(|&c| c < delta));
        for (i, &(u1, v1)) in edges.iter().enumerate() {
            for (j2, &(u2, v2)) in edges.iter().enumerate().skip(i + 1) {
                if u1 == u2 || v1 == v2 || u1 == v2 || v1 == u2 {
                    assert_ne!(colors[i], colors[j2], "edges {i} and {j2} clash");
                }
            }
        }
    }

    #[test]
    fn conflict_classes_bipartite_and_triangle() {
        let css = css_checks();
        let refs: Vec<&PauliAxis<usize>> = css.iter().collect();
        let classes = conflict_classes(&refs);
        assert_eq!(classes, vec![0, 1, 1]);

        let tri = triangle_checks();
        let refs: Vec<&PauliAxis<usize>> = tri.iter().collect();
        let classes = conflict_classes(&refs);
        assert_eq!(classes.iter().copied().max(), Some(2));
    }

    #[test]
    fn staggered_schedule_invariants() {
        let checks = css_checks();
        let (m, r, base) = (checks.len(), 5, 4);
        let expanded = expand_gates_lrc(&round_stream(&checks, r), base);
        assert!(
            matches!(expanded.iter().find(|g| matches!(g, PhysicalGate::Repeat(..))),
                Some(PhysicalGate::Repeat(n, _)) if *n == r - 2),
            "staggered segment should carry a REPEAT(r-2) steady body"
        );
        let replay = replay(&unroll(&expanded));
        assert_round_structure(&replay, base, m, r, false);
        assert_hooks_cancel(&replay, base, &checks);
    }

    #[test]
    fn sequential_fallback_preserves_record_order() {
        let checks = triangle_checks();
        let (m, r, base) = (checks.len(), 4, 2);
        let expanded = expand_gates_lrc(&round_stream(&checks, r), base);
        let replay = replay(&unroll(&expanded));
        assert_round_structure(&replay, base, m, r, true);
        assert_hooks_cancel(&replay, base, &checks);
    }

    #[test]
    fn single_round_uses_sequential_schedule() {
        let checks = css_checks();
        let base = 4;
        let expanded = expand_gates_lrc(&round_stream(&checks, 1), base);
        let replay = replay(&unroll(&expanded));
        let expected: Vec<(usize, usize)> = (0..checks.len()).map(|j| (base + j, 0)).collect();
        assert_eq!(replay.records, expected);
        assert_hooks_cancel(&replay, base, &checks);
    }

    #[test]
    fn record_permutation_rewrites_consumers() {
        // Reference every MPP record from a trailing consumer; after expansion
        // each rewritten offset must resolve to the measurement of the same
        // (round, check) instance.
        let checks = css_checks();
        let (m, r, base) = (checks.len(), 4, 4);
        let mut stream = round_stream(&checks, r);
        for k in 1..=r * m {
            stream.push(PhysicalGate::DeclareDetector(vec![k]));
        }
        let expanded = expand_gates_lrc(&stream, base);
        let replay = replay(&unroll(&expanded));
        // Trailing detectors sit after the (r-1)*m round detectors.
        let trailing = &replay.detectors[(r - 1) * m..];
        assert_eq!(trailing.len(), r * m);
        for (i, det) in trailing.iter().enumerate() {
            // Consumer i was emitted with old offset k = i+1: old record
            // rm−1−i = round n check j with n = (rm−1−i)/m, j = (rm−1−i)%m.
            let old_abs = r * m - 1 - i;
            let (n, j) = (old_abs / m, old_abs % m);
            assert_eq!(det.len(), 1);
            assert_eq!(
                replay.records[det[0]],
                (base + j, n),
                "consumer {i} rewired to the wrong record"
            );
        }
    }
}
