use crate::pbc::{CodeQubit, Pauli, PhysicalPauliString, PhysicalQubit};
use petgraph::graph::{NodeIndex, UnGraph};
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

fn paulis_commute(p: Pauli, q: Pauli) -> bool {
    p == Pauli::I || q == Pauli::I || p == q
}

/// A measurement (surgery) graph together with its *ports* — the graph vertices
/// the logical operator's support maps to under the port function `f: L → P`.
/// Bridges (Lemma 25 of arXiv:2503.10390) attach to these ports.
pub struct SurgeryGraph<K> {
    /// Node weights label the port vertices with their code qubit
    /// (`Some(CodeQubit { block, index })`); all other vertices are `None`. Edges
    /// (the ancilla / edge qubits) carry no weight. The ports — the graph vertices
    /// the operator's support maps to — are exactly the `Some`-weighted vertices,
    /// recovered grouped by block via [`SurgeryGraph::ports_by_block`].
    pub graph: UnGraph<Option<CodeQubit<K>>, ()>,
    /// The cellulated cycle basis carried through from the congestion-aware
    /// expander: each entry is a face (cycle check / stabilizer) as the set of
    /// graph edges (vertex pairs, in `graph`'s vertex ids) bounding it. These are
    /// the bounded-weight faces produced by [`cellulate`] and become the cycle
    /// checks of the surgery code.
    pub cycle_checks: Vec<Vec<Edge>>,
    /// Path-matching edges per code stabilizer (aligned with the `stabilizers`
    /// passed to [`surgery_graph`]), in `graph`'s vertex ids. The Definition 2
    /// (arXiv:2503.10390) *code check* for stabilizer `S` extends `S` with `X` on
    /// these edge qubits; a stabilizer that commutes with the operator everywhere
    /// has an empty list. Edge ids are stable through expander construction,
    /// thickening (level 0), and cellulation, so they index the final graph.
    pub path_matching: Vec<Vec<Edge>>,
}

impl<K: Ord + Copy> SurgeryGraph<K> {
    /// The ports — the graph vertices the operator's support maps to — grouped by
    /// code block, recovered from the node weights (a port is a `Some`-weighted
    /// vertex). Deterministic: a `BTreeMap` keyed by block, each block's vertices
    /// in ascending node-id order, so [`bridge_surgery_graphs`] feeds a stable
    /// order to `skiptree_labeling` (whose result depends on it).
    pub fn ports_by_block(&self) -> BTreeMap<K, Vec<NodeIndex>> {
        let mut ports: BTreeMap<K, Vec<NodeIndex>> = BTreeMap::new();
        for v in self.graph.node_indices() {
            if let Some(cq) = self.graph[v] {
                ports.entry(cq.block).or_default().push(v);
            }
        }
        ports
    }
}

pub fn surgery_graph<K: Ord + Copy>(
    stabilizers: &Vec<PhysicalPauliString>,
    operator: &PhysicalPauliString,
    block_name: K,
    config: &SurgeryGraphConfig,
) -> SurgeryGraph<K> {
    surgery_graph_cached(stabilizers, operator, block_name, config, &mut ShapeCache::new())
}

/// Cellulated structural graphs memoized on the operator's position-space
/// *shape*: its support weight plus the sorted multiset of path-matching edges
/// (position pairs). Everything expensive in [`surgery_graph_cached`] — expander
/// construction, thickening, cellulation — is a pure function of that shape (the
/// RNG is reseeded from `config.seed` per build), so two operators sharing a key
/// share the structure, even when they live on different qubits or anticommute
/// with different stabilizers (e.g. the same logical measured on two translated
/// qubits of a quasi-cyclic code). The key deliberately sorts away edge order:
/// any cellulation of the same edge multiset contains every path-matching edge
/// and its cycle checks are faces of the graph itself, so the reuse is exact,
/// though outputs can differ from what an uncached build of the later operator
/// would have produced (a different but equally valid surgery graph).
///
/// One cache is valid for one `SurgeryGraphConfig`; keep it scoped to a single
/// lowering pass alongside the exact per-support cache.
pub struct ShapeCache(HashMap<(usize, Vec<Edge>), Cellulation>);

impl ShapeCache {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
}

/// [`surgery_graph`] with the structural stages memoized in `cache`; see
/// [`ShapeCache`]. The per-operator finishing — path matchings, lifting the
/// structural graph to a node-weighted one, port labeling — still runs per call.
pub fn surgery_graph_cached<K: Ord + Copy>(
    stabilizers: &Vec<PhysicalPauliString>,
    operator: &PhysicalPauliString,
    block_name: K,
    config: &SurgeryGraphConfig,
    cache: &mut ShapeCache,
) -> SurgeryGraph<K> {
    let (path_graph, path_matching) = path_matching_graph(&stabilizers, &operator);
    // The shape key: a sorted multiset (not a set — parallel edges from two
    // stabilizers matching the same position pair must survive) of the path
    // graph's edges, plus its vertex count for edgeless corner cases.
    let mut shape_edges: Vec<Edge> = path_graph
        .edge_indices()
        .map(|e| {
            let (a, b) = path_graph.edge_endpoints(e).expect("edge has endpoints");
            canonical_edge(a.index(), b.index())
        })
        .collect();
    shape_edges.sort_unstable();
    let shape_edge_count = shape_edges.len();
    let build = || {
        let expander = build_congestion_aware_expander(
            &path_graph,
            config,
            &mut StdRng::seed_from_u64(config.seed),
        );
        let thickend = thicken(&expander);
        let cellulated = cellulate(&thickend, config.max_check_degree);
        eprintln!(
            "[surgery_graph] op_weight={} base_vertices={} base_edges={} levels={} \
             lambda2={:.3} edge-qubits={} (chords={})",
            operator.len(),
            thickend.base_vertices,
            expander.graph.edge_count(),
            thickend.levels,
            expander.lambda2,
            cellulated.graph.edge_count(),
            cellulated.chords.len(),
        );
        cellulated
    };
    let uncached;
    let cellulated: &Cellulation = if config.caching {
        match cache.0.entry((path_graph.node_count(), shape_edges)) {
            Entry::Occupied(entry) => {
                eprintln!(
                    "[surgery_graph] shape cache hit! op_weight={} with {shape_edge_count} \
                     matching edges has the same position-space shape as an earlier build.",
                    operator.len(),
                );
                entry.into_mut()
            }
            Entry::Vacant(entry) => entry.insert(build()),
        }
    } else {
        uncached = build();
        &uncached
    };
    // The cellulated graph is unlabeled (`UnGraph<(), ()>`); lift it to carry node
    // weights, then label each port vertex with its code qubit. The ports are the
    // operator's support, vertices `0..|support|` in the order `path_matching_graph`
    // created them: that position id is preserved through expander construction
    // (node indices unchanged), thickening (level 0 maps `v ↦ v`), and cellulation
    // (only vertices/edges added), so port position `p` is still vertex `p` here.
    let mut graph: UnGraph<Option<CodeQubit<K>>, ()> =
        cellulated.graph.map(|_, _| None, |_, _| ());
    // Label port position `p` with the absolute within-block code-qubit index `q.0`
    // — the absolute id lives only in the label, never as a structural vertex id.
    for (p, (q, _)) in operator.iter().enumerate() {
        graph[NodeIndex::new(p)] = Some(CodeQubit {
            block: block_name,
            index: q.0,
        });
    }
    SurgeryGraph {
        graph,
        cycle_checks: cellulated.checks.clone(),
        path_matching,
    }
}

/// Lemma 25 (Bridging Lemma, arXiv:2503.10390): connect two measurement graphs
/// for operators `L₁, L₂` (non-overlapping support) into one for the product
/// `L₁·L₂` by adding a *bridge* of `distance` edges between their ports.
///
/// The bridge is a matching (Definition 22: non-overlapping edges): the first
/// `d = min(distance, |P₁|, |P₂|)` ports of each side are paired one-to-one.
/// The result's ports are `P₁ ∪ P₂`, the port set for `L₁·L₂`. Bridging two
/// connected graphs with `d` edges adds `d − 1` new basis cycles (Lemma 23),
/// which would then be cellulated into cycle checks.
///
/// The bridge here is the *low-congestion* SkipTree bridge of Lemma 24 / Lemma 10
/// of Ref. [71] (arXiv:2410.03628): the two port subgraphs are SkipTree-labeled
/// and equal labels are matched (see [`skiptree_labeling`]), which keeps the new
/// bridge cycles short (length ≤ max(γ, 8)) and low-congestion (ρ + 2).
pub fn bridge_surgery_graphs<K: Ord + Copy + std::fmt::Debug>(
    left: &SurgeryGraph<K>,
    right: &SurgeryGraph<K>,
    distance: usize,
) -> SurgeryGraph<K> {
    // Disjoint union: copy `left`, then append `right` with its ids shifted.
    let mut graph = left.graph.clone();
    let offset = left.graph.node_count();
    // Carry the right side's node weights across verbatim. `node_indices()` yields
    // `0..count` in order, so the appended ids are exactly `offset + original`, and
    // each port keeps its `CodeQubit { block, index }` — no relabeling needed.
    for n in right.graph.node_indices() {
        graph.add_node(right.graph[n].clone());
    }
    for e in right.graph.edge_indices() {
        let (a, b) = right.graph.edge_endpoints(e).expect("edge has endpoints");
        graph.add_edge(
            NodeIndex::new(offset + a.index()),
            NodeIndex::new(offset + b.index()),
            (),
        );
    }

    // The two graphs must use disjoint block keys (caller invariant) so the bridged
    // graph's node weights still identify each code block unambiguously.
    let left_blocks = left.ports_by_block();
    let right_blocks = right.ports_by_block();
    debug_assert!(
        left_blocks.keys().all(|b| !right_blocks.contains_key(b)),
        "bridge_surgery_graphs requires disjoint block keys; left={:?}, right={:?}",
        left_blocks.keys().collect::<Vec<_>>(),
        right_blocks.keys().collect::<Vec<_>>(),
    );

    // SkipTree-label each side's port subgraph; the bridge matches equal labels.
    // Ports are grouped by block, but the bridge labels them as one set, so
    // flatten across blocks (deterministic: `ports_by_block` is a `BTreeMap`).
    let left_ports: Vec<NodeIndex> = left_blocks.values().flatten().copied().collect();
    let right_ports: Vec<NodeIndex> = right_blocks.values().flatten().copied().collect();
    let left_order = skiptree_labeling(&left.graph, &left_ports);
    let right_order = skiptree_labeling(&right.graph, &right_ports);

    // Bridge B: connect the port labeled `i` on the left to the port labeled `i`
    // on the right, for the first `d` labels (a matching: labels are distinct).
    let d = distance.min(left_order.len()).min(right_order.len());
    for i in 0..d {
        let l = left_order[i];
        let r = NodeIndex::new(offset + right_order[i].index());
        graph.add_edge(l, r, ());
    }

    // The bridged graph's ports (`P₁ ∪ P₂`) need no separate bookkeeping: the
    // disjoint union already carried both sides' node weights across (left verbatim,
    // right shifted by `offset`), so `ports_by_block` on the result recovers them.

    // Carry both sides' cellulated cycle checks through the disjoint union,
    // shifting the right side's vertex ids by `offset`. (The `d − 1` new cycles
    // the bridge itself creates are not yet cellulated into checks here.)
    let mut cycle_checks = left.cycle_checks.clone();
    cycle_checks.extend(right.cycle_checks.iter().map(|face| {
        face.iter()
            .map(|&(a, b)| canonical_edge(offset + a, offset + b))
            .collect()
    }));

    // Path matchings for `L₁·L₂` are those of `L₁` followed by those of `L₂`
    // (the two operators have non-overlapping support and their own code
    // stabilizers); the right side's edge ids shift by `offset`.
    let mut path_matching = left.path_matching.clone();
    path_matching.extend(right.path_matching.iter().map(|edges| {
        edges
            .iter()
            .map(|&(a, b)| canonical_edge(offset + a, offset + b))
            .collect()
    }));

    SurgeryGraph {
        graph,
        cycle_checks,
        path_matching,
    }
}

/// SkipTree labeling pass: `LabelFirst` labels a node before recursing into its
/// children (each via `LabelLast`).
fn skiptree_label_first(v: usize, children: &[Vec<usize>], label: &mut [usize], index: &mut usize) {
    label[*index] = v;
    *index += 1;
    for &c in &children[v] {
        skiptree_label_last(c, children, label, index);
    }
}

/// SkipTree labeling pass: `LabelLast` labels a node only after recursing into
/// its children (each via `LabelFirst`).
fn skiptree_label_last(v: usize, children: &[Vec<usize>], label: &mut [usize], index: &mut usize) {
    for &c in &children[v] {
        skiptree_label_first(c, children, label, index);
    }
    label[*index] = v;
    *index += 1;
}

/// SkipTree labeling (Algorithm 1 of arXiv:2410.03628) of `ports` within
/// `graph`. Builds a spanning tree of the subgraph induced by the ports (rooted
/// at `ports[0]`), then labels vertices `0,1,…` by the alternating
/// `LabelFirst`/`LabelLast` recursion. Returns the ports in label order, so that
/// matching equal labels across two graphs (Lemma 10) yields short, low-congestion
/// bridge cycles.
///
/// Only the connected component of `ports[0]` in the induced subgraph is labeled
/// — Lemma 10 assumes the port set induces a connected subgraph; if it does not,
/// the bridge spans that component (and is correspondingly smaller).
fn skiptree_labeling<N>(graph: &UnGraph<N, ()>, ports: &[NodeIndex]) -> Vec<NodeIndex> {
    let k = ports.len();
    if k == 0 {
        return Vec::new();
    }
    let local: HashMap<NodeIndex, usize> = ports.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    // Adjacency of the subgraph induced by the ports (local indices).
    let mut induced: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &p) in ports.iter().enumerate() {
        for w in graph.neighbors(p) {
            if let Some(&j) = local.get(&w) {
                if j != i {
                    induced[i].push(j);
                }
            }
        }
    }

    // Spanning tree (BFS) rooted at port 0, recording parent→children.
    let mut visited = vec![false; k];
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); k];
    let mut queue = VecDeque::new();
    visited[0] = true;
    queue.push_back(0);
    while let Some(u) = queue.pop_front() {
        for &w in &induced[u] {
            if !visited[w] {
                visited[w] = true;
                children[u].push(w);
                queue.push_back(w);
            }
        }
    }

    let reached = visited.iter().filter(|&&b| b).count();
    let mut label = vec![usize::MAX; reached];
    let mut index = 0;
    skiptree_label_first(0, &children, &mut label, &mut index);
    label.iter().map(|&l| ports[l]).collect()
}
/// Build the path-matching graph for `operator` and record, per code stabilizer,
/// the path-matching edges that feed the Definition 2 (arXiv:2503.10390) *code
/// checks*.
///
/// For each stabilizer `S` we collect its anticommuting positions `K_S` — the
/// operator qubits where `S` and the operator fail to commute — and form a path
/// matching of `K_S` (Definition 5) by pairing them up. `|K_S|` is even (a code
/// stabilizer commutes with the logical operator, so they anticommute on an even
/// number of qubits), so the pairing visits every position in `K_S` exactly once
/// (odd) and every other vertex zero times (even). Each pair becomes a graph
/// edge, and a code check later extends `S` with `X` on those edge qubits.
///
/// Returns the graph together with a per-stabilizer list of path-matching edges,
/// aligned with the input `stabilizers` (a stabilizer that commutes with the
/// operator everywhere contributes an empty list).
fn path_matching_graph(
    stabilizers: &[PhysicalPauliString],
    operator: &PhysicalPauliString,
) -> (UnGraph<(), ()>, Vec<Vec<Edge>>) {
    let mut g = UnGraph::new_undirected();
    let operator_physical_qubits: Vec<PhysicalQubit> = operator.iter().map(|(q, _)| *q).collect();
    // One graph vertex per operator-support qubit, indexed by position (a dense
    // `0..|support|`), not by the qubit's absolute id — the surgery ancilla scales
    // with the operator's support weight, and `NodeIndex`es must be dense anyway.
    // The absolute id `q.0` survives only as the port's `CodeQubit` label in
    // `surgery_graph`; everything structural (graph ids, `Edge`s, `path_matching`)
    // lives in this position space, so the `offset` shifts in `bridge_surgery_graphs`
    // stay correct.
    let nodes: Vec<_> = operator_physical_qubits
        .iter()
        .map(|_| g.add_node(()))
        .collect();
    let position: BTreeMap<PhysicalQubit, usize> = operator_physical_qubits
        .iter()
        .enumerate()
        .map(|(p, &q)| (q, p))
        .collect();
    let mut path_matching: Vec<Vec<Edge>> = Vec::with_capacity(stabilizers.len());
    for stab in stabilizers {
        let mut anticommuting_positions: Vec<PhysicalQubit> = Vec::new();
        for (operator_qubit, operator_pauli) in operator.iter() {
            if let Some(&stab_pauli) = stab
                .iter()
                .find_map(|(q, p)| if *q == *operator_qubit { Some(p) } else { None })
            {
                if !paulis_commute(stab_pauli, *operator_pauli) {
                    anticommuting_positions.push(*operator_qubit);
                }
            }
        }
        // `|K(S)|` is even because a code stabilizer commutes with the logical
        // operator (they overlap on an even number of anticommuting qubits), so
        // `chunks_exact(2)` consumes every position. An odd count would silently
        // drop the last position, yielding a wrong path matching.
        debug_assert!(
            anticommuting_positions.len() % 2 == 0,
            "stabilizer anticommutes with the operator on an odd number of qubits \
             ({}); it does not commute with the logical operator",
            anticommuting_positions.len(),
        );
        // Pair the (even number of) anticommuting positions into a path matching.
        let mut edges: Vec<Edge> = Vec::new();
        for pair in anticommuting_positions.chunks_exact(2) {
            let (i, j) = (position[&pair[0]], position[&pair[1]]);
            g.add_edge(nodes[i], nodes[j], ());
            edges.push(canonical_edge(i, j));
        }
        path_matching.push(edges);
    }
    (g, path_matching)
}

/// Degree of `v` in a simple graph (no self-loops): each neighbor is counted once.
fn degree(g: &UnGraph<(), ()>, v: NodeIndex) -> usize {
    g.neighbors(v).count()
}

/// Algorithm 1 of "GeneCS: Synthesizing Resource-Efficient Code Surgery for
/// Arbitrary Quantum Stabilizer Codes" (Zhou, Javadi-Abhari, Li; arXiv:2605.21746):
/// *Conditioned Expander Construction*.
///
/// Starting from the path-matching graph `g0`, progressively add edges in a
/// randomized manner until the graph's second-smallest Laplacian eigenvalue
/// `λ₂` reaches `2·β`. By Cheeger's inequality `β_G ≥ λ₂ / 2`, so this lower
/// bounds the Cheeger constant by `β` while reusing the structure already
/// present in `g0` to keep the number of added edges small.
///
/// `tau` bounds the number of Phase-2 augmentation rounds. If the threshold is
/// never reached the best-effort augmented graph is returned. The graph is kept
/// simple: duplicate edges are never added.
fn conditioned_expander_graph(
    g0: &UnGraph<(), ()>,
    beta: f64,
    tau: usize,
    rng: &mut impl Rng,
) -> UnGraph<(), ()> {
    let mut g = g0.clone();
    let nodes: Vec<NodeIndex> = g.node_indices().collect();
    let n = nodes.len();
    let threshold = 2.0 * beta;

    if n < 2 || laplacian_lambda2(&g) >= threshold {
        return g;
    }

    // Phase 1: Regularization to maximum degree.
    //
    // Δ is the maximum degree over all vertices, fixed for the whole phase
    // (we only ever connect two vertices that are both below Δ, so no vertex
    // can exceed it). At each step we pick a non-adjacent pair {u, v} that both
    // still have spare capacity, with probability proportional to the product
    // of their degree deficits Δ(u)·Δ(v).
    let max_deg = nodes.iter().map(|&v| degree(&g, v)).max().unwrap_or(0);
    loop {
        let deficit: Vec<usize> = nodes.iter().map(|&v| max_deg - degree(&g, v)).collect();

        let mut candidates: Vec<(usize, usize)> = Vec::new();
        let mut weights: Vec<usize> = Vec::new();
        for i in 0..n {
            if deficit[i] == 0 {
                continue;
            }
            for j in (i + 1)..n {
                if deficit[j] == 0 || g.find_edge(nodes[i], nodes[j]).is_some() {
                    continue;
                }
                candidates.push((i, j));
                weights.push(deficit[i] * deficit[j]);
            }
        }
        if candidates.is_empty() {
            break;
        }

        let dist = WeightedIndex::new(&weights).expect("weights are positive");
        let (i, j) = candidates[dist.sample(rng)];
        g.add_edge(nodes[i], nodes[j], ());
        if laplacian_lambda2(&g) >= threshold {
            return g;
        }
    }

    // Phase 2: Degree augmentation.
    //
    // If structure alone was insufficient, add random 1-regular layers: each
    // round draws a uniformly random matching over the vertices and adds its
    // edges one at a time, checking λ₂ after each addition.
    for _ in 0..tau {
        let mut unmatched = nodes.clone();
        unmatched.shuffle(rng);
        let mut k = 0;
        while k + 1 < unmatched.len() {
            let (u, v) = (unmatched[k], unmatched[k + 1]);
            k += 2;
            if g.find_edge(u, v).is_some() {
                continue;
            }
            g.add_edge(u, v, ());
            if laplacian_lambda2(&g) >= threshold {
                return g;
            }
        }
    }

    g
}

/// Tuning parameters for the [`surgery_graph`] construction pipeline: the
/// randomized congestion-aware expander build (`trials` and the three fields
/// forwarded to [`congestion_aware_expander`]), the RNG `seed` that makes that
/// build deterministic, and the cellulation `max_check_degree`. [`Default`]
/// reproduces the values that were previously hard-coded.
#[derive(Clone, Copy, Debug)]
pub struct SurgeryGraphConfig {
    /// Number of randomized expander constructions to try (smallest wins).
    pub trials: usize,
    /// `reset_period` passed to [`congestion_aware_expander`].
    pub reset_period: usize,
    /// `max_iterations` passed to [`congestion_aware_expander`].
    pub max_iterations: usize,
    /// `qubit_degree` (`d_q`) passed to [`congestion_aware_expander`].
    pub qubit_degree: usize,
    /// Seed for the RNG driving the (randomized) expander construction; fixing
    /// it makes [`surgery_graph`] deterministic.
    pub seed: u64,
    /// `max_check_degree` passed to [`cellulate`]: the largest face (cycle
    /// check) degree the zigzag cellulation may produce.
    pub max_check_degree: usize,
    /// Enables the surgery-graph caches (the full-support, per-block, and shape
    /// layers). Off, every measurement rebuilds its graphs from scratch — useful
    /// for timing the caches or ruling them out while debugging.
    pub caching: bool,
}

impl Default for SurgeryGraphConfig {
    fn default() -> Self {
        Self {
            trials: 100,
            reset_period: 2,
            max_iterations: 8,
            // d_q / d_c degree bounds. Set to 12 to match GeneCS (arXiv:2605.21746,
            // §7.1: bound set just above the benchmark codes' max check degree of
            // ~10). Higher d_q packs more cycles per decongestion layer (less
            // thickening) and higher d_c yields larger cellulation faces (far fewer
            // chord qubits); the prior 5/4 ran well under the codes' degree budget.
            qubit_degree: 12,
            seed: 42,
            max_check_degree: 12,
            caching: true,
        }
    }
}

fn build_congestion_aware_expander(
    g0: &UnGraph<(), ()>,
    config: &SurgeryGraphConfig,
    rng: &mut impl Rng,
) -> CongestionAwareExpander {
    let mut best: Option<CongestionAwareExpander> = None;
    for _ in 0..config.trials.max(1) {
        let g = congestion_aware_expander(
            g0,
            config.reset_period,
            config.max_iterations,
            config.qubit_degree,
            rng,
        );
        let smaller = best
            .as_ref()
            .is_none_or(|b| g.graph.edge_count() < b.graph.edge_count());
        // Prefer a graph that meets the threshold; among those, the smallest.
        if smaller {
            best = Some(g);
        }
    }
    best.expect("at least one trial runs")
}

/// An undirected edge in canonical (sorted) form, keyed by vertex position.
pub type Edge = (usize, usize);

fn canonical_edge(a: usize, b: usize) -> Edge {
    if a <= b { (a, b) } else { (b, a) }
}

/// A spanning forest over a fixed vertex set `{0, …, n-1}` supporting unique
/// path queries and incremental edge updates. Drives the cycle tracking of
/// Algorithm 2.
struct SpanningForest {
    adjacency: Vec<Vec<usize>>,
}

impl SpanningForest {
    fn new(n: usize) -> Self {
        Self {
            adjacency: vec![Vec::new(); n],
        }
    }

    fn add_edge(&mut self, a: usize, b: usize) {
        self.adjacency[a].push(b);
        self.adjacency[b].push(a);
    }

    fn remove_edge(&mut self, a: usize, b: usize) {
        self.adjacency[a].retain(|&x| x != b);
        self.adjacency[b].retain(|&x| x != a);
    }

    /// The unique forest path from `a` to `b` as a vertex sequence, or `None`
    /// if they lie in different trees. BFS with parent tracking.
    fn path(&self, a: usize, b: usize) -> Option<Vec<usize>> {
        if a == b {
            return Some(vec![a]);
        }
        let n = self.adjacency.len();
        let mut parent = vec![usize::MAX; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        visited[a] = true;
        queue.push_back(a);
        while let Some(x) = queue.pop_front() {
            for &y in &self.adjacency[x] {
                if visited[y] {
                    continue;
                }
                visited[y] = true;
                parent[y] = x;
                if y == b {
                    let mut path = vec![b];
                    let mut cur = b;
                    while cur != a {
                        cur = parent[cur];
                        path.push(cur);
                    }
                    path.reverse();
                    return Some(path);
                }
                queue.push_back(y);
            }
        }
        None
    }
}

/// BFS spanning forest of `g`, with vertices identified by their position in
/// `nodes` (`pos` is the inverse map).
fn bfs_spanning_forest(
    g: &UnGraph<(), ()>,
    nodes: &[NodeIndex],
    pos: &HashMap<NodeIndex, usize>,
) -> SpanningForest {
    let n = nodes.len();
    let mut forest = SpanningForest::new(n);
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        queue.push_back(start);
        while let Some(x) = queue.pop_front() {
            for w in g.neighbors(nodes[x]) {
                let y = pos[&w];
                if !visited[y] {
                    visited[y] = true;
                    forest.add_edge(x, y);
                    queue.push_back(y);
                }
            }
        }
    }
    forest
}

/// The edge set of the fundamental cycle formed by `node_path` together with
/// the closing edge `(a, b)`.
fn cycle_edges_from_path(node_path: &[usize], a: usize, b: usize) -> Vec<Edge> {
    let mut edges: Vec<Edge> = node_path
        .windows(2)
        .map(|w| canonical_edge(w[0], w[1]))
        .collect();
    edges.push(canonical_edge(a, b));
    edges
}

/// Algorithm 2: Dynamic Cycle Basis Maintenance.
///
/// Given the current spanning `forest` and a newly inserted graph edge
/// `(u, v)`, return the fundamental cycle it closes (if any) and update the
/// forest so it stays spanning. If `u` and `v` are in different trees the edge
/// extends the forest and no cycle is created; otherwise the cycle is the
/// forest path `u … v` plus `(u, v)`, and the forest swaps in `(u, v)` while
/// dropping one path edge (keeping tree paths bounded).
fn dynamic_cycle_basis_update(
    forest: &mut SpanningForest,
    u: usize,
    v: usize,
) -> Option<Vec<Edge>> {
    match forest.path(u, v) {
        None => {
            forest.add_edge(u, v);
            None
        }
        Some(node_path) => {
            let cycle = cycle_edges_from_path(&node_path, u, v);
            let (a, b) = (node_path[0], node_path[1]);
            forest.remove_edge(a, b);
            forest.add_edge(u, v);
            Some(cycle)
        }
    }
}

/// Greedy partition of the cycle basis into "decongestion layers", where the
/// number of layers is the decongestion layer count `t`. Each layer carries a
/// per-edge *load tracker* `ℓ_k(e)` — the number of cycles in that layer using
/// edge `e` — and a cycle may join a layer only while every edge stays within a
/// load bound `max_load` (`d_q`, the LDPC qubit-degree limit).
///
/// This is Algorithm 5 (Degree-Constrained Cycle Partition); Algorithm 3
/// (Dynamic Greedy Cycle Partition) is the special case `max_load == 1`, where
/// the bound forces layers to be strictly edge-disjoint.
struct CyclePartition {
    layer_loads: Vec<HashMap<Edge, usize>>,
    /// The cycles assigned to each layer, parallel to `layer_loads`. Retained so
    /// the downstream thickening step can place layer `P_r` onto level `r`.
    layer_cycles: Vec<Vec<Vec<Edge>>>,
    max_load: usize,
}

impl CyclePartition {
    /// Partition with qubit-degree bound `max_load` (`d_q`). Use `1` for the
    /// edge-disjoint Algorithm 3 behavior.
    fn with_max_load(max_load: usize) -> Self {
        debug_assert!(max_load >= 1, "load bound d_q must be at least 1");
        Self {
            layer_loads: Vec::new(),
            layer_cycles: Vec::new(),
            max_load: max_load.max(1),
        }
    }

    /// Place `cycle` into the first layer where every edge can absorb one more
    /// unit of load without exceeding `max_load`, bumping that layer's loads.
    /// Opens a new layer if none fits. Returns the updated layer count `t`.
    fn insert(&mut self, cycle: &[Edge]) -> usize {
        for (loads, cycles) in self
            .layer_loads
            .iter_mut()
            .zip(self.layer_cycles.iter_mut())
        {
            let fits = cycle
                .iter()
                .all(|e| loads.get(e).copied().unwrap_or(0) + 1 <= self.max_load);
            if fits {
                for &e in cycle {
                    *loads.entry(e).or_insert(0) += 1;
                }
                cycles.push(cycle.to_vec());
                return self.layer_loads.len();
            }
        }
        let mut loads = HashMap::new();
        for &e in cycle {
            *loads.entry(e).or_insert(0) += 1;
        }
        self.layer_loads.push(loads);
        self.layer_cycles.push(vec![cycle.to_vec()]);
        self.layer_loads.len()
    }

    fn layers(&self) -> usize {
        self.layer_loads.len()
    }

    /// The cycles of each layer; `layer_cycles()[r]` is the cycle group `P_r`.
    fn layer_cycles(&self) -> &[Vec<Vec<Edge>>] {
        &self.layer_cycles
    }
}

/// A working multigraph edge for the decongestion algorithm: its two endpoints
/// and the *trail* of original `G` edges it represents (a path, accumulated
/// through vertex suppressions).
type WorkingEdge = (usize, usize, Vec<Edge>);

/// Remove a single occurrence of `id` from an incidence list.
fn dl_remove_occurrence(list: &mut Vec<usize>, id: usize) {
    if let Some(p) = list.iter().position(|&x| x == id) {
        list.swap_remove(p);
    }
}

/// Delete working edge `id`, updating both endpoints' incidence (a self-loop is
/// listed — and so removed — twice at its vertex).
fn dl_remove_edge(wedges: &mut [Option<WorkingEdge>], incidence: &mut [Vec<usize>], id: usize) {
    if let Some((a, b, _)) = wedges[id].take() {
        dl_remove_occurrence(&mut incidence[a], id);
        dl_remove_occurrence(&mut incidence[b], id);
    }
}

/// Add a working edge `(a, b)` carrying `trail`; returns its id. A self-loop
/// (`a == b`) is listed twice at `a` so degree counts it as 2.
fn dl_add_edge(
    wedges: &mut Vec<Option<WorkingEdge>>,
    incidence: &mut [Vec<usize>],
    a: usize,
    b: usize,
    trail: Vec<Edge>,
) -> usize {
    let id = wedges.len();
    wedges.push(Some((a, b, trail)));
    incidence[a].push(id);
    incidence[b].push(id);
    id
}

/// BFS shortest path of working-edge ids from `src` to `dst`, never using edge
/// `excluded` (nor traversing self-loops). `None` if `dst` is unreachable.
fn dl_bfs_path(
    wedges: &[Option<WorkingEdge>],
    incidence: &[Vec<usize>],
    src: usize,
    dst: usize,
    excluded: usize,
) -> Option<Vec<usize>> {
    let n = incidence.len();
    let mut visited = vec![false; n];
    let mut parent_edge = vec![usize::MAX; n];
    let mut parent_vertex = vec![usize::MAX; n];
    let mut queue = VecDeque::new();
    visited[src] = true;
    queue.push_back(src);
    while let Some(u) = queue.pop_front() {
        if u == dst {
            let mut path = Vec::new();
            let mut cur = dst;
            while cur != src {
                path.push(parent_edge[cur]);
                cur = parent_vertex[cur];
            }
            path.reverse();
            return Some(path);
        }
        for &eid in &incidence[u] {
            if eid == excluded {
                continue;
            }
            let (a, b, _) = wedges[eid].as_ref().unwrap();
            let w = if *a == u { *b } else { *a };
            if w == u || visited[w] {
                continue;
            }
            visited[w] = true;
            parent_edge[w] = eid;
            parent_vertex[w] = u;
            queue.push_back(w);
        }
    }
    None
}

/// A shortest simple cycle (as working-edge ids) through vertex `v`, or `None` if
/// `v` lies on no cycle (every incident edge is a bridge — possible even at degree
/// ≥ 3, e.g. a cut vertex joined by bridges to otherwise-cyclic components).
/// Self-loops are length-1 cycles; otherwise close each incident edge with a
/// shortest path back to `v`.
fn dl_shortest_cycle(
    wedges: &[Option<WorkingEdge>],
    incidence: &[Vec<usize>],
    v: usize,
) -> Option<Vec<usize>> {
    for &eid in &incidence[v] {
        let (a, b, _) = wedges[eid].as_ref().unwrap();
        if a == b {
            return Some(vec![eid]);
        }
    }
    let mut best: Option<Vec<usize>> = None;
    for &eid in &incidence[v].clone() {
        let (a, b, _) = wedges[eid].as_ref().unwrap();
        let x = if *a == v { *b } else { *a };
        if let Some(path) = dl_bfs_path(wedges, incidence, x, v, eid) {
            let mut cycle = Vec::with_capacity(path.len() + 1);
            cycle.push(eid);
            cycle.extend(path);
            if best.as_ref().is_none_or(|b| cycle.len() < b.len()) {
                best = Some(cycle);
            }
        }
    }
    best
}

/// Freedman–Hastings Decongestion Lemma (Lemma A.0.2 of arXiv:2012.02249),
/// invoked by GeneCS as "the Decongestion Lemma applied to G" to seed `R₀`.
///
/// Returns a *weakly fundamental* cycle basis of the simple graph on `n`
/// vertices with edge set `edges`; with constant probability every edge appears
/// in only `O(log² V)` of the cycles (low congestion). This is the randomized
/// recursive algorithm A from the lemma's proof, made iterative.
///
/// The working graph is a multigraph — vertex suppression (case 2b) creates
/// parallel and self edges — so each working edge carries the *trail* of
/// original edges it stands for. A cycle found in the working graph expands to
/// a real cycle of `G` as the union of its edges' trails, which automatically
/// realizes the lemma's "replace edge `(x,y)` with `(x,v),(v,y)`" step. Rules,
/// applied in priority order until no edges remain:
///   1. a degree-1 vertex is peeled (its edge is on no cycle);
///   2. a degree-2 vertex is either emitted as a cycle (self-loop) or suppressed
///      by merging its two edges;
///   3. otherwise (min degree ≥ 3) a short cycle is emitted and one of its edges
///      is deleted uniformly at random.
fn decongestion_cycle_basis(n: usize, edges: &[Edge], rng: &mut impl Rng) -> Vec<Vec<Edge>> {
    let mut wedges: Vec<Option<WorkingEdge>> = Vec::with_capacity(edges.len());
    let mut incidence: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        dl_add_edge(&mut wedges, &mut incidence, a, b, vec![(a, b)]);
    }
    let mut remaining = edges.len();
    let mut basis: Vec<Vec<Edge>> = Vec::new();

    while remaining > 0 {
        // Case 1: peel a degree-1 vertex.
        if let Some(v) = (0..n).find(|&v| incidence[v].len() == 1) {
            let eid = incidence[v][0];
            dl_remove_edge(&mut wedges, &mut incidence, eid);
            remaining -= 1;
            continue;
        }
        // Case 2: a degree-2 vertex.
        if let Some(v) = (0..n).find(|&v| incidence[v].len() == 2) {
            let (e1, e2) = (incidence[v][0], incidence[v][1]);
            if e1 == e2 {
                // (a) self-loop: it is itself a cycle.
                basis.push(wedges[e1].as_ref().unwrap().2.clone());
                dl_remove_edge(&mut wedges, &mut incidence, e1);
            } else {
                // (b) suppress v: merge its two edges into one between the far
                // endpoints, concatenating their trails.
                let (x, mut trail) = {
                    let (a, b, t) = wedges[e1].as_ref().unwrap();
                    (if *a == v { *b } else { *a }, t.clone())
                };
                let y = {
                    let (a, b, t) = wedges[e2].as_ref().unwrap();
                    trail.extend(t.iter().copied());
                    if *a == v { *b } else { *a }
                };
                dl_remove_edge(&mut wedges, &mut incidence, e1);
                dl_remove_edge(&mut wedges, &mut incidence, e2);
                dl_add_edge(&mut wedges, &mut incidence, x, y, trail);
            }
            remaining -= 1; // net change (case 2b removes two, adds one)
            continue;
        }
        // Case 3: min degree ≥ 3 — emit a short cycle, drop a random edge of it.
        // A min-degree-≥3 graph has a cycle, and every vertex on it has degree ≥ 3,
        // so scanning the degree-≥3 vertices for the first that lies on a cycle
        // always finds one (not every degree-≥3 vertex is on a cycle — a bridge-only
        // cut vertex is not — so we cannot just take the first such vertex).
        let cycle = (0..n)
            .filter(|&v| incidence[v].len() >= 3)
            .find_map(|v| dl_shortest_cycle(&wedges, &incidence, v))
            .expect("edges remain at min degree ≥ 3 ⇒ the graph contains a cycle");
        let mut cycle_edges = Vec::new();
        for &eid in &cycle {
            cycle_edges.extend(wedges[eid].as_ref().unwrap().2.iter().copied());
        }
        basis.push(cycle_edges);
        let drop = cycle[rng.gen_range(0..cycle.len())];
        dl_remove_edge(&mut wedges, &mut incidence, drop);
        remaining -= 1;
    }

    basis
}

/// Result of [`congestion_aware_expander`]: the constructed graph together with
/// the congestion/expansion state at the balance point.
struct CongestionAwareExpander {
    graph: UnGraph<(), ()>,
    /// Decongestion layer count `t` (number of edge-disjoint cycle groups).
    decongestion_layers: usize,
    /// `λ₂(G)` at termination.
    lambda2: f64,
    /// The maintained cycle basis `R` (each cycle as its edge set).
    cycle_basis: Vec<Vec<Edge>>,
    /// The decongestion-layer partition; `partition.layer_cycles()[r]` feeds
    /// level `r` of the thickening step.
    partition: CyclePartition,
}

/// Layers needed to boost expansion at a given `λ₂`: `⌈2 / λ₂⌉`, or "unbounded"
/// (`usize::MAX`) while the graph is effectively disconnected (`λ₂ ≈ 0`).
fn expansion_layers_needed(lambda2: f64) -> usize {
    if lambda2 > 1e-12 {
        (2.0 / lambda2).ceil() as usize
    } else {
        usize::MAX
    }
}

/// Algorithm 4: Congestion-Aware Expander Construction.
///
/// Like Algorithm 1 ([`conditioned_expander_graph`]) it regularizes degrees
/// (Phase 1) then adds random matchings (Phase 2), but rather than stopping at
/// a fixed expansion threshold `2β`, it tracks the decongestion layer count `t`
/// (Algorithms 2 & 3) after every edge insertion and stops at the
/// expansion–congestion *balance point*: when `t ≥ ⌈2 / λ₂(G)⌉`. The decongestion
/// layers `t` increase while the expansion layers needed `⌈2/λ₂⌉` decrease as
/// edges are added, so the crossing is the optimal trade-off and no resources
/// are spent on an unbalanced construction.
///
/// `reset_period` rebuilds the BFS spanning forest every that-many steps (to
/// keep tree paths short); `max_iterations` bounds Phase 2.
///
/// `qubit_degree` is the LDPC qubit-degree bound `d_q` used by the degree-aware
/// partition (Algorithm 5): each layer may load an edge with up to `d_q` cycles.
/// Passing `d_q = 1` recovers the strictly edge-disjoint partition of
/// Algorithm 3; larger `d_q` packs more cycles per layer, reducing the layer
/// count `t` (and hence thickening) at the cost of higher qubit degree.
///
/// `R₀` is the low-congestion cycle basis from the Decongestion Lemma
/// (Freedman–Hastings, arXiv:2012.02249), partitioned with the same Algorithm 5
/// rule; subsequent cycles are tracked dynamically by Algorithm 2.
fn congestion_aware_expander(
    g0: &UnGraph<(), ()>,
    reset_period: usize,
    max_iterations: usize,
    qubit_degree: usize,
    rng: &mut impl Rng,
) -> CongestionAwareExpander {
    let mut g = g0.clone();
    let nodes: Vec<NodeIndex> = g.node_indices().collect();
    let n = nodes.len();
    // Node indices are stable under edge insertion, so this map is computed once.
    let pos: HashMap<NodeIndex, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();

    // T₀: BFS spanning forest.
    let mut forest = bfs_spanning_forest(&g, &nodes, &pos);

    // R₀ via the Decongestion Lemma (Freedman–Hastings), greedily partitioned
    // into decongestion layers. This is the low-congestion basis the paper
    // specifies, replacing the naive fundamental-cycle basis.
    let mut original_edges: Vec<Edge> = g
        .edge_indices()
        .map(|e| {
            let (a, b) = g.edge_endpoints(e).expect("edge has endpoints");
            canonical_edge(pos[&a], pos[&b])
        })
        .collect();
    original_edges.sort_unstable();
    original_edges.dedup();

    let mut partition = CyclePartition::with_max_load(qubit_degree);
    let mut cycle_basis: Vec<Vec<Edge>> = Vec::new();
    for cycle in decongestion_cycle_basis(n, &original_edges, rng) {
        partition.insert(&cycle);
        cycle_basis.push(cycle);
    }

    let mut step = 0usize;
    let mut t = partition.layers();
    let mut lambda2 = laplacian_lambda2(&g);

    if t >= expansion_layers_needed(lambda2) {
        return CongestionAwareExpander {
            graph: g,
            decongestion_layers: t,
            lambda2,
            cycle_basis,
            partition,
        };
    }

    // Phase 1: Regularization to maximum degree (cf. Algorithm 1), now also
    // tracking the cycle basis and decongestion layers after each insertion.
    let max_deg = nodes.iter().map(|&v| degree(&g, v)).max().unwrap_or(0);
    'phase1: loop {
        let deficit: Vec<usize> = nodes.iter().map(|&v| max_deg - degree(&g, v)).collect();
        let mut candidates: Vec<(usize, usize)> = Vec::new();
        let mut weights: Vec<usize> = Vec::new();
        for i in 0..n {
            if deficit[i] == 0 {
                continue;
            }
            for j in (i + 1)..n {
                if deficit[j] == 0 || g.find_edge(nodes[i], nodes[j]).is_some() {
                    continue;
                }
                candidates.push((i, j));
                weights.push(deficit[i] * deficit[j]);
            }
        }
        if candidates.is_empty() {
            break 'phase1;
        }

        let dist = WeightedIndex::new(&weights).expect("weights are positive");
        let (i, j) = candidates[dist.sample(rng)];
        g.add_edge(nodes[i], nodes[j], ());
        lambda2 = laplacian_lambda2(&g);
        if let Some(cycle) = dynamic_cycle_basis_update(&mut forest, i, j) {
            t = partition.insert(&cycle);
            cycle_basis.push(cycle);
        }
        if t >= expansion_layers_needed(lambda2) {
            return CongestionAwareExpander {
                graph: g,
                decongestion_layers: t,
                lambda2,
                cycle_basis,
                partition,
            };
        }
        step += 1;
        if reset_period > 0 && step % reset_period == 0 {
            forest = bfs_spanning_forest(&g, &nodes, &pos);
        }
    }

    // Phase 2: Degree augmentation via random matchings.
    for _ in 0..max_iterations {
        let mut unmatched: Vec<usize> = (0..n).collect();
        unmatched.shuffle(rng);
        let mut k = 0;
        while k + 1 < unmatched.len() {
            let (i, j) = (unmatched[k], unmatched[k + 1]);
            k += 2;
            if g.find_edge(nodes[i], nodes[j]).is_some() {
                continue;
            }
            g.add_edge(nodes[i], nodes[j], ());
            lambda2 = laplacian_lambda2(&g);
            if let Some(cycle) = dynamic_cycle_basis_update(&mut forest, i, j) {
                t = partition.insert(&cycle);
                cycle_basis.push(cycle);
            }
            if t >= expansion_layers_needed(lambda2) {
                return CongestionAwareExpander {
                    graph: g,
                    decongestion_layers: t,
                    lambda2,
                    cycle_basis,
                    partition,
                };
            }
            step += 1;
            if reset_period > 0 && step % reset_period == 0 {
                forest = bfs_spanning_forest(&g, &nodes, &pos);
            }
        }
    }

    CongestionAwareExpander {
        graph: g,
        decongestion_layers: t,
        lambda2,
        cycle_basis,
        partition,
    }
}

/// The thickened ancilla graph `G = G₁ □ Jℓ` (Step 4 of Algorithm 6).
struct ThickenedGraph {
    /// `G₁ □ Jℓ`: `levels` stacked copies of the base graph, consecutive copies
    /// joined by a path edge at every vertex. Vertex `(v, r)` has id
    /// `r * base_vertices + v`.
    graph: UnGraph<(), ()>,
    /// The thickening factor `ℓ`.
    levels: usize,
    /// `|V₁|`, the base graph's vertex count.
    base_vertices: usize,
    /// Cycles placed on each level (in thickened-vertex coordinates); level `r`
    /// receives the decongestion group `P_r`, and levels `≥ t` carry none.
    level_cycles: Vec<Vec<Vec<Edge>>>,
}

/// Thicken `result.graph` (Step 4 of Algorithm 6: `G = G₁ □ Jℓ`).
///
/// The thickening factor is `ℓ = max(t, ⌈2/λ₂⌉)`: the decongestion layers `t`
/// ensure each level's cycles are edge-disjoint, while `⌈2/λ₂⌉ ≈ 1/β_G` (since
/// `β_G ≥ λ₂/2`) is the factor that amplifies the relative Cheeger constant to
/// `≥ 1` so the code distance is preserved. At Algorithm 4's balance point these
/// two are about equal, so a single thickening serves both purposes.
///
/// Each decongestion group `P_r` is laid onto level `r`, where (being internally
/// edge-disjoint) every qubit participates in only `O(1)` cycle checks.
fn thicken(result: &CongestionAwareExpander) -> ThickenedGraph {
    let base_vertices = result.graph.node_count();
    let base_index: HashMap<NodeIndex, usize> = result
        .graph
        .node_indices()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    let base_edges: Vec<Edge> = result
        .graph
        .edge_indices()
        .map(|e| {
            let (a, b) = result.graph.edge_endpoints(e).expect("edge has endpoints");
            canonical_edge(base_index[&a], base_index[&b])
        })
        .collect();

    let layer_cycles = result.partition.layer_cycles();
    let t = layer_cycles.len();
    let levels = t
        .max(expansion_layers_needed(result.lambda2).min(base_vertices.max(1)))
        .max(1);

    let id = |v: usize, r: usize| r * base_vertices + v;
    let mut graph = UnGraph::new_undirected();
    let nodes: Vec<NodeIndex> = (0..levels * base_vertices)
        .map(|_| graph.add_node(()))
        .collect();
    // Horizontal edges: a copy of G₁ on every level.
    for r in 0..levels {
        for &(a, b) in &base_edges {
            graph.add_edge(nodes[id(a, r)], nodes[id(b, r)], ());
        }
    }
    // Vertical edges: the path Jℓ at every vertex.
    for v in 0..base_vertices {
        for r in 0..levels.saturating_sub(1) {
            graph.add_edge(nodes[id(v, r)], nodes[id(v, r + 1)], ());
        }
    }

    // Place each group P_r onto level r (in thickened coordinates).
    let mut level_cycles: Vec<Vec<Vec<Edge>>> = vec![Vec::new(); levels];
    for (r, group) in layer_cycles.iter().enumerate() {
        for cycle in group {
            let mapped: Vec<Edge> = cycle
                .iter()
                .map(|&(a, b)| canonical_edge(id(a, r), id(b, r)))
                .collect();
            level_cycles[r].push(mapped);
        }
    }

    ThickenedGraph {
        graph,
        levels,
        base_vertices,
        level_cycles,
    }
}

/// The result of cellulating the thickened graph (Step 5 of Algorithm 6).
struct Cellulation {
    /// The cellulated ancilla graph: the thickened graph with every cellulation
    /// chord added as an edge. Its edges are the qubits of the ancilla system.
    /// Vertex ids match the thickened graph (`r * base_vertices + v`).
    graph: UnGraph<(), ()>,
    /// Cycle checks (stabilizers): each is the set of qubit-edges on a face,
    /// every face having weight (qubit count) at most `d_c`.
    checks: Vec<Vec<Edge>>,
    /// Chord edges introduced by cellulation — the auxiliary qubits added to
    /// decompose long cycles into bounded-weight faces.
    chords: Vec<Edge>,
}

/// Recover the cyclic vertex order of a simple cycle given only its edge set.
/// Every vertex of a simple cycle has degree exactly two, so we walk the loop.
fn cycle_vertex_order(cycle: &[Edge]) -> Vec<usize> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in cycle {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    debug_assert!(
        adjacency.values().all(|nbrs| nbrs.len() == 2),
        "cellulation expects simple cycles (every vertex has degree 2)"
    );
    let start = cycle[0].0;
    let mut order = vec![start];
    let mut prev = usize::MAX;
    let mut cur = start;
    loop {
        let nbrs = &adjacency[&cur];
        let next = if nbrs[0] != prev { nbrs[0] } else { nbrs[1] };
        if next == start {
            break;
        }
        order.push(next);
        prev = cur;
        cur = next;
    }
    order
}

/// Zigzag `d_c`-gon cellulation of a single simple cycle given as an ordered
/// vertex loop `v₀ … v_{m-1}`. Returns the faces (each a set of edges, including
/// chords) of weight at most `d_c`.
///
/// Two pointers sweep inward from the cycle's ends, alternating sides so that
/// chords are spread across vertices (bounded degree) rather than fanned from
/// one vertex. Each step closes a face spanning `d_c` vertices using the current
/// "open" edge and a freshly added chord, which becomes the next face's open
/// edge. Because every interior chord borders exactly two faces, the symmetric
/// difference of all faces is the original cycle.
fn zigzag_faces(vertices: &[usize], d_c: usize) -> Vec<Vec<Edge>> {
    let d_c = d_c.max(3);
    let m = vertices.len();
    let mut faces: Vec<Vec<Edge>> = Vec::new();
    if m < 3 {
        return faces;
    }
    let boundary = |a: usize, b: usize| canonical_edge(vertices[a], vertices[b]);

    if m <= d_c {
        // The whole cycle already fits in one face.
        let mut face: Vec<Edge> = (0..m).map(|s| boundary(s, (s + 1) % m)).collect();
        face.dedup();
        faces.push(face);
        return faces;
    }

    let (mut i, mut j) = (0usize, m - 1);
    // The first open edge is the real cycle edge (v₀, v_{m-1}).
    let mut open = boundary(i, j);
    let mut low_turn = true;
    let step = d_c - 2; // boundary vertices consumed per face beyond the two ends
    while j - i + 1 > d_c {
        let mut face = vec![open];
        if low_turn {
            for s in i..(i + step) {
                face.push(boundary(s, s + 1));
            }
            open = boundary(i + step, j);
            face.push(open);
            i += step;
        } else {
            for s in 0..step {
                face.push(boundary(j - s, j - s - 1));
            }
            open = boundary(i, j - step);
            face.push(open);
            j -= step;
        }
        faces.push(face);
        low_turn = !low_turn;
    }
    // Final face: the remaining boundary path plus the last open edge.
    let mut face = vec![open];
    for s in i..j {
        face.push(boundary(s, s + 1));
    }
    faces.push(face);
    faces
}

/// Degree-aware cellulation (Step 5 of Algorithm 6, generalized to `d_c`-gons).
///
/// Each cycle placed on each level of the thickened graph is decomposed by
/// [`zigzag_faces`] into faces of weight at most `d_c` (the LDPC check-degree
/// bound). The faces become the cycle checks (stabilizers), and the chords they
/// introduce are added to a copy of the thickened graph, yielding the cellulated
/// ancilla graph (whose edges are the system's qubits). `d_c = 3` recovers the
/// standard triangular cellulation.
fn cellulate(thickened: &ThickenedGraph, max_check_degree: usize) -> Cellulation {
    let d_c = max_check_degree.max(3);
    let mut graph = thickened.graph.clone();
    let mut checks: Vec<Vec<Edge>> = Vec::new();
    let mut chords: Vec<Edge> = Vec::new();

    for level in &thickened.level_cycles {
        for cycle in level {
            let order = cycle_vertex_order(cycle);
            let cycle_edges: HashSet<Edge> = cycle.iter().copied().collect();
            for face in zigzag_faces(&order, d_c) {
                for &e in &face {
                    // A face edge that is neither on the original cycle nor
                    // already in the graph is a freshly added chord (qubit).
                    if !cycle_edges.contains(&e)
                        && graph
                            .find_edge(NodeIndex::new(e.0), NodeIndex::new(e.1))
                            .is_none()
                    {
                        graph.add_edge(NodeIndex::new(e.0), NodeIndex::new(e.1), ());
                        chords.push(e);
                    }
                }
                checks.push(face);
            }
        }
    }

    Cellulation {
        graph,
        checks,
        chords,
    }
}

/// Graphs up to this many vertices compute `λ₂` via a full dense symmetric
/// eigendecomposition; larger graphs switch to matrix-free Lanczos so the cost
/// scales with the number of edges rather than `n³`.
const DENSE_LAMBDA2_THRESHOLD: usize = 128;

/// Krylov subspace dimension for the Lanczos `λ₂` estimate on large graphs.
const LANCZOS_ITERS: usize = 100;

/// Second-smallest eigenvalue of the graph Laplacian `L = D - A` (the algebraic
/// connectivity / Fiedler value), used as the tractable proxy for the Cheeger
/// constant. Returns `0.0` for graphs with fewer than two vertices.
///
/// Small graphs are handled by [`lambda2_dense`]; large graphs use matrix-free
/// Lanczos ([`lambda2_lanczos`]) so a single `λ₂` evaluation costs
/// `O(edges · iters)` instead of `O(n³)` — important because Algorithm 1
/// recomputes `λ₂` after every edge addition.
fn laplacian_lambda2(g: &UnGraph<(), ()>) -> f64 {
    let nodes: Vec<NodeIndex> = g.node_indices().collect();
    let n = nodes.len();
    if n < 2 {
        return 0.0;
    }
    let (adjacency, degrees) = laplacian_adjacency(g, &nodes);
    if n <= DENSE_LAMBDA2_THRESHOLD {
        lambda2_dense(&adjacency, &degrees, n)
    } else {
        lambda2_lanczos(&adjacency, &degrees, n)
    }
}

/// Compressed neighbor-index adjacency and per-vertex degrees, in the `0..n`
/// index space used by the eigenvalue routines.
fn laplacian_adjacency<N>(g: &UnGraph<N, ()>, nodes: &[NodeIndex]) -> (Vec<Vec<usize>>, Vec<f64>) {
    let pos: HashMap<NodeIndex, usize> = nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, &v) in nodes.iter().enumerate() {
        for w in g.neighbors(v) {
            adjacency[i].push(pos[&w]);
        }
    }
    let degrees: Vec<f64> = adjacency.iter().map(|a| a.len() as f64).collect();
    (adjacency, degrees)
}

/// `λ₂` via a full dense symmetric eigendecomposition of the Laplacian.
fn lambda2_dense(adjacency: &[Vec<usize>], degrees: &[f64], n: usize) -> f64 {
    let mut lap = vec![0.0f64; n * n];
    for i in 0..n {
        lap[i * n + i] = degrees[i];
        for &j in &adjacency[i] {
            lap[i * n + j] -= 1.0;
        }
    }
    symmetric_eigenvalues(lap, n)[1]
}

/// `λ₂` via matrix-free Lanczos, so a single evaluation costs `O(edges · iters)`
/// instead of `O(n³)` — important because Algorithm 1/4 recompute `λ₂` after
/// every edge addition.
fn lambda2_lanczos(adjacency: &[Vec<usize>], degrees: &[f64], n: usize) -> f64 {
    // The zero eigenvalue of `L` has multiplicity equal to the number of
    // connected components, so `λ₂ = 0` exactly iff the graph is disconnected.
    // Single-vector Lanczos cannot resolve that multiplicity, so detect it
    // directly and cheaply (this also covers the edge-free graph).
    if connected_components(n, adjacency) >= 2 {
        return 0.0;
    }

    // Connected ⇒ the only zero mode of `L` is the all-ones vector. Shift the
    // Laplacian into `M = c·I − L` (eigenvalues `c − λ_i(L)`), deflate the
    // all-ones direction, and find the largest remaining eigenvalue: it is
    // exactly `c − λ₂`, an *extremal* eigenvalue that Lanczos resolves robustly.
    // Gershgorin gives `λ_max(L) ≤ 2·max_v deg(v)`, so `c ≥ λ_max(L)` always.
    let shift = 2.0 * degrees.iter().copied().fold(0.0, f64::max);
    let mu_max = lanczos_largest_deflating_ones(n, adjacency, degrees, shift);
    shift - mu_max
}

/// Number of connected components of the graph given by `adjacency`, via
/// union-find with path halving.
fn connected_components(n: usize, adjacency: &[Vec<usize>]) -> usize {
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        for &j in &adjacency[i] {
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri != rj {
                parent[ri] = rj;
            }
        }
    }
    (0..n).filter(|&i| find(&mut parent, i) == i).count()
}

/// Estimate the largest eigenvalue of `M = shift·I − L` *after deflating the
/// all-ones direction*, via Lanczos with full reorthogonalization. For a
/// connected graph this equals `shift − λ₂`. The matrix is never materialized:
/// the only operation needed is the product `M·x`, computed directly from the
/// adjacency structure, and the all-ones component is removed at each step by
/// subtracting the mean.
fn lanczos_largest_deflating_ones(
    n: usize,
    adjacency: &[Vec<usize>],
    degrees: &[f64],
    shift: f64,
) -> f64 {
    // (M·x)_i = (shift − deg_i)·x_i + Σ_{j ~ i} x_j
    let matvec = |x: &[f64]| -> Vec<f64> {
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut acc = (shift - degrees[i]) * x[i];
            for &j in &adjacency[i] {
                acc += x[j];
            }
            y[i] = acc;
        }
        y
    };

    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    // Remove the all-ones component (the λ=0 eigenvector of a connected graph):
    // projecting onto the complement of the constant vector is just mean removal.
    let deflate = |w: &mut [f64]| {
        let mean = w.iter().sum::<f64>() / n as f64;
        for x in w.iter_mut() {
            *x -= mean;
        }
    };

    // Deterministic, non-constant start vector, deflated and normalized.
    let mut rng = StdRng::seed_from_u64(0x5EED_C0DE);
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect();
    deflate(&mut v);
    let norm0 = dot(&v, &v).sqrt();
    for x in &mut v {
        *x /= norm0;
    }

    let m = n.min(LANCZOS_ITERS);
    let mut basis: Vec<Vec<f64>> = Vec::with_capacity(m);
    let mut alphas: Vec<f64> = Vec::with_capacity(m);
    let mut betas: Vec<f64> = Vec::with_capacity(m);
    let mut prev_beta = 0.0f64;
    let mut prev_v = vec![0.0f64; n];

    let mut ritz = 0.0f64;
    let mut prev_ritz = f64::NEG_INFINITY;
    for _ in 0..m {
        let mut w = matvec(&v);
        for i in 0..n {
            w[i] -= prev_beta * prev_v[i];
        }
        let alpha = dot(&w, &v);
        for i in 0..n {
            w[i] -= alpha * v[i];
        }
        // Full reorthogonalization (twice) against the accumulated basis to
        // counter the loss of orthogonality that plagues plain Lanczos, plus
        // continual deflation of the all-ones direction so roundoff cannot
        // reintroduce the λ=0 mode.
        for _ in 0..2 {
            deflate(&mut w);
            for q in &basis {
                let proj = dot(&w, q);
                for i in 0..n {
                    w[i] -= proj * q[i];
                }
            }
        }

        basis.push(v.clone());
        alphas.push(alpha);

        // Largest Ritz value of the order-k tridiagonal accumulated so far
        // (αs on the diagonal, the βs from prior iterations off it). Only this
        // single extremal value is needed, so it comes from Sturm-bisection
        // rather than a full eigendecomposition. At this point `betas` holds
        // exactly the k−1 off-diagonals of the order-k matrix.
        ritz = tridiagonal_largest_eigenvalue(&alphas, &betas);

        let beta = dot(&w, &w).sqrt();
        if beta < 1e-10 {
            break; // exact invariant subspace reached
        }
        // The extremal Ritz value converges long before the iteration cap for a
        // well-separated top eigenvalue; stop once it has stabilized.
        if (ritz - prev_ritz).abs() <= 1e-10 * (1.0 + ritz.abs()) {
            break;
        }
        prev_ritz = ritz;

        betas.push(beta);
        prev_beta = beta;
        prev_v = v;
        v = w;
        for x in &mut v {
            *x /= beta;
        }
    }

    // With the all-ones mode deflated, the largest Ritz value approximates c − λ₂.
    ritz
}

/// Largest eigenvalue of the symmetric tridiagonal matrix with diagonal `diag`
/// and off-diagonal `offdiag` (`offdiag[i]` couples rows `i` and `i+1`), via
/// Sturm-sequence bisection. The Lanczos driver needs only this one extremal
/// Ritz value, and bisection costs `O(k)` per step against the `O(k³)` of a
/// full eigendecomposition that would discard all but the top eigenvalue.
fn tridiagonal_largest_eigenvalue(diag: &[f64], offdiag: &[f64]) -> f64 {
    let k = diag.len();
    if k == 0 {
        return 0.0;
    }
    if k == 1 {
        return diag[0];
    }

    // Number of eigenvalues strictly below `x`, read off the signs of the LDLᵀ
    // pivots of `T − xI` (a Sturm sequence): the count of negative pivots.
    let count_below = |x: f64| -> usize {
        let mut count = 0usize;
        let mut q = diag[0] - x;
        if q < 0.0 {
            count += 1;
        }
        for i in 1..k {
            // Guard a zero pivot so the recurrence can't divide by zero.
            let denom = if q.abs() < 1e-300 { 1e-300 } else { q };
            q = (diag[i] - x) - offdiag[i - 1] * offdiag[i - 1] / denom;
            if q < 0.0 {
                count += 1;
            }
        }
        count
    };

    // Gershgorin bracket: every eigenvalue lies in some disk [d_i − r_i, d_i + r_i]
    // with r_i the sum of the adjacent off-diagonal magnitudes.
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for i in 0..k {
        let left = if i > 0 { offdiag[i - 1].abs() } else { 0.0 };
        let right = if i + 1 < k { offdiag[i].abs() } else { 0.0 };
        lo = lo.min(diag[i] - left - right);
        hi = hi.max(diag[i] + left + right);
    }
    // Widen so the largest eigenvalue is strictly interior to the bracket.
    let pad = (1.0 + hi.abs()) * 1e-9;
    lo -= pad;
    hi += pad;

    // Bisect for the boundary where `count_below` first reaches k — the largest
    // eigenvalue. `O(k)` per step, ~60 steps to roundoff-level precision.
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if count_below(mid) >= k {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo <= 1e-12 * (1.0 + hi.abs()) {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Eigenvalues of a real symmetric `n × n` matrix (row-major), sorted ascending,
/// via the cyclic Jacobi rotation method.
fn symmetric_eigenvalues(mut a: Vec<f64>, n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        let off: f64 = (0..n)
            .flat_map(|p| ((p + 1)..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p * n + q] * a[p * n + q])
            .sum();
        if off < 1e-12 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Right-multiply by the rotation: update columns p and q.
                for i in 0..n {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip - s * aiq;
                    a[i * n + q] = s * aip + c * aiq;
                }
                // Left-multiply by its transpose: update rows p and q.
                for j in 0..n {
                    let apj = a[p * n + j];
                    let aqj = a[q * n + j];
                    a[p * n + j] = c * apj - s * aqj;
                    a[q * n + j] = s * apj + c * aqj;
                }
            }
        }
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigenvalues.sort_by(|x, y| x.partial_cmp(y).expect("no NaN eigenvalues"));
    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::PauliString;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn graph_from_edges(n: usize, edges: &[(usize, usize)]) -> UnGraph<(), ()> {
        let mut g = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
        for &(i, j) in edges {
            g.add_edge(nodes[i], nodes[j], ());
        }
        g
    }

    /// `graph_from_edges` lifted to the labeled node-weight type a `SurgeryGraph`
    /// carries, with `port_vertices` labeled as code qubits of `block` (each port's
    /// own id used as its within-block index — the bridge tests don't read it). The
    /// `Some`-weighted vertices are exactly what `ports_by_block` recovers as ports.
    fn surgery_graph_with_ports(
        n: usize,
        edges: &[(usize, usize)],
        block: &'static str,
        port_vertices: &[usize],
    ) -> UnGraph<Option<CodeQubit<&'static str>>, ()> {
        let mut g: UnGraph<Option<CodeQubit<&'static str>>, ()> =
            graph_from_edges(n, edges).map(|_, _| None, |_, _| ());
        for &p in port_vertices {
            g[NodeIndex::new(p)] = Some(CodeQubit { block, index: p });
        }
        g
    }

    #[test]
    fn lambda2_of_path_p3() {
        // Path 0-1-2 has Laplacian eigenvalues {0, 1, 3}; λ₂ = 1.
        let g = graph_from_edges(3, &[(0, 1), (1, 2)]);
        assert!((laplacian_lambda2(&g) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn lambda2_of_complete_k3() {
        // K3 has Laplacian eigenvalues {0, 3, 3}; λ₂ = 3.
        let g = graph_from_edges(3, &[(0, 1), (1, 2), (0, 2)]);
        assert!((laplacian_lambda2(&g) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn lambda2_of_disconnected_is_zero() {
        // Two components ⇒ algebraic connectivity is 0.
        let g = graph_from_edges(4, &[(0, 1), (2, 3)]);
        assert!(laplacian_lambda2(&g).abs() < 1e-9);
    }

    #[test]
    fn expander_construction_reaches_threshold() {
        // Path P6 has λ₂ ≈ 0.268, below the 2β = 0.6 target, so augmentation runs.
        let g0 = graph_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let beta = 0.3;
        let mut rng = StdRng::seed_from_u64(42);
        let g = conditioned_expander_graph(&g0, beta, 50, &mut rng);

        assert!(laplacian_lambda2(&g) >= 2.0 * beta);
        // Existing structure is preserved (edges are only added, never removed).
        assert!(g.edge_count() >= g0.edge_count());
    }

    #[test]
    fn build_keeps_a_graph_meeting_threshold() {
        let g0 = graph_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        let beta = 0.3;
        let mut rng = StdRng::seed_from_u64(7);
        let g = build_congestion_aware_expander(&g0, &SurgeryGraphConfig::default(), &mut rng);
        assert!(laplacian_lambda2(&g.graph) >= 2.0 * beta);
    }

    fn complete_graph(n: usize) -> UnGraph<(), ()> {
        let mut g = UnGraph::new_undirected();
        let nodes: Vec<_> = (0..n).map(|_| g.add_node(())).collect();
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(nodes[i], nodes[j], ());
            }
        }
        g
    }

    fn star_graph(leaves: usize) -> UnGraph<(), ()> {
        let mut g = UnGraph::new_undirected();
        let center = g.add_node(());
        for _ in 0..leaves {
            let leaf = g.add_node(());
            g.add_edge(center, leaf, ());
        }
        g
    }

    #[test]
    fn lanczos_path_matches_complete_graph_spectrum() {
        // K_n (n > DENSE_LAMBDA2_THRESHOLD ⇒ Lanczos path) has λ₂ = n exactly.
        let n = 300;
        assert!(n > DENSE_LAMBDA2_THRESHOLD);
        let g = complete_graph(n);
        assert!((laplacian_lambda2(&g) - n as f64).abs() < 1e-6);
    }

    #[test]
    fn lanczos_path_matches_star_graph_spectrum() {
        // Star K_{1,m} has Laplacian eigenvalues {0, 1 (×m-1), m+1}; λ₂ = 1.
        let leaves = 400; // n = 401 ⇒ Lanczos path
        let g = star_graph(leaves);
        assert!(g.node_count() > DENSE_LAMBDA2_THRESHOLD);
        assert!((laplacian_lambda2(&g) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn forest_path_and_dynamic_cycle() {
        // Triangle 0-1-2 with forest edges {0-1, 1-2}; adding 0-2 closes a cycle.
        let mut forest = SpanningForest::new(3);
        forest.add_edge(0, 1);
        forest.add_edge(1, 2);
        assert_eq!(forest.path(0, 2), Some(vec![0, 1, 2]));

        let cycle = dynamic_cycle_basis_update(&mut forest, 0, 2).expect("closes a cycle");
        let edges: HashSet<Edge> = cycle.into_iter().collect();
        assert_eq!(
            edges,
            HashSet::from([(0, 1), (1, 2), (0, 2)]),
            "fundamental cycle is the path plus the closing edge"
        );
        // A bridge between separate trees forms no cycle.
        let mut forest2 = SpanningForest::new(4);
        forest2.add_edge(0, 1);
        forest2.add_edge(2, 3);
        assert!(dynamic_cycle_basis_update(&mut forest2, 1, 2).is_none());
    }

    #[test]
    fn greedy_partition_layers_count_overlap() {
        // d_q = 1 reproduces Algorithm 3 (strictly edge-disjoint layers).
        // Edge-disjoint cycles share no layer pressure ⇒ 1 layer.
        let mut p = CyclePartition::with_max_load(1);
        p.insert(&[(0, 1), (1, 2), (0, 2)]);
        assert_eq!(p.insert(&[(3, 4), (4, 5), (3, 5)]), 1);

        // A cycle overlapping the first layer needs a second layer.
        let mut q = CyclePartition::with_max_load(1);
        q.insert(&[(0, 1), (1, 2), (0, 2)]);
        assert_eq!(q.insert(&[(0, 1), (1, 3), (0, 3)]), 2);
    }

    #[test]
    fn degree_constrained_partition_packs_under_load_bound() {
        // With d_q = 2, two cycles sharing edge (0,1) can coexist in one layer
        // (load on (0,1) reaches 2), but a third pushes it over ⇒ new layer.
        let mut p = CyclePartition::with_max_load(2);
        assert_eq!(p.insert(&[(0, 1), (1, 2), (0, 2)]), 1);
        assert_eq!(p.insert(&[(0, 1), (1, 3), (0, 3)]), 1); // (0,1) load now 2
        assert_eq!(p.insert(&[(0, 1), (1, 4), (0, 4)]), 2); // (0,1) would hit 3

        // Larger d_q never needs more layers than smaller d_q on the same input.
        let cycles = [
            vec![(0, 1), (1, 2), (0, 2)],
            vec![(0, 1), (1, 3), (0, 3)],
            vec![(0, 1), (1, 4), (0, 4)],
            vec![(2, 3), (3, 4), (2, 4)],
        ];
        let layers = |d_q: usize| {
            let mut part = CyclePartition::with_max_load(d_q);
            let mut t = 0;
            for c in &cycles {
                t = part.insert(c);
            }
            t
        };
        assert!(layers(3) <= layers(2));
        assert!(layers(2) <= layers(1));
    }

    /// Cycle rank |E| − |V| + (#components) of a simple edge set on `n` vertices.
    fn cycle_rank(n: usize, edges: &[Edge]) -> usize {
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let mut touched = vec![false; n];
        for &(a, b) in edges {
            touched[a] = true;
            touched[b] = true;
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let components = (0..n)
            .filter(|&v| touched[v] && find(&mut parent, v) == v)
            .count();
        edges.len() + components - (0..n).filter(|&v| touched[v]).count()
    }

    /// Every vertex must touch an even number of a cycle's edges (it is a cycle).
    fn is_closed_chain(cycle: &[Edge]) -> bool {
        let mut deg: HashMap<usize, usize> = HashMap::new();
        for &(a, b) in cycle {
            *deg.entry(a).or_default() += 1;
            *deg.entry(b).or_default() += 1;
        }
        deg.values().all(|&d| d % 2 == 0)
    }

    #[test]
    fn decongestion_basis_is_valid_for_triangle() {
        let edges = vec![(0, 1), (1, 2), (0, 2)];
        let mut rng = StdRng::seed_from_u64(1);
        let basis = decongestion_cycle_basis(3, &edges, &mut rng);
        assert_eq!(basis.len(), 1);
        let cycle: HashSet<Edge> = basis[0].iter().copied().collect();
        assert_eq!(cycle, HashSet::from([(0, 1), (1, 2), (0, 2)]));
    }

    #[test]
    fn decongestion_basis_has_correct_rank_and_closed_cycles() {
        // A graph with several independent cycles across two components.
        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2), // component A: 4-cycle plus a chord ⇒ rank 2
            (4, 5),
            (5, 6),
            (6, 4), // component B: triangle ⇒ rank 1
            (6, 7), // pendant edge on component B (not in any cycle)
        ];
        let n = 8;
        for seed in 0..16 {
            let mut rng = StdRng::seed_from_u64(seed);
            let basis = decongestion_cycle_basis(n, &edges, &mut rng);
            assert_eq!(
                basis.len(),
                cycle_rank(n, &edges),
                "basis size must equal the cycle rank"
            );
            for cycle in &basis {
                assert!(!cycle.is_empty());
                assert!(is_closed_chain(cycle), "each basis element is a cycle");
                // Cycles use real graph edges only.
                for e in cycle {
                    assert!(edges.contains(e), "cycle edge {:?} exists in G", e);
                }
            }
        }
    }

    #[test]
    fn decongestion_basis_keeps_congestion_low() {
        // On a random sparse graph, no edge should appear in too many cycles.
        let n = 60;
        let mut rng = StdRng::seed_from_u64(99);
        let mut edge_set: HashSet<Edge> = HashSet::new();
        // A connecting cycle through all vertices, plus random extra chords.
        for i in 0..n {
            edge_set.insert(canonical_edge(i, (i + 1) % n));
        }
        for _ in 0..40 {
            let a = rng.gen_range(0..n);
            let b = rng.gen_range(0..n);
            if a != b {
                edge_set.insert(canonical_edge(a, b));
            }
        }
        let edges: Vec<Edge> = edge_set.into_iter().collect();
        let basis = decongestion_cycle_basis(n, &edges, &mut rng);
        assert_eq!(basis.len(), cycle_rank(n, &edges));

        let mut appearances: HashMap<Edge, usize> = HashMap::new();
        for cycle in &basis {
            for &e in cycle {
                *appearances.entry(e).or_default() += 1;
            }
        }
        let max_congestion = appearances.values().copied().max().unwrap_or(0);
        // O(log² V) ≈ 34 for V=60; allow generous slack but catch blow-ups.
        assert!(
            max_congestion <= 20,
            "max congestion {max_congestion} unexpectedly high"
        );
    }

    #[test]
    fn congestion_aware_construction_balances_and_terminates() {
        // Start from a sparse path; Algorithm 4 should add edges and stop at the
        // expansion–congestion balance point with a connected, valid result.
        let g0 = graph_from_edges(8, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]);
        let mut rng = StdRng::seed_from_u64(123);
        // d_q = 1: Algorithm 3 partitioning (strictly edge-disjoint layers).
        let result = congestion_aware_expander(&g0, 4, 50, 1, &mut rng);

        assert!(result.graph.edge_count() >= g0.edge_count());
        // Terminated at the balance point: layers reached the expansion budget.
        assert!(result.decongestion_layers >= expansion_layers_needed(result.lambda2));
        // A meaningful expander has positive algebraic connectivity.
        assert!(result.lambda2 > 0.0);
        // The cycle basis has the right cardinality: |E| − |V| + #components,
        // and the graph is connected here, so |E| − |V| + 1.
        let expected_cycles =
            result.graph.edge_count() as isize - result.graph.node_count() as isize + 1;
        assert_eq!(result.cycle_basis.len() as isize, expected_cycles);
    }

    #[test]
    fn lanczos_path_detects_disconnected_large_graph() {
        // Two disjoint complete graphs ⇒ λ₂ = 0 even on the Lanczos path.
        let mut g = complete_graph(150);
        let mut h = complete_graph(150);
        // Splice h into g as a second component.
        let offset = g.node_count();
        for _ in 0..h.node_count() {
            g.add_node(());
        }
        for e in h.raw_edges() {
            g.add_edge(
                NodeIndex::new(offset + e.source().index()),
                NodeIndex::new(offset + e.target().index()),
                (),
            );
        }
        let _ = &mut h;
        assert!(g.node_count() > DENSE_LAMBDA2_THRESHOLD);
        assert!(laplacian_lambda2(&g).abs() < 1e-6);
    }

    /// XOR (symmetric difference over GF(2)) of a collection of edge sets.
    fn xor_edges<'a>(faces: impl IntoIterator<Item = &'a Vec<Edge>>) -> HashSet<Edge> {
        let mut acc: HashSet<Edge> = HashSet::new();
        for face in faces {
            for &e in face {
                if !acc.remove(&e) {
                    acc.insert(e);
                }
            }
        }
        acc
    }

    #[test]
    fn zigzag_faces_partition_a_cycle() {
        // A 9-cycle cellulated into triangles (d_c = 3): faces tile the disk, so
        // their XOR is exactly the original cycle and each face has ≤ 3 edges.
        let verts: Vec<usize> = (0..9).collect();
        let cycle_edges: Vec<Edge> = (0..9).map(|i| canonical_edge(i, (i + 1) % 9)).collect();
        for d_c in [3usize, 4, 5] {
            let faces = zigzag_faces(&verts, d_c);
            for f in &faces {
                assert!(
                    f.len() <= d_c,
                    "face weight {} exceeds d_c {}",
                    f.len(),
                    d_c
                );
                assert!(f.len() >= 3);
            }
            let boundary = xor_edges(&faces);
            let expected: HashSet<Edge> = cycle_edges.iter().copied().collect();
            assert_eq!(
                boundary, expected,
                "faces must generate the cycle (d_c={d_c})"
            );
        }
    }

    #[test]
    fn thicken_has_product_structure() {
        // Build a small expander, thicken it, and check G₁ □ Jℓ dimensions.
        let g0 = graph_from_edges(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]);
        let mut rng = StdRng::seed_from_u64(5);
        let result = congestion_aware_expander(&g0, 4, 50, 1, &mut rng);
        let thick = thicken(&result);

        let n1 = result.graph.node_count();
        let e1 = result.graph.edge_count();
        let l = thick.levels;
        assert!(l >= 1);
        assert_eq!(thick.graph.node_count(), l * n1);
        // ℓ copies of G₁ plus (ℓ−1) vertical path edges per vertex.
        assert_eq!(thick.graph.edge_count(), l * e1 + (l - 1) * n1);
        // Level r holds group P_r; total placed cycles = the whole basis.
        let placed: usize = thick.level_cycles.iter().map(|c| c.len()).sum();
        assert_eq!(placed, result.cycle_basis.len());
    }

    #[test]
    fn cellulate_respects_check_degree_and_generates_cycles() {
        let g0 = graph_from_edges(8, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]);
        let mut rng = StdRng::seed_from_u64(321);
        let result = congestion_aware_expander(&g0, 4, 50, 1, &mut rng);
        let thick = thicken(&result);

        for d_c in [3usize, 4, 5, 6] {
            let cell = cellulate(&thick, d_c);
            // Every cycle check has weight at most d_c.
            for check in &cell.checks {
                assert!(
                    check.len() <= d_c,
                    "check weight {} > d_c {}",
                    check.len(),
                    d_c
                );
            }
            // The faces of each placed cycle regenerate that cycle. Group checks
            // back by which cycle they came from via XOR over all faces per
            // level-cycle: reconstruct and compare against the placed cycles.
            let mut all_placed: Vec<Vec<Edge>> = Vec::new();
            for level in &thick.level_cycles {
                all_placed.extend(level.iter().cloned());
            }
            // Cellulate each cycle individually to check the generation property.
            for cycle in &all_placed {
                let order = cycle_vertex_order(cycle);
                let faces = zigzag_faces(&order, d_c);
                let boundary = xor_edges(&faces);
                let expected: HashSet<Edge> = cycle.iter().copied().collect();
                assert_eq!(boundary, expected);
            }
            // Sanity: there is at least one check whenever there are cycles.
            if !all_placed.is_empty() {
                assert!(!cell.checks.is_empty());
            }

            // The returned graph is the thickened graph plus the chord edges:
            // same vertices, and exactly `chords.len()` extra edges.
            assert_eq!(cell.graph.node_count(), thick.graph.node_count());
            assert_eq!(
                cell.graph.edge_count(),
                thick.graph.edge_count() + cell.chords.len()
            );
            // Every chord and every check edge is present in the cellulated graph.
            for &(a, b) in &cell.chords {
                assert!(
                    cell.graph
                        .find_edge(NodeIndex::new(a), NodeIndex::new(b))
                        .is_some()
                );
            }
            for check in &cell.checks {
                for &(a, b) in check {
                    assert!(
                        cell.graph
                            .find_edge(NodeIndex::new(a), NodeIndex::new(b))
                            .is_some()
                    );
                }
            }
        }
    }

    // ----- Independent cross-checks -----------------------------------------

    /// Build a connected random graph on `n` vertices (a spanning cycle plus
    /// `extra` random chords), as a petgraph `UnGraph`.
    fn random_connected_graph(n: usize, extra: usize, rng: &mut StdRng) -> UnGraph<(), ()> {
        let mut edges: HashSet<Edge> = HashSet::new();
        for i in 0..n {
            edges.insert(canonical_edge(i, (i + 1) % n));
        }
        for _ in 0..extra {
            let (a, b) = (rng.gen_range(0..n), rng.gen_range(0..n));
            if a != b {
                edges.insert(canonical_edge(a, b));
            }
        }
        let edge_vec: Vec<Edge> = edges.into_iter().collect();
        graph_from_edges(n, &edge_vec)
    }

    /// Two cliques `K_s` joined by a single edge — a barbell, whose `λ₂` is tiny.
    fn barbell_graph(s: usize) -> UnGraph<(), ()> {
        let mut edges: Vec<Edge> = Vec::new();
        for i in 0..s {
            for j in (i + 1)..s {
                edges.push((i, j));
                edges.push((s + i, s + j));
            }
        }
        edges.push((0, s)); // the single bridge
        graph_from_edges(2 * s, &edges)
    }

    #[test]
    fn lanczos_agrees_with_dense_lambda2() {
        // The two λ₂ implementations are independent oracles; on the same graph
        // they must agree. We call each path directly (bypassing the size-based
        // dispatch) so both run on identical inputs.
        let lambda2_both = |g: &UnGraph<(), ()>| -> (f64, f64) {
            let nodes: Vec<NodeIndex> = g.node_indices().collect();
            let (adj, deg) = laplacian_adjacency(g, &nodes);
            (
                lambda2_dense(&adj, &deg, nodes.len()),
                lambda2_lanczos(&adj, &deg, nodes.len()),
            )
        };

        let mut rng = StdRng::seed_from_u64(2024);
        // Several random graphs of varying density.
        for &(n, extra) in &[(150usize, 50usize), (200, 300), (180, 20)] {
            let g = random_connected_graph(n, extra, &mut rng);
            let (dense, lanczos) = lambda2_both(&g);
            assert!(
                (dense - lanczos).abs() <= 1e-6 * dense.max(1.0) + 1e-9,
                "λ₂ mismatch (n={n}, extra={extra}): dense={dense}, lanczos={lanczos}"
            );
        }

        // Small-spectral-gap stress: a 200-cycle (λ₂ ≈ 9.87e-4, near-degenerate).
        let cycle = graph_from_edges(
            200,
            &(0..200)
                .map(|i| canonical_edge(i, (i + 1) % 200))
                .collect::<Vec<_>>(),
        );
        let (dense, lanczos) = lambda2_both(&cycle);
        assert!(
            (dense - lanczos).abs() <= 1e-6,
            "small-gap λ₂ mismatch on C200: dense={dense}, lanczos={lanczos}"
        );

        // Barbell: tiny λ₂ but well separated from λ₃, so both must agree closely.
        let bell = barbell_graph(75);
        let (dense, lanczos) = lambda2_both(&bell);
        assert!(
            (dense - lanczos).abs() <= 1e-6 * dense.max(1.0) + 1e-9,
            "barbell λ₂ mismatch: dense={dense}, lanczos={lanczos}"
        );
    }

    /// GF(2) rank of a set of edge-indexed bit-rows via Gaussian elimination.
    fn gf2_rank(mut rows: Vec<HashSet<usize>>) -> usize {
        let mut rank = 0;
        let mut pivots: Vec<HashSet<usize>> = Vec::new();
        for row in &mut rows {
            // Reduce against existing pivots (each keyed by its minimum element).
            loop {
                let Some(&lead) = row.iter().min() else { break };
                if let Some(p) = pivots.iter().find(|p| p.iter().min() == Some(&lead)) {
                    for &e in p {
                        if !row.remove(&e) {
                            row.insert(e);
                        }
                    }
                } else {
                    break;
                }
            }
            if !row.is_empty() {
                pivots.push(row.clone());
                rank += 1;
            }
        }
        rank
    }

    #[test]
    fn decongestion_basis_is_linearly_independent() {
        // A right-sized set of cycles is only a *basis* if it is also GF(2)
        // independent; verify rank == count (≠ just count == cycle rank).
        let n = 50;
        let mut rng = StdRng::seed_from_u64(7);
        for trial in 0..12 {
            let g = random_connected_graph(n, 30, &mut rng);
            let edges: Vec<Edge> = {
                let nodes: Vec<NodeIndex> = g.node_indices().collect();
                let pos: HashMap<NodeIndex, usize> =
                    nodes.iter().enumerate().map(|(i, &v)| (v, i)).collect();
                let mut es: Vec<Edge> = g
                    .edge_indices()
                    .map(|e| {
                        let (a, b) = g.edge_endpoints(e).unwrap();
                        canonical_edge(pos[&a], pos[&b])
                    })
                    .collect();
                es.sort_unstable();
                es.dedup();
                es
            };
            let basis = decongestion_cycle_basis(n, &edges, &mut rng);

            // Map each edge to a column index, each cycle to a GF(2) row.
            let edge_index: HashMap<Edge, usize> =
                edges.iter().enumerate().map(|(i, &e)| (e, i)).collect();
            let rows: Vec<HashSet<usize>> = basis
                .iter()
                .map(|cycle| cycle.iter().map(|e| edge_index[e]).collect())
                .collect();

            let rank = gf2_rank(rows);
            assert_eq!(
                rank,
                basis.len(),
                "trial {trial}: cycle basis is not GF(2) independent"
            );
            assert_eq!(
                basis.len(),
                cycle_rank(n, &edges),
                "trial {trial}: basis size must equal the cycle rank"
            );
        }
    }

    // ----- End-to-end pipeline ----------------------------------------------

    /// A simple cycle has every vertex at degree exactly two.
    fn is_simple_cycle(cycle: &[Edge]) -> bool {
        let mut deg: HashMap<usize, usize> = HashMap::new();
        for &(a, b) in cycle {
            *deg.entry(a).or_default() += 1;
            *deg.entry(b).or_default() += 1;
        }
        !cycle.is_empty() && deg.values().all(|&d| d == 2)
    }

    fn graph_is_connected<N>(g: &UnGraph<N, ()>) -> bool {
        let nodes: Vec<NodeIndex> = g.node_indices().collect();
        if nodes.is_empty() {
            return true;
        }
        let (adj, _) = laplacian_adjacency(g, &nodes);
        connected_components(nodes.len(), &adj) == 1
    }

    #[test]
    fn end_to_end_pipeline_invariants() {
        // expander → thicken → cellulate, asserting cross-stage invariants over
        // several seeds and starting graphs.
        for seed in 0..6u64 {
            let mut rng = StdRng::seed_from_u64(seed);
            // A sparse connected starting graph (path) on 10 vertices.
            let g0 = graph_from_edges(10, &(0..9).map(|i| (i, i + 1)).collect::<Vec<_>>());
            let d_q = 1 + (seed as usize % 2); // exercise d_q = 1 and 2
            let result = congestion_aware_expander(&g0, 4, 80, d_q, &mut rng);

            // Stage 1: a real expander — connected, positive algebraic connectivity.
            assert!(result.lambda2 > 0.0, "seed {seed}: expander not connected");
            assert!(graph_is_connected(&result.graph));
            // Every maintained cycle is a *simple* cycle (cellulation depends on it).
            for cycle in &result.cycle_basis {
                assert!(
                    is_simple_cycle(cycle),
                    "seed {seed}: non-simple cycle in basis"
                );
            }
            // Cycle basis cardinality is the cycle rank (connected ⇒ E − V + 1).
            let expected = result.graph.edge_count() + 1 - result.graph.node_count();
            assert_eq!(result.cycle_basis.len(), expected);

            // Stage 2: thickening preserves the product structure.
            let thick = thicken(&result);
            let (n1, e1, l) = (
                result.graph.node_count(),
                result.graph.edge_count(),
                thick.levels,
            );
            assert_eq!(thick.graph.node_count(), l * n1);
            assert_eq!(thick.graph.edge_count(), l * e1 + (l - 1) * n1);
            // Placed cycles remain simple in thickened coordinates.
            for level in &thick.level_cycles {
                for cycle in level {
                    assert!(is_simple_cycle(cycle));
                }
            }

            // Stage 3: cellulation yields a connected ancilla graph whose checks
            // are bounded-weight and live on real edges, and regenerate cycles.
            let d_c = 3 + (seed as usize % 3); // d_c ∈ {3,4,5}
            let cell = cellulate(&thick, d_c);
            assert!(
                graph_is_connected(&cell.graph),
                "seed {seed}: cellulated graph disconnected"
            );
            assert_eq!(
                cell.graph.edge_count(),
                thick.graph.edge_count() + cell.chords.len()
            );
            for check in &cell.checks {
                assert!(check.len() <= d_c);
                for &(a, b) in check {
                    assert!(
                        cell.graph
                            .find_edge(NodeIndex::new(a), NodeIndex::new(b))
                            .is_some()
                    );
                }
            }
            // The faces covering each placed cycle XOR back to exactly that cycle.
            for level in &thick.level_cycles {
                for cycle in level {
                    let faces = zigzag_faces(&cycle_vertex_order(cycle), d_c);
                    let expected: HashSet<Edge> = cycle.iter().copied().collect();
                    assert_eq!(xor_edges(&faces), expected);
                }
            }
        }
    }

    // ----- Bridging (Lemma 25) ----------------------------------------------

    /// Cycle rank of a petgraph (|E| − |V| + #components).
    fn graph_cycle_rank<N>(g: &UnGraph<N, ()>) -> usize {
        let nodes: Vec<NodeIndex> = g.node_indices().collect();
        let (adj, _) = laplacian_adjacency(g, &nodes);
        g.edge_count() + connected_components(nodes.len(), &adj) - nodes.len()
    }

    #[test]
    fn bridge_connects_ports_and_adds_expected_cycles() {
        // Two connected measurement graphs with designated ports.
        // Disjoint block keys, per the bridge invariant.
        let s1 = SurgeryGraph {
            // 4-cycle, ports {0, 1, 2}
            graph: surgery_graph_with_ports(4, &[(0, 1), (1, 2), (2, 3), (3, 0)], "a", &[0, 1, 2]),
            cycle_checks: vec![],
            path_matching: vec![],
        };
        let s2 = SurgeryGraph {
            // triangle, ports {0, 1, 2}
            graph: surgery_graph_with_ports(3, &[(0, 1), (1, 2), (2, 0)], "b", &[0, 1, 2]),
            cycle_checks: vec![],
            path_matching: vec![],
        };
        let d = 2;
        let bridged = bridge_surgery_graphs(&s1, &s2, d);

        // Disjoint union plus exactly d bridge edges.
        assert_eq!(bridged.graph.node_count(), 4 + 3);
        assert_eq!(bridged.graph.edge_count(), 4 + 3 + d);
        // Ports of the product operator are P₁ ∪ P₂ (counting port vertices).
        let port_count = |s: &SurgeryGraph<&str>| s.ports_by_block().values().map(Vec::len).sum::<usize>();
        assert_eq!(port_count(&bridged), port_count(&s1) + port_count(&s2));

        // Exactly d edges cross from the left side to the right side, and each
        // crossing edge joins a left port to a right port (a matching).
        let offset = 4;
        let mut crossings = 0;
        let left_ports: HashSet<usize> =
            s1.ports_by_block().values().flatten().map(|p| p.index()).collect();
        let right_ports: HashSet<usize> =
            s2.ports_by_block().values().flatten().map(|p| offset + p.index()).collect();
        for e in bridged.graph.edge_indices() {
            let (a, b) = bridged.graph.edge_endpoints(e).unwrap();
            let (a, b) = (a.index(), b.index());
            if (a < offset) != (b < offset) {
                crossings += 1;
                let (l, r) = if a < offset { (a, b) } else { (b, a) };
                assert!(left_ports.contains(&l) && right_ports.contains(&r));
            }
        }
        assert_eq!(crossings, d);

        // Lemma 23: bridging two connected graphs with d edges adds d−1 cycles.
        let before = graph_cycle_rank(&s1.graph) + graph_cycle_rank(&s2.graph);
        assert_eq!(graph_cycle_rank(&bridged.graph), before + (d - 1));
    }

    #[test]
    fn bridge_caps_distance_at_available_ports() {
        let s1 = SurgeryGraph {
            // only one port available
            graph: surgery_graph_with_ports(3, &[(0, 1), (1, 2), (2, 0)], "a", &[0]),
            cycle_checks: vec![],
            path_matching: vec![],
        };
        let s2 = SurgeryGraph {
            graph: surgery_graph_with_ports(3, &[(0, 1), (1, 2), (2, 0)], "b", &[0, 1]),
            cycle_checks: vec![],
            path_matching: vec![],
        };
        // Requested distance 5 is capped at min(1, 2) = 1 bridge edge.
        let bridged = bridge_surgery_graphs(&s1, &s2, 5);
        assert_eq!(bridged.graph.edge_count(), 3 + 3 + 1);
    }

    #[test]
    fn bridge_two_surgery_graphs_end_to_end() {
        // Build two real surgery graphs and bridge them into one for L₁·L₂.
        let ps = |paulis: Vec<Pauli>| {
            PauliString::new(
                paulis
                    .into_iter()
                    .enumerate()
                    .map(|(i, p)| (PhysicalQubit(i), p))
                    .collect(),
            )
        };
        let op1 = ps(vec![Pauli::X, Pauli::X, Pauli::X, Pauli::X]);
        let stabs1 = vec![
            ps(vec![Pauli::Z, Pauli::Z, Pauli::I, Pauli::I]),
            ps(vec![Pauli::I, Pauli::I, Pauli::Z, Pauli::Z]),
        ];
        let op2 = ps(vec![Pauli::X, Pauli::X, Pauli::X, Pauli::X]);
        let stabs2 = vec![
            ps(vec![Pauli::Z, Pauli::Z, Pauli::I, Pauli::I]),
            ps(vec![Pauli::I, Pauli::I, Pauli::Z, Pauli::Z]),
        ];

        let s1 = surgery_graph(&stabs1, &op1, "left", &SurgeryGraphConfig::default());
        let s2 = surgery_graph(&stabs2, &op2, "right", &SurgeryGraphConfig::default());
        let (n1, e1) = (s1.graph.node_count(), s1.graph.edge_count());
        let (n2, e2) = (s2.graph.node_count(), s2.graph.edge_count());

        let d = 3;
        let bridged = bridge_surgery_graphs(&s1, &s2, d);

        assert_eq!(bridged.graph.node_count(), n1 + n2);
        let port_count = |s: &SurgeryGraph<&str>| s.ports_by_block().values().map(Vec::len).sum::<usize>();
        assert_eq!(port_count(&bridged), port_count(&s1) + port_count(&s2));

        // Count the actual bridge edges (crossing the disjoint-union boundary):
        // a matching of size ≤ d, each joining a left port to a right port.
        let left_ports: HashSet<usize> =
            s1.ports_by_block().values().flatten().map(|p| p.index()).collect();
        let right_ports: HashSet<usize> =
            s2.ports_by_block().values().flatten().map(|p| n1 + p.index()).collect();
        let mut bridge_edges = 0;
        let (mut used_left, mut used_right): (HashSet<usize>, HashSet<usize>) =
            (HashSet::new(), HashSet::new());
        for e in bridged.graph.edge_indices() {
            let (a, b) = bridged.graph.edge_endpoints(e).unwrap();
            let (a, b) = (a.index(), b.index());
            if (a < n1) != (b < n1) {
                bridge_edges += 1;
                let (l, r) = if a < n1 { (a, b) } else { (b, a) };
                assert!(left_ports.contains(&l) && right_ports.contains(&r));
                assert!(
                    used_left.insert(l) && used_right.insert(r),
                    "bridge is a matching"
                );
            }
        }
        assert_eq!(bridged.graph.edge_count(), e1 + e2 + bridge_edges);
        assert!(bridge_edges >= 1 && bridge_edges <= d);
        // The two previously separate measurement graphs are now one connected
        // graph (so it actually measures the joint operator).
        assert!(graph_is_connected(&bridged.graph));
    }

    #[test]
    fn skiptree_consecutive_labels_are_close() {
        // The defining SkipTree property: consecutive labels are within a short
        // tree path. We check induced-subgraph distance ≤ 3 between ports labeled
        // i and i+1 (the tree path, which the paper bounds by 3, is an upper bound).
        let test_graphs = [
            // A path — the classic SkipTree example.
            graph_from_edges(8, &(0..7).map(|i| (i, i + 1)).collect::<Vec<_>>()),
            // A balanced binary tree on 7 nodes.
            graph_from_edges(7, &[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]),
            // A denser connected graph.
            graph_from_edges(
                9,
                &[
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (6, 7),
                    (7, 8),
                    (0, 4),
                    (2, 7),
                ],
            ),
        ];

        for g in &test_graphs {
            let ports: Vec<NodeIndex> = g.node_indices().collect();
            let order = skiptree_labeling(g, &ports);
            assert_eq!(order.len(), ports.len(), "all ports labeled when connected");
            // Labels form a permutation of the ports.
            let distinct: HashSet<usize> = order.iter().map(|p| p.index()).collect();
            assert_eq!(distinct.len(), order.len());

            for w in order.windows(2) {
                let dist = bfs_distance(g, w[0], w[1]);
                assert!(
                    dist <= 3,
                    "consecutive labels {:?}->{:?} are {} apart (SkipTree bounds this by 3)",
                    w[0],
                    w[1],
                    dist
                );
            }
        }
    }

    /// Shortest-path hop distance between two vertices via BFS.
    fn bfs_distance(g: &UnGraph<(), ()>, src: NodeIndex, dst: NodeIndex) -> usize {
        if src == dst {
            return 0;
        }
        let mut dist: HashMap<NodeIndex, usize> = HashMap::from([(src, 0)]);
        let mut queue = std::collections::VecDeque::from([src]);
        while let Some(u) = queue.pop_front() {
            let du = dist[&u];
            for w in g.neighbors(u) {
                if w == dst {
                    return du + 1;
                }
                dist.entry(w).or_insert_with(|| {
                    queue.push_back(w);
                    du + 1
                });
            }
        }
        usize::MAX
    }
}
