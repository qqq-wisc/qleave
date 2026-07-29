use clap::{Parser, ValueEnum};
use qleave::{
    arch::{
        Architecture, BALANCED_LP_20, BALANCED_LP_24, GROSS, LP_20_2GROSS, LP_20_GROSS, SMALL,
        SPACE_EFFICIENT_LP_20, SPACE_EFFICIENT_LP_24, TWO_GROSS,
    },
    checks_to_physical_circuit::{
        CircuitStats, MergedQubit, PhysicalCircuit, compile_memory_experiment,
        compile_plain_memory_experiment, stream_sec_tick_count, stream_stim,
    },
    circuit::Circuit,
    circuit_to_checks::{
        CodeData, DeformedCheckSequence, pauli_product_circuit_to_physical_supports,
        physical_supports_to_stabilizer_checks,
    },
    compile::{
        compile, compile_explicit_clifford, compile_explicit_clifford_steps, compile_hybrid,
        compile_hybrid_steps, compile_steps,
    },
    graph_construction::SurgeryGraphConfig,
    parse::parse,
    pbc::Pauli,
    sec_schedule::SecSchedule,
};
use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process,
};

#[derive(Clone, Copy, ValueEnum)]
enum Arch {
    SpaceEfficientLp20,
    SpaceEfficientLp24,
    BalancedLp20,
    BalancedLp24,
    Lp20Gross,
    Lp20TwoGross,
    /// Tiny all-`bb18` architecture for quick end-to-end tests.
    Small,
    /// All-`gross` ([[144,12,12]]) architecture for benchmarking against GeneCS.
    Gross,
    /// All-`two_gross` ([[288,12,18]]) architecture for benchmarking against GeneCS.
    TwoGross,
}

impl Arch {
    fn architecture(self) -> &'static Architecture {
        match self {
            Arch::SpaceEfficientLp20 => &SPACE_EFFICIENT_LP_20,
            Arch::SpaceEfficientLp24 => &SPACE_EFFICIENT_LP_24,
            Arch::BalancedLp20 => &BALANCED_LP_20,
            Arch::BalancedLp24 => &BALANCED_LP_24,
            Arch::Lp20Gross => &LP_20_GROSS,
            Arch::Lp20TwoGross => &LP_20_2GROSS,
            Arch::Small => &SMALL,
            Arch::Gross => &GROSS,
            Arch::TwoGross => &TWO_GROSS,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Arch::SpaceEfficientLp20 => "space_efficient_lp20",
            Arch::SpaceEfficientLp24 => "space_efficient_lp24",
            Arch::BalancedLp20 => "balanced_lp20",
            Arch::BalancedLp24 => "balanced_lp24",
            Arch::Lp20Gross => "lp20_gross",
            Arch::Lp20TwoGross => "lp20_two_gross",
            Arch::Small => "small",
            Arch::Gross => "gross",
            Arch::TwoGross => "two_gross",
        }
    }
}

/// What the binary should produce from the input QASM.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// Compile to a Pauli-product circuit (`.pbc`).
    Compile,
    /// Lower all the way to a physical (stim) memory experiment (`.stim`).
    #[value(name = "to_physical", alias = "to-physical")]
    ToPhysical,
    /// Emit a plain (surgery-free) memory experiment for the chosen architecture:
    /// the bare code idling for `--distance` rounds, then transversal readout. The
    /// input QASM is ignored. Useful as a distance-debugging baseline.
    Memory,
}

/// Logical readout basis for the memory experiment. Only X and Z are supported:
/// these are CSS codes, which have no uniform-basis transversal Y readout (the
/// logical Y is the mixed-Pauli product X_L·Z_L, with no pure-Y stabilizers to
/// anchor the final detectors).
#[derive(Clone, Copy, ValueEnum)]
enum Basis {
    X,
    Z,
}

impl From<Basis> for Pauli {
    fn from(b: Basis) -> Self {
        match b {
            Basis::X => Pauli::X,
            Basis::Z => Pauli::Z,
        }
    }
}

/// CLI face of [`SecSchedule`]; see the `--sec-schedule` flag.
#[derive(Clone, Copy, ValueEnum)]
enum SecScheduleArg {
    Lrc,
    Legacy,
}

impl From<SecScheduleArg> for SecSchedule {
    fn from(s: SecScheduleArg) -> Self {
        match s {
            SecScheduleArg::Lrc => SecSchedule::Lrc,
            SecScheduleArg::Legacy => SecSchedule::Legacy,
        }
    }
}

#[derive(Parser)]
#[command(about = "Compile a QASM circuit to a Pauli product circuit")]
struct Cli {
    /// Input QASM file (not required for `--mode memory`).
    circuit: Option<PathBuf>,

    /// Named architecture preset (mutually exclusive with --mem-cap/--proc-cap)
    #[arg(long, group = "target")]
    arch: Option<Arch>,

    /// Memory qubit capacity (requires --proc-cap)
    #[arg(long, group = "target", requires = "proc_cap")]
    mem_cap: Option<usize>,

    /// Processor qubit capacity (requires --mem-cap)
    #[arg(long, requires = "mem_cap")]
    proc_cap: Option<usize>,

    /// Output path for the compiled circuit. Use "-" for stdout. Defaults to <input_stem>.pbc in cwd.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Write intermediate circuits to this directory.
    /// If passed without a value, defaults to <input_stem>-intermediates/.
    #[arg(long, num_args = 0..=1, default_missing_value = "<auto>")]
    intermediates: Option<String>,

    /// Randomly resolve conditional rotations to unconditional ones before applying the Clifford frame
    #[arg(long)]
    simulate_corrections: bool,

    /// Skip redundant load/store pairs when a qubit appears in consecutive subcircuits
    #[arg(long)]
    skip_redundant_ls: bool,

    /// Materialize Cliffords as explicit Clifford gadgets on a magic ancilla
    /// instead of absorbing them into the measurement frame. Produces a
    /// measurement-only Pauli-product circuit. (`--mode compile` only.) With
    /// --intermediates, also writes pbc_explicit_clifford.txt.
    #[arg(long)]
    explicit_clifford: bool,

    /// Materialize only the Cliffords that came from a *correction*, absorbing
    /// the circuit's own Cliffords into the measurement frame as usual. The
    /// middle ground between the default and --explicit-clifford: every emitted
    /// measurement's Pauli string is fixed at compile time (only its sign, and
    /// whether a correction gadget runs at all, depend on measurement outcomes),
    /// at a fraction of the gadget count of --explicit-clifford. Only has an
    /// effect together with --simulate-corrections, which is what puts the
    /// conditional rotations in the circuit. (`--mode compile` only.) With
    /// --intermediates, also writes pbc_hybrid_clifford.txt.
    #[arg(long, conflicts_with = "explicit_clifford")]
    hybrid: bool,

    /// Use SAT-based optimal partitioner (max_k and initial bound learned from greedy).
    /// Implies --skip-redundant-ls (required for Belady's optimality guarantee).
    #[arg(long)]
    sat: bool,

    /// Timeout in seconds for the SAT optimizer (returns best solution found so far)
    #[arg(long, requires = "sat")]
    sat_timeout: Option<u64>,

    /// Pipeline target. `to_physical` lowers the compiled circuit through code
    /// surgery to a physical stim memory experiment (requires qldpc).
    #[arg(long, value_enum, default_value_t = Mode::Compile)]
    mode: Mode,

    /// (to_physical) Code-surgery distance: the number of stabilizer-measurement
    /// rounds and the number of bridge qubits per merge. In the paper this scales
    /// with the code distance.
    #[arg(long, default_value_t = 18)]
    distance: usize,

    /// (to_physical) Logical basis the memory experiment reads out.
    #[arg(long, value_enum, default_value_t = Basis::Z)]
    basis: Basis,

    /// (to_physical / memory) Lower MPP syndrome measurements to an ancilla-based,
    /// TICK-layered syndrome-extraction circuit (no noise injected).
    #[arg(long)]
    syndrome_extraction_circuits: bool,

    /// (to_physical / memory) Syndrome-extraction schedule: staggered left-right
    /// circuits (arXiv:2603.05481; conflict-phased, edge-colored, cross-round
    /// staggering) or the legacy check-colored lockstep schedule.
    #[arg(long, value_enum, default_value_t = SecScheduleArg::Lrc)]
    sec_schedule: SecScheduleArg,

    /// (to_physical / memory) Skip the transversal readout, logical-observable and
    /// final-detector pass (the stabilizer-frame solve). Produces a circuit with no
    /// observables — useful for profiling, since the solve dominates compile time.
    #[arg(long)]
    no_memory_observables: bool,

    /// (to_physical / memory) Run qldpc's BP+OSD logical-operator weight reduction
    /// only on code blocks with at most this many physical qubits. Reduction lowers
    /// logical weight (cheaper surgery) but its cost explodes with code size: the
    /// small processor/magic blocks reduce in seconds, while the large memory codes
    /// (lp3_7_20, 4350 qubits) take >10 min. The default reduces the former and skips
    /// the latter; all results are cached. Set 0 to skip reduction entirely, or a
    /// large value to reduce every block.
    #[arg(long, default_value_t = 2000)]
    logical_operator_reduction_threshold: usize,

    /// (to_physical / memory) Write a JSON stats record (architecture, surgery-graph
    /// config, rounds, per-type qubit counts, syndrome cycles, SEC depth) to this path.
    /// The path is optional; passing `--stats` with no value writes to `stats.json`.
    #[arg(long, num_args = 0..=1, default_missing_value = "stats.json")]
    stats: Option<PathBuf>,

    /// (to_physical) Number of randomized congestion-aware expander
    /// constructions to try per surgery graph; the smallest result is kept.
    #[arg(long, default_value_t = SurgeryGraphConfig::default().trials)]
    expander_trials: usize,

    /// (to_physical) Expander construction: rebuild the BFS spanning forest
    /// every this-many steps (keeps tree paths short).
    #[arg(long, default_value_t = SurgeryGraphConfig::default().reset_period)]
    expander_reset_period: usize,

    /// (to_physical) Expander construction: iteration bound for the random
    /// matching phase (Phase 2).
    #[arg(long, default_value_t = SurgeryGraphConfig::default().max_iterations)]
    expander_max_iterations: usize,

    /// (to_physical) Expander construction: LDPC qubit-degree bound `d_q` used
    /// by the degree-aware cycle partition (larger packs more cycles per layer).
    #[arg(long, default_value_t = SurgeryGraphConfig::default().qubit_degree)]
    expander_qubit_degree: usize,

    /// (to_physical) Seed for the RNG driving the randomized expander
    /// construction; fixing it keeps surgery-graph construction deterministic.
    #[arg(long, default_value_t = SurgeryGraphConfig::default().seed)]
    surgery_seed: u64,

    /// (to_physical) Cellulation: the largest face (cycle check) degree the
    /// zigzag cellulation may produce.
    #[arg(long, default_value_t = SurgeryGraphConfig::default().max_check_degree)]
    cellulation_degree: usize,

    /// (to_physical) Disable the surgery-graph caches (the full-support,
    /// per-block, and shape layers); every measurement rebuilds its graphs
    /// from scratch.
    #[arg(long)]
    no_surgery_cache: bool,
}

impl Cli {
    /// Assemble the surgery-graph tuning parameters from the relevant flags.
    fn surgery_graph_config(&self) -> SurgeryGraphConfig {
        SurgeryGraphConfig {
            trials: self.expander_trials,
            reset_period: self.expander_reset_period,
            max_iterations: self.expander_max_iterations,
            qubit_degree: self.expander_qubit_degree,
            seed: self.surgery_seed,
            max_check_degree: self.cellulation_degree,
            caching: !self.no_surgery_cache,
        }
    }
}

fn main() {
    let cli = Cli::parse();

    if cli.mode == Mode::Memory {
        run_plain_memory(&cli);
        return;
    }

    let circuit_path = cli.circuit.clone().unwrap_or_else(|| {
        eprintln!("error: a circuit (QASM) argument is required for this mode");
        process::exit(1);
    });

    let (mem_cap, proc_cap) = match cli.arch {
        Some(arch) => {
            let a = arch.architecture();
            (a.memory_capacity, a.processor_capacity)
        }
        None => match cli.mem_cap {
            Some(mem) => (mem, cli.proc_cap.unwrap()),
            None => (
                SPACE_EFFICIENT_LP_20.memory_capacity,
                SPACE_EFFICIENT_LP_20.processor_capacity,
            ),
        },
    };

    let qasm = fs::read_to_string(&circuit_path).unwrap_or_else(|e| {
        eprintln!("error reading {}: {e}", circuit_path.display());
        process::exit(1);
    });

    let circuit = parse(&qasm).unwrap_or_else(|e| {
        eprintln!("parse error: {e}");
        process::exit(1);
    });

    if cli.mode == Mode::ToPhysical {
        run_to_physical(&cli, &circuit_path, circuit);
        return;
    }

    if circuit.num_qubits > mem_cap {
        eprintln!("circuit qubits exceeds memory capacity");
        process::exit(1);
    }

    let output_path = cli.output.unwrap_or_else(|| {
        let stem = circuit_path.file_stem().unwrap_or_default();
        PathBuf::from(stem).with_extension("pbc")
    });

    let intermediates_dir = cli.intermediates.map(|s| {
        if s == "<auto>" {
            let stem = circuit_path.file_stem().unwrap_or_default();
            let mut name = stem.to_os_string();
            name.push("-intermediates");
            PathBuf::from(name)
        } else {
            PathBuf::from(s)
        }
    });

    // If using SAT, we must skip redundant load/stores to guarantee optimality (Belady's algorithm)
    let skip_redundant = cli.skip_redundant_ls || cli.sat;

    let pbc_final = if let Some(dir) = intermediates_dir {
        fs::create_dir_all(&dir).unwrap_or_else(|e| {
            eprintln!("error creating intermediates dir: {e}");
            process::exit(1);
        });
        let (writes, pbc_final) = if cli.explicit_clifford {
            let (load_store, pbc_pre, explicit, pbc_final) = compile_explicit_clifford_steps(
                circuit,
                proc_cap,
                cli.simulate_corrections,
                skip_redundant,
                cli.sat,
                cli.sat_timeout,
            );
            (
                vec![
                    ("load_store.txt", load_store.to_string()),
                    ("pbc_w_clifford.txt", pbc_pre.to_string()),
                    ("pbc_explicit_clifford.txt", explicit.to_string()),
                ],
                pbc_final,
            )
        } else if cli.hybrid {
            let (load_store, pbc_pre, hybrid, pbc_final) = compile_hybrid_steps(
                circuit,
                proc_cap,
                cli.simulate_corrections,
                skip_redundant,
                cli.sat,
                cli.sat_timeout,
            );
            (
                vec![
                    ("load_store.txt", load_store.to_string()),
                    ("pbc_w_clifford.txt", pbc_pre.to_string()),
                    ("pbc_hybrid_clifford.txt", hybrid.to_string()),
                ],
                pbc_final,
            )
        } else {
            let (load_store, pbc_pre, pbc_final) = compile_steps(
                circuit,
                proc_cap,
                cli.simulate_corrections,
                skip_redundant,
                cli.sat,
                cli.sat_timeout,
            );
            (
                vec![
                    ("load_store.txt", load_store.to_string()),
                    ("pbc_w_clifford.txt", pbc_pre.to_string()),
                ],
                pbc_final,
            )
        };
        for (name, content) in writes {
            fs::write(dir.join(name), content).unwrap_or_else(|e| {
                eprintln!("error writing {name}: {e}");
                process::exit(1);
            });
        }
        eprintln!("intermediates written to {}/", dir.display());
        pbc_final
    } else if cli.explicit_clifford {
        compile_explicit_clifford(
            circuit,
            proc_cap,
            cli.simulate_corrections,
            skip_redundant,
            cli.sat,
            cli.sat_timeout,
        )
    } else if cli.hybrid {
        compile_hybrid(
            circuit,
            proc_cap,
            cli.simulate_corrections,
            skip_redundant,
            cli.sat,
            cli.sat_timeout,
        )
    } else {
        compile(
            circuit,
            proc_cap,
            cli.simulate_corrections,
            skip_redundant,
            cli.sat,
            cli.sat_timeout,
        )
    };

    if output_path == Path::new("-") {
        let stdout = io::stdout();
        let mut h = stdout.lock();
        if let Err(e) = writeln!(h, "{pbc_final}") {
            if e.kind() == io::ErrorKind::BrokenPipe {
                process::exit(0);
            }
            eprintln!("error writing to stdout: {e}");
            process::exit(1);
        }
    } else {
        fs::write(&output_path, pbc_final.to_string()).unwrap_or_else(|e| {
            eprintln!("error writing {}: {e}", output_path.display());
            process::exit(1);
        });
        eprintln!(
            "Wrote {} instructions to {}",
            pbc_final.instructions.len(),
            output_path.display()
        );
    }
}

/// `--mode to_physical`: compile the logical circuit to Pauli-product
/// measurements, lower those through code surgery to stabilizer checks, and emit
/// a physical (stim) memory experiment. Requires qldpc to build the code blocks.
fn run_to_physical(cli: &Cli, circuit_path: &Path, circuit: Circuit) {
    let arch = cli.arch.map(Arch::architecture).unwrap_or(&SMALL);
    eprintln!(
        "Building CSS blocks via qldpc (memory={}, processor={}, magic={}); \
         large memory codes are slow...",
        arch.memory_code, arch.processor_code, arch.magic_code,
    );
    let blocks = arch.css_blocks().unwrap_or_else(|e| {
        eprintln!("error building css blocks: {e}");
        process::exit(1);
    });
    let codes = CodeData::from_blocks(
        &blocks.0,
        &blocks.1,
        &blocks.2,
        cli.logical_operator_reduction_threshold,
    )
    .unwrap_or_else(|e| {
        eprintln!("error building code data: {e}");
        process::exit(1);
    });

    let ppm = if cli.explicit_clifford {
        compile_explicit_clifford(
            circuit,
            arch.processor_capacity,
            cli.simulate_corrections,
            cli.skip_redundant_ls || cli.sat,
            cli.sat,
            cli.sat_timeout,
        )
    } else {
        compile(
            circuit,
            arch.processor_capacity,
            cli.simulate_corrections,
            cli.skip_redundant_ls || cli.sat,
            cli.sat,
            cli.sat_timeout,
        )
    };
    eprintln!("Number of PPM instructions: {}", ppm.instructions.len());

    let supports = pauli_product_circuit_to_physical_supports(&ppm, &codes);
    eprintln!("Number of supports: {}", supports.len());
    let checks = physical_supports_to_stabilizer_checks(
        &supports,
        &codes,
        cli.distance,
        &cli.surgery_graph_config(),
    );
    let output_path = cli.output.clone().unwrap_or_else(|| {
        let stem = circuit_path.file_stem().unwrap_or_default();
        PathBuf::from(stem).with_extension("stim")
    });
    // Without observables the circuit is a pure function of the check sequence, so it
    // can be lowered and written in one streaming pass and never has to exist in full.
    // The observable-bearing path cannot: its records come from a frame solve over the
    // whole circuit.
    if cli.no_memory_observables {
        stream_physical(&checks, cli, &output_path, "physical circuit");
        return;
    }
    let memory = compile_memory_experiment(checks, cli.distance, &codes, cli.basis.into(), true);
    write_physical(memory, cli, &output_path, "physical circuit");
}

/// `--no-memory-observables` counterpart of [`write_physical`]: lower the check
/// sequence and write it as stim text in a single streaming pass, so peak memory
/// stays at roughly one deformation's expansion instead of the whole circuit's.
///
/// `sec_depth` is free when the SEC expansion is what gets written; otherwise it
/// costs a second pass over the (regenerated, never retained) gate stream, which is
/// the same trade the materializing path makes.
fn stream_physical(checks: &DeformedCheckSequence, cli: &Cli, output_path: &Path, what: &str) {
    const BUF_BYTES: usize = 1 << 20;
    let schedule: Option<SecSchedule> = cli
        .syndrome_extraction_circuits
        .then(|| cli.sec_schedule.into());

    let written = if output_path == Path::new("-") {
        let stdout = io::stdout();
        let out = io::BufWriter::with_capacity(BUF_BYTES, stdout.lock());
        stream_stim(checks, cli.distance, schedule, out)
    } else {
        let file = fs::File::create(output_path).unwrap_or_else(|e| {
            eprintln!("error creating {}: {e}", output_path.display());
            process::exit(1);
        });
        let out = io::BufWriter::with_capacity(BUF_BYTES, file);
        stream_stim(checks, cli.distance, schedule, out)
    };
    let (ticks, mut stats) = written.unwrap_or_else(|e| {
        if e.kind() == io::ErrorKind::BrokenPipe {
            process::exit(0);
        }
        eprintln!("error writing {}: {e}", output_path.display());
        process::exit(1);
    });
    if output_path != Path::new("-") {
        eprintln!("Wrote {what} to {}", output_path.display());
    }

    if let Some(path) = &cli.stats {
        stats.sec_depth = match schedule {
            Some(_) => ticks,
            None => stream_sec_tick_count(checks, cli.distance, cli.sec_schedule.into()),
        };
        write_stats(path, cli, stats);
    }
}

/// Lower the compiled merged-qubit circuit and write it out as stim text, emitting
/// a `--stats` record along the way.
///
/// Nothing downstream of the flattened MPP circuit is ever materialized: with
/// `--syndrome-extraction-circuits` the SEC expansion (which multiplies the gate
/// count by the check weight) and the stim text (gigabytes on large circuits) are
/// both streamed gate-by-gate into the output file, so peak memory stays at roughly
/// the size of the MPP circuit. `sec_depth` is counted during that same pass; when
/// SEC output is off but `--stats` still wants the metric, a separate counting pass
/// supplies it — cheap, since it keeps nothing.
fn write_physical(
    memory: PhysicalCircuit<MergedQubit>,
    cli: &Cli,
    output_path: &Path,
    what: &str,
) {
    /// Output buffer: the stim text runs to gigabytes and is produced a few bytes
    /// at a time, so a bigger buffer than `BufWriter`'s default 8 KiB is worth it.
    const BUF_BYTES: usize = 1 << 20;

    // Every other --stats metric is read off the merged circuit, before it is
    // consumed by the flattening below.
    let mut stats = cli
        .stats
        .as_ref()
        .map(|_| CircuitStats::compute(&memory, 0));
    let flat = memory.into_flat();

    let schedule: Option<SecSchedule> = cli
        .syndrome_extraction_circuits
        .then(|| cli.sec_schedule.into());
    if let (Some(stats), None) = (stats.as_mut(), schedule) {
        stats.sec_depth = flat.sec_tick_count(cli.sec_schedule.into());
    }

    let ticks = if output_path == Path::new("-") {
        let stdout = io::stdout();
        flat.write_stim(schedule, io::BufWriter::with_capacity(BUF_BYTES, stdout.lock()))
    } else {
        let file = fs::File::create(output_path).unwrap_or_else(|e| {
            eprintln!("error creating {}: {e}", output_path.display());
            process::exit(1);
        });
        flat.write_stim(schedule, io::BufWriter::with_capacity(BUF_BYTES, file))
    };
    let ticks = ticks.unwrap_or_else(|e| {
        if e.kind() == io::ErrorKind::BrokenPipe {
            process::exit(0);
        }
        eprintln!("error writing {}: {e}", output_path.display());
        process::exit(1);
    });
    if output_path != Path::new("-") {
        eprintln!("Wrote {what} to {}", output_path.display());
    }

    if let (Some(path), Some(mut stats)) = (&cli.stats, stats) {
        if schedule.is_some() {
            stats.sec_depth = ticks;
        }
        write_stats(path, cli, stats);
    }
}

/// Write a `--stats` JSON record: the architecture, the surgery-graph config, the
/// round count, the per-type qubit counts, and the syndrome-cycle / SEC-depth
/// metrics. The ancilla count is the SEC pool only when `--syndrome-extraction-circuits`
/// is set (the MPP form uses no ancillas), otherwise 0.
fn write_stats(path: &Path, cli: &Cli, stats: CircuitStats) {
    let arch = cli.arch.map(Arch::name).unwrap_or("small");
    let c = cli.surgery_graph_config();
    let ancilla = if cli.syndrome_extraction_circuits {
        stats.sec_ancilla_pool
    } else {
        0
    };
    let total = stats.code_qubits + stats.edge_qubits + ancilla;
    let json = format!(
        "{{\n  \"arch\": \"{arch}\",\n  \"rounds\": {rounds},\n  \
         \"config\": {{\n    \
         \"expander_trials\": {trials},\n    \
         \"expander_reset_period\": {reset_period},\n    \
         \"expander_max_iterations\": {max_iterations},\n    \
         \"expander_qubit_degree\": {qubit_degree},\n    \
         \"surgery_seed\": {seed},\n    \
         \"cellulation_degree\": {max_check_degree}\n  }},\n  \
         \"qubits\": {{\n    \
         \"code\": {code},\n    \
         \"edge\": {edge},\n    \
         \"ancilla\": {ancilla},\n    \
         \"total\": {total}\n  }},\n  \
         \"syndrome_cycles\": {cycles},\n  \
         \"sec_depth\": {depth}\n}}\n",
        rounds = cli.distance,
        trials = c.trials,
        reset_period = c.reset_period,
        max_iterations = c.max_iterations,
        qubit_degree = c.qubit_degree,
        seed = c.seed,
        max_check_degree = c.max_check_degree,
        code = stats.code_qubits,
        edge = stats.edge_qubits,
        cycles = stats.syndrome_cycles,
        depth = stats.sec_depth,
    );
    if path == Path::new("-") {
        print!("{json}");
    } else {
        fs::write(path, &json).unwrap_or_else(|e| {
            eprintln!("error writing stats to {}: {e}", path.display());
            process::exit(1);
        });
        eprintln!("Wrote stats to {}", path.display());
    }
}

/// `--mode memory`: emit a plain (surgery-free) memory experiment for the chosen
/// architecture — the bare code idling for `--distance` rounds then transversal
/// `--basis` readout. The QASM input is ignored. A distance-debugging baseline.
fn run_plain_memory(cli: &Cli) {
    let arch = cli.arch.map(Arch::architecture).unwrap_or(&SMALL);
    eprintln!(
        "Building CSS blocks via qldpc (memory={}, processor={}, magic={}); \
         large memory codes are slow...",
        arch.memory_code, arch.processor_code, arch.magic_code,
    );
    let blocks = arch.css_blocks().unwrap_or_else(|e| {
        eprintln!("error building css blocks: {e}");
        process::exit(1);
    });
    let codes = CodeData::from_blocks(
        &blocks.0,
        &blocks.1,
        &blocks.2,
        cli.logical_operator_reduction_threshold,
    )
    .unwrap_or_else(|e| {
        eprintln!("error building code data: {e}");
        process::exit(1);
    });

    let memory = compile_plain_memory_experiment(
        &codes,
        cli.distance,
        cli.basis.into(),
        !cli.no_memory_observables,
    );
    let output_path = cli
        .output
        .clone()
        .unwrap_or_else(|| PathBuf::from("memory.stim"));
    write_physical(memory, cli, &output_path, "plain memory experiment");
}
