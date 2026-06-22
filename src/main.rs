use clap::{Parser, ValueEnum};
use qleave::{
    arch::{
        Architecture, BALANCED_LP_20, BALANCED_LP_24, SMALL, SPACE_EFFICIENT_LP_20,
        SPACE_EFFICIENT_LP_24,
    },
    checks_to_physical_circuit::{compile_memory_experiment, compile_plain_memory_experiment},
    circuit::Circuit,
    circuit_to_checks::{
        CodeData, pauli_product_circuit_to_physical_supports,
        physical_supports_to_stabilizer_checks,
    },
    compile::{compile, compile_steps},
    parse::parse,
    pbc::Pauli,
};
use std::{fs, io::{self, Write}, path::{Path, PathBuf}, process};

#[derive(Clone, Copy, ValueEnum)]
enum Arch {
    SpaceEfficientLp20,
    SpaceEfficientLp24,
    BalancedLp20,
    BalancedLp24,
    /// Tiny all-`bb18` architecture for quick end-to-end tests.
    Small,
}

impl Arch {
    fn architecture(self) -> &'static Architecture {
        match self {
            Arch::SpaceEfficientLp20 => &SPACE_EFFICIENT_LP_20,
            Arch::SpaceEfficientLp24 => &SPACE_EFFICIENT_LP_24,
            Arch::BalancedLp20 => &BALANCED_LP_20,
            Arch::BalancedLp24 => &BALANCED_LP_24,
            Arch::Small => &SMALL,
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

/// Logical readout basis for the memory experiment.
#[derive(Clone, Copy, ValueEnum)]
enum Basis {
    X,
    Y,
    Z,
}

impl From<Basis> for Pauli {
    fn from(b: Basis) -> Self {
        match b {
            Basis::X => Pauli::X,
            Basis::Y => Pauli::Y,
            Basis::Z => Pauli::Z,
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
            None => (SPACE_EFFICIENT_LP_20.memory_capacity, SPACE_EFFICIENT_LP_20.processor_capacity),
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

    let pbc_final = if let Some(dir) = intermediates_dir {
        fs::create_dir_all(&dir).unwrap_or_else(|e| {
            eprintln!("error creating intermediates dir: {e}");
            process::exit(1);
        });
        let (load_store, pbc_pre, pbc_final) = compile_steps(
            circuit,
            proc_cap,
            cli.simulate_corrections,
            cli.skip_redundant_ls || cli.sat, // If using SAT, we must skip redundant load/stores to guarantee optimality (Belady's algorithm)
            cli.sat,
            cli.sat_timeout,
        );
        let writes = [
            ("load_store.txt", load_store.to_string()),
            ("pbc_w_clifford.txt", pbc_pre.to_string()),
        ];
        for (name, content) in writes {
            fs::write(dir.join(name), content).unwrap_or_else(|e| {
                eprintln!("error writing {name}: {e}");
                process::exit(1);
            });
        }
        eprintln!("intermediates written to {}/", dir.display());
        pbc_final
    } else {
        compile(
            circuit,
            proc_cap,
            cli.simulate_corrections,
            cli.skip_redundant_ls || cli.sat, // If using SAT, we must skip redundant load/stores to guarantee optimality (Belady's algorithm)
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
    let codes = CodeData::from_blocks(&blocks.0, &blocks.1, &blocks.2).unwrap_or_else(|e| {
        eprintln!("error building code data: {e}");
        process::exit(1);
    });

    let ppm = compile(
        circuit,
        arch.processor_capacity,
        cli.simulate_corrections,
        cli.skip_redundant_ls || cli.sat,
        cli.sat,
        cli.sat_timeout,
    );
    eprintln!("Number of PPM instructions: {}", ppm.instructions.len());

    let supports = pauli_product_circuit_to_physical_supports(&ppm, &codes);
    eprintln!("Number of supports: {}", supports.len());
    let checks = physical_supports_to_stabilizer_checks(&supports, &codes, cli.distance);
    let memory = compile_memory_experiment(checks, cli.distance, &codes, cli.basis.into());
    let stim = memory.flatten().to_string();

    let output_path = cli.output.clone().unwrap_or_else(|| {
        let stem = circuit_path.file_stem().unwrap_or_default();
        PathBuf::from(stem).with_extension("stim")
    });

    if output_path == Path::new("-") {
        let stdout = io::stdout();
        let mut h = stdout.lock();
        if let Err(e) = write!(h, "{stim}") {
            if e.kind() == io::ErrorKind::BrokenPipe {
                process::exit(0);
            }
            eprintln!("error writing to stdout: {e}");
            process::exit(1);
        }
    } else {
        fs::write(&output_path, &stim).unwrap_or_else(|e| {
            eprintln!("error writing {}: {e}", output_path.display());
            process::exit(1);
        });
        eprintln!("Wrote physical circuit to {}", output_path.display());
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
    let codes = CodeData::from_blocks(&blocks.0, &blocks.1, &blocks.2).unwrap_or_else(|e| {
        eprintln!("error building code data: {e}");
        process::exit(1);
    });

    let memory = compile_plain_memory_experiment(&codes, cli.distance, cli.basis.into());
    let stim = memory.flatten().to_string();

    let output_path = cli.output.clone().unwrap_or_else(|| PathBuf::from("memory.stim"));
    if output_path == Path::new("-") {
        let stdout = io::stdout();
        let mut h = stdout.lock();
        if let Err(e) = write!(h, "{stim}") {
            if e.kind() == io::ErrorKind::BrokenPipe {
                process::exit(0);
            }
            eprintln!("error writing to stdout: {e}");
            process::exit(1);
        }
    } else {
        fs::write(&output_path, &stim).unwrap_or_else(|e| {
            eprintln!("error writing {}: {e}", output_path.display());
            process::exit(1);
        });
        eprintln!("Wrote plain memory experiment to {}", output_path.display());
    }
}
