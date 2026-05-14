use clap::{Parser, ValueEnum};
use qleave::{
    arch::{BALANCED_LP_20, BALANCED_LP_24, SPACE_EFFICIENT_LP_20, SPACE_EFFICIENT_LP_24},
    compile::{compile, compile_steps},
    parse::parse,
};
use std::{fs, io::{self, Write}, path::{Path, PathBuf}, process};

#[derive(Clone, ValueEnum)]
enum Arch {
    SpaceEfficientLp20,
    SpaceEfficientLp24,
    BalancedLp20,
    BalancedLp24,
}

#[derive(Parser)]
#[command(about = "Compile a QASM circuit to a Pauli product circuit")]
struct Cli {
    /// Input QASM file
    circuit: PathBuf,

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
}

fn main() {
    let cli = Cli::parse();

    let (mem_cap, proc_cap) = match cli.arch {
        Some(Arch::SpaceEfficientLp20) => (
            SPACE_EFFICIENT_LP_20.memory_capacity,
            SPACE_EFFICIENT_LP_20.processor_capacity,
        ),
        Some(Arch::SpaceEfficientLp24) => (
            SPACE_EFFICIENT_LP_24.memory_capacity,
            SPACE_EFFICIENT_LP_24.processor_capacity,
        ),
        Some(Arch::BalancedLp20) => (
            BALANCED_LP_20.memory_capacity,
            BALANCED_LP_20.processor_capacity,
        ),
        Some(Arch::BalancedLp24) => (
            BALANCED_LP_24.memory_capacity,
            BALANCED_LP_24.processor_capacity,
        ),
        None => match cli.mem_cap {
            Some(mem) => (mem, cli.proc_cap.unwrap()),
            None => (SPACE_EFFICIENT_LP_20.memory_capacity, SPACE_EFFICIENT_LP_20.processor_capacity),
        },
    };

    let qasm = fs::read_to_string(&cli.circuit).unwrap_or_else(|e| {
        eprintln!("error reading {}: {e}", cli.circuit.display());
        process::exit(1);
    });

    let circuit = parse(&qasm).unwrap_or_else(|e| {
        eprintln!("parse error: {e}");
        process::exit(1);
    });
    if circuit.num_qubits > mem_cap {
        eprintln!("circuit qubits exceeds memory capacity");
        process::exit(1);
    }

    let output_path = cli.output.unwrap_or_else(|| {
        let stem = cli.circuit.file_stem().unwrap_or_default();
        PathBuf::from(stem).with_extension("pbc")
    });

    let intermediates_dir = cli.intermediates.map(|s| {
        if s == "<auto>" {
            let stem = cli.circuit.file_stem().unwrap_or_default();
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
