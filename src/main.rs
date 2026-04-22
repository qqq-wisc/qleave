use clap::{Parser, ValueEnum};
use oratomic_compiler::{
    arch::{BALANCED_LP_20, BALANCED_LP_24, SPACE_EFFICIENT_LP_20, SPACE_EFFICIENT_LP_24},
    compile::{compile, compile_steps},
    parse::parse,
};
use std::{fs, path::PathBuf, process};

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

    /// Write intermediate circuits to this directory (default: "out")
    #[arg(long, num_args = 0..=1, default_missing_value = "out")]
    intermediates: Option<PathBuf>,
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
            None => (BALANCED_LP_20.memory_capacity, BALANCED_LP_20.processor_capacity),
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

    if let Some(dir) = cli.intermediates {
        fs::create_dir_all(&dir).unwrap_or_else(|e| {
            eprintln!("error creating intermediates dir: {e}");
            process::exit(1);
        });
        let (load_store, pbc_pre, pbc_final) = compile_steps(circuit, proc_cap);
        let writes = [
            ("load_store.txt", load_store.to_string()),
            ("pbc_w_clifford.txt", pbc_pre.to_string()),
            ("pbc_final.txt", pbc_final.to_string()),
        ];
        for (name, content) in writes {
            fs::write(dir.join(name), content).unwrap_or_else(|e| {
                eprintln!("error writing {name}: {e}");
                process::exit(1);
            });
        }
        println!("{pbc_final}");
    } else {
        println!("{}", compile(circuit, proc_cap));
    }
}
