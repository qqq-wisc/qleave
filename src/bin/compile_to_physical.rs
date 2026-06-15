//! Drives the physical-circuit backend: a sequence of logical Pauli-product
//! measurements over an architecture's memory / processor / magic code blocks is
//! lowered, via code surgery, to stabilizer checks and then to a physical (stim)
//! circuit.
//!
//! With no arguments it runs a small self-contained demo over `bb18`
//! [[248, 10, <=18]] blocks so the whole pipeline executes quickly. Pass a
//! preset name (e.g. `space-efficient-lp20`) to build that architecture's real
//! code blocks instead — note the large memory codes make qldpc's
//! logical-operator reduction slow.
//!
//! Requires a Python interpreter with `qldpc` installed; point `$QLDPC_PYTHON`
//! at it (default `python3`).

use std::{error::Error, fs, process};

use qleave::{
    arch::Architecture,
    checks_to_physical_circuit::{PhysicalCircuit, checks_to_physical_circuit},
    circuit::Circuit,
    circuit_to_checks::{
        CSSCodeBlock, CodeData, css_block, pauli_product_circuit_to_physical_supports,
        physical_supports_to_stabilizer_checks,
    },
    compile::compile,
    parse::parse,
    pbc::{
        ArchitectureQubit::{self, Magic, Memory, Processor},
        Pauli, PauliAxis, PauliProductCircuit, PauliProductOperation, PauliString, Sign,
    },
};

/// Number of code-surgery bridge qubits / stabilizer rounds. Kept small for the
/// demo; in the paper this scales with the code distance.
const DISTANCE: usize = 18;

fn ppc_to_physical_circuit(
    circ: &PauliProductCircuit,
    memory_block: &CSSCodeBlock,
    processor_block: &CSSCodeBlock,
    magic_block: &CSSCodeBlock,
    distance: usize,
) -> Result<PhysicalCircuit<usize>, Box<dyn std::error::Error>> {
    let codes = CodeData::from_blocks(memory_block, processor_block, magic_block)?;
    let supports = pauli_product_circuit_to_physical_supports(circ, &codes);
    let checks = physical_supports_to_stabilizer_checks(&supports, &codes, distance);
    Ok(checks_to_physical_circuit(checks, distance).flatten())
}

fn logical_circuit_to_physical_circuit(
    circ: Circuit,
    arch: &Architecture,
    distance: usize,
) -> Result<PhysicalCircuit<usize>, Box<dyn std::error::Error>> {
    let blocks = arch.css_blocks()?;
    let codes = CodeData::from_blocks(&blocks.0, &blocks.1, &blocks.2)?;
    let ppm = compile(circ, arch.processor_capacity, false, true, false, None);
    let supports = pauli_product_circuit_to_physical_supports(&ppm, &codes);
    let checks = physical_supports_to_stabilizer_checks(&supports, &codes, distance);
    Ok(checks_to_physical_circuit(checks, distance).flatten())
}

/// Append a logical Pauli-product measurement to `circ`. 
fn measure(circ: &mut PauliProductCircuit, pairs: Vec<(ArchitectureQubit, Pauli)>) {
    let id = circ.allocate_meas_id();
    circ.instructions.push(PauliProductOperation::Measurement {
        axis: PauliAxis {
            sign: Sign::One,
            pauli_string: PauliString::new(pairs),
        },
        id,
    });
}

/// A tiny measurement-only circuit: a memory<->processor joint Z measurement and
/// a memory<->magic joint X measurement, the two-block PPMs code surgery handles.
fn demo_ppm_circuit() -> PauliProductCircuit {
    let mut circ = PauliProductCircuit::new();
    measure(
        &mut circ,
        vec![(Memory(0), Pauli::Z), (Processor(0), Pauli::Z)],
    );
    measure(&mut circ, vec![(Memory(1), Pauli::X), (Magic(0), Pauli::X)]);
    circ
}

fn demo_circuit() -> Circuit {
    let qasm = fs::read_to_string(&"test.qasm").unwrap_or_else(|e| {
        eprintln!("error reading {}: {e}", "test.qasm");
        process::exit(1);
    });

    let circuit = parse(&qasm).unwrap_or_else(|e| {
        eprintln!("parse error: {e}");
        process::exit(1);
    });
    circuit
}
fn ppm_demo() -> Result<(), Box<dyn Error>> {
    let (memory_block, processor_block, magic_block) = match std::env::args().nth(1) {
        Some(name) => {
            let arch = Architecture::preset(&name)
                .ok_or_else(|| format!("unknown architecture preset `{name}`"))?;
            eprintln!(
                "Building CSS blocks for `{name}` via qldpc \
                 (memory={}, processor={}, magic={}); the large memory code is slow...",
                arch.memory_code, arch.processor_code, arch.magic_code,
            );
            arch.css_blocks()?
        }
        None => {
            eprintln!("Building small bb18 [[248, 10, <=18]] test blocks via qldpc...");
            (css_block("bb18")?, css_block("bb18")?, css_block("bb18")?)
        }
    };

    let circ = demo_ppm_circuit();
    eprintln!(
        "Lowering {} Pauli-product measurement(s) to a physical (stim) circuit...",
        circ.instructions.len(),
    );
    let physical = ppc_to_physical_circuit(
        &circ,
        &memory_block,
        &processor_block,
        &magic_block,
        DISTANCE,
    )?;
    print!("{physical}");
    Ok(())
}

fn circuit_demo() -> Result<(), Box<dyn std::error::Error>> {
    let arch = Architecture::preset("small").unwrap();
    let circ = demo_circuit();
    eprintln!(
        "Lowering a {}-gate logical circuit to a physical (stim) circuit over the (memory={}, processor={}, magic={})...",
        circ.gates.len(),
        arch.memory_code,
        arch.processor_code,
        arch.magic_code,
    );
    let physical = logical_circuit_to_physical_circuit(circ, &arch, DISTANCE)?;
    print!("{physical}");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    circuit_demo()
}
