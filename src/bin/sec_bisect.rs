//! Debug harness: lower a hand-picked PPM sequence through the surgery pipeline
//! and report the compiled observable count, to bisect which measurement types
//! mis-lower. Compare against the logical-level expectation (scratch
//! `expected_obs.py`). Usage: `sec_bisect <variant>`.

use qleave::{
    arch::SMALL,
    checks_to_physical_circuit::compile_memory_experiment,
    circuit_to_checks::{
        CodeData, pauli_product_circuit_to_physical_supports,
        physical_supports_to_stabilizer_checks,
    },
    graph_construction::SurgeryGraphConfig,
    pbc::{
        ArchitectureQubit, Pauli, PauliAxis, PauliProductCircuit, PauliProductOperation,
        PauliString, Sign,
    },
};

fn meas(c: &mut PauliProductCircuit, sign: Sign, sites: Vec<(ArchitectureQubit, Pauli)>) {
    let id = c.allocate_meas_id();
    c.instructions.push(PauliProductOperation::Measurement {
        axis: PauliAxis { sign, pauli_string: PauliString::new(sites) },
        id,
    });
}

fn main() {
    use ArchitectureQubit::{Memory as M, Processor as P};
    use Pauli::*;
    let variant = std::env::args().nth(1).expect("usage: sec_bisect <variant> [x|z]");
    let basis = match std::env::args().nth(2).as_deref() {
        Some("x") => X,
        _ => Z,
    };

    let mut c = PauliProductCircuit::new();
    let load = |c: &mut PauliProductCircuit| {
        meas(c, Sign::One, vec![(M(0), Z), (P(0), Z)]);
        meas(c, Sign::One, vec![(M(0), X)]);
    };
    let store = |c: &mut PauliProductCircuit| {
        meas(c, Sign::One, vec![(M(0), Z), (P(0), Z)]);
        meas(c, Sign::One, vec![(P(0), X)]);
    };
    load(&mut c);
    match variant.as_str() {
        "load-store" => {}
        "g1" => {
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), X)]);
        }
        "g2" => {
            meas(&mut c, Sign::NegOne, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
        }
        "g2plus" => {
            meas(&mut c, Sign::One, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
        }
        "zy-only" => {
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
        }
        "bare-xy" => {
            meas(&mut c, Sign::One, vec![(P(0), X), (P(9), Y)]);
        }
        "xy-z9" => {
            meas(&mut c, Sign::One, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
        }
        "x9-xy" => {
            meas(&mut c, Sign::One, vec![(P(9), X)]);
            meas(&mut c, Sign::One, vec![(P(0), X), (P(9), Y)]);
        }
        "xprep-g2plus" => {
            meas(&mut c, Sign::One, vec![(P(9), X)]);
            meas(&mut c, Sign::One, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
        }
        "xprep-zy" => {
            meas(&mut c, Sign::One, vec![(P(9), X)]);
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
        }
        "xprep-g2" => {
            meas(&mut c, Sign::One, vec![(P(9), X)]);
            meas(&mut c, Sign::NegOne, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
        }
        "zprep-g1" => {
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), X)]);
        }
        "g1g2" => {
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), X)]);
            meas(&mut c, Sign::NegOne, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
        }
        "g2g3" => {
            meas(&mut c, Sign::NegOne, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), X)]);
        }
        "full" => {
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), X)]);
            meas(&mut c, Sign::NegOne, vec![(P(0), X), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), Z)]);
            meas(&mut c, Sign::One, vec![(P(0), Z), (P(9), Y)]);
            meas(&mut c, Sign::One, vec![(P(9), X)]);
        }
        v => panic!("unknown variant {v}"),
    }
    store(&mut c);

    let blocks = SMALL.css_blocks().expect("css blocks");
    let codes = CodeData::from_blocks(&blocks.0, &blocks.1, &blocks.2, 2000).expect("code data");
    let supports = pauli_product_circuit_to_physical_supports(&c, &codes);
    let mut config = SurgeryGraphConfig::default();
    config.caching = std::env::var("SEC_BISECT_NO_CACHE").is_err();
    let checks = physical_supports_to_stabilizer_checks(&supports, &codes, 3, &config);
    // Any pair of anticommuting checks inside one deformation round is an extra
    // projection: the merge would not realize the intended logical measurement.
    for (d, def) in checks.deformations.iter().enumerate() {
        let mut bad = 0;
        for i in 0..def.checks.len() {
            for j in i + 1..def.checks.len() {
                if !qleave::pbc::axes_commute(
                    &def.checks[i].pauli_string,
                    &def.checks[j].pauli_string,
                ) {
                    bad += 1;
                    if bad <= 3 {
                        eprintln!("deformation {d}: checks {i} and {j} ANTICOMMUTE");
                    }
                }
            }
        }
        eprintln!("deformation {d}: {} checks, {bad} anticommuting pairs", def.checks.len());
    }
    let memory = compile_memory_experiment(checks, 1, &codes, basis, true);
    let stim = memory.flatten().to_string();
    if let Ok(path) = std::env::var("SEC_BISECT_DUMP") {
        std::fs::write(&path, &stim).expect("dump write");
        eprintln!("dumped MPP-level circuit to {path}");
    }
    let obs = stim.lines().filter(|l| l.starts_with("OBSERVABLE")).count();
    println!("{variant} ({basis:?} basis): compiled observables = {obs}");
}
