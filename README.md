# qleave

A compiler targeting quantum device architectures with a large memory code block and smaller processor code block, like the ones proposed in the paper: ["Shor’s algorithm is possible with as few as 10,000 reconfigurable atomic qubits"](https://arxiv.org/abs/2603.28627v1) by Cain et al.

``qleave`` (pronounced like cleave) translates a QASM2 circuit to a sequence of Pauli Product Measurements following the scheme proposed in the paper, with some additional optimizations. 
Use it to get precise resource estimates for arbitrary workloads with explicit instruction sequences. 

## Overview

This compiler implements the compilation scheme described by Cain et al. as the following sequence of passes.

1. **Partition** the circuit into subcircuits acting on at most $k$ qubits, executable on a $[[n,k,d]]$ processor code block.

2. Insert **Load** and **Store** operations as appropriate between subcircuits.

3. Translate gates and load/store operations in **Pauli Product** rotations and measurements.

4. Optionally, randomly resolve any conditional operations. 

5. Absorb Clifford rotations into the final measurements of a subcircuit.

## Optimizations

**Load/Store minimization.** ``qleave`` can optimize the number of inserted load/store instructions, and consequently the total depth of the final Pauli Product Measurement sequence. 

``--skip-redundant-ls``: For lightweight optimization, the ``--skip-redundant-ls`` flag simply leaves qubits that appear in consecutive subcircuits in the processor block, rather than storing and immediately reloading, as suggested in Appendix E of the paper. 

``--sat``: For further reduction, the ``--sat`` flag replaces the default greedy partitioning pass with a SAT-based search. In this mode, ``qleave`` makes iterative calls to the CaDiCal SAT solver to find the circuit partition that minimizes the total load/store count. For large inputs, SAT solving may not terminate within a reasonable time bound, so it's wise to control the time budget with the ``--sat-timeout`` flag. When this timeout expires, the solver is interrupted and returns the best solution found so far.

## Installation

First, [install Rust](https://www.rust-lang.org/tools/install) if you don't already have it. Then:

```sh
cargo install --path .
```

This places a `qleave` binary on your `$PATH` (typically `~/.cargo/bin`).

## Usage

```sh
qleave <circuit.qasm> [options]
```

By default, the compiled circuit is written to `<input_stem>.pbc` in the current directory. Override with `-o <path>`, or use `-o -` to write to stdout (for piping).

### Architecture presets

Targets are described by a `(memory_capacity, processor_capacity)` pair. Built-in presets from the paper:

| Preset                  | Memory | Processor |
| ----------------------- | ------ | --------- |
| `space-efficient-lp-20` | 1124   | 10        |
| `space-efficient-lp-24` | 1480   | 10        |
| `balanced-lp-20`        | 1124   | 148       |
| `balanced-lp-24`        | 1480   | 148       |

Select one with `--arch <preset>`, or specify custom capacities with `--mem-cap <N> --proc-cap <M>`.

### Options

- `-o, --output <PATH>` — output path for the compiled circuit. Use `-` for stdout. Defaults to `<input_stem>.pbc` in the current directory.
- `--arch <preset>` — use a named architecture preset.
- `--mem-cap <N> --proc-cap <M>` — custom memory / processor capacity (mutually exclusive with `--arch`).
- `--intermediates [DIR]` — also write intermediate representations to `DIR` (default: `<input_stem>-intermediates/`).
- `--simulate-corrections` — randomly resolve conditional rotations to unconditional ones before applying the Clifford frame.
- `--skip-redundant-ls` — skip redundant `Load`/`Store` pairs when a qubit appears in consecutive subcircuits.
- `--sat` — use the SAT-based optimal partitioner (implies `--skip-redundant-ls`).
- `--sat-timeout <SECS>` — timeout for the SAT optimizer; returns the best solution found so far.

### Example

```sh
qleave circuits/qft_16.qasm --arch space-efficient-lp-20 --intermediates
```

## Repository layout

- [src/main.rs](src/main.rs) — CLI entry point.
- [src/parse.rs](src/parse.rs) — QASM parser.
- [src/circuit.rs](src/circuit.rs) — gate-level circuit IR and DAG.
- [src/compile.rs](src/compile.rs) — logic for the enumerated compiler passes.
- [src/pbc.rs](src/pbc.rs) — Pauli product circuit IR and Clifford frame tracking.
- [src/arch.rs](src/arch.rs) — architecture presets.
- [src/sat_partition.rs](src/sat_partition.rs) — SAT-based optimal partitioner.
- [circuits/](circuits/) — example QASM benchmarks (QFT, adders, etc.).

