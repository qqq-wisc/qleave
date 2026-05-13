# oratomic-compiler

A compiler targeting a memory/processor architecture like the one proposed in the paper: ["Shor’s algorithm is possible with as few as 10,000 reconfigurable atomic qubits."](https://arxiv.org/abs/2603.28627v1)  Translates a QASM 2.0 circuit to a sequence of Pauli Product Measurements.

## Overview

This compiler faithfully implements the compilation scheme described in Appendix E of the paper as the following sequence of passes.

1. **Partition** the circuit into subcircuits acting on at most ``k`` qubits.

2. Insert **Load** and **Store** operations as appropriate between subcircuits

3. Translate gates and load/store operations in **Pauli Product** rotations and measurements

4. Optionally, randomly resolve any conditional operations 

5. Absorb Clifford rotations into the final measurements of a subcircuit

## Build

```sh
cargo build --release
```

## Usage

```sh
cargo run --release -- <circuit.qasm> [options]
```

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

- `--arch <preset>` — use a named architecture preset.
- `--mem-cap <N> --proc-cap <M>` — custom memory / processor capacity (mutually exclusive with `--arch`).
- `--intermediates [DIR]` — write intermediate circuits (`load_store.txt`, `pbc_w_clifford.txt`, `final_res.txt`) to `DIR` (default: `out/`).
- `--simulate-corrections` — randomly resolve conditional rotations to unconditional ones before applying the Clifford frame.
- `--skip-redundant-ls` — skip redundant `Load`/`Store` pairs when a qubit appears in consecutive subcircuits.
- `--sat` — use the SAT-based optimal partitioner (implies `--skip-redundant-ls`).
- `--sat-timeout <SECS>` — timeout for the SAT optimizer; returns the best solution found so far.

### Example

```sh
cargo run --release -- circuits/qft_16.qasm --arch balanced-lp-20 --intermediates out
```

## Repository layout

- [src/main.rs](src/main.rs) — CLI entry point.
- [src/parse.rs](src/parse.rs) — QASM parser.
- [src/circuit.rs](src/circuit.rs) — gate-level circuit IR and DAG.
- [src/compile.rs](src/compile.rs) — end-to-end compilation pipeline.
- [src/pbc.rs](src/pbc.rs) — Pauli product circuit IR and Clifford frame tracking.
- [src/arch.rs](src/arch.rs) — architecture presets.
- [src/sat_partition.rs](src/sat_partition.rs) — SAT-based optimal partitioner.
- [circuits/](circuits/) — example QASM benchmarks (QFT, adders, Hamming, etc.).

