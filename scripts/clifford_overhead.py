#!/usr/bin/env python3
"""Measure the instruction-count overhead of the --explicit-clifford flag.

For every circuit in circuits/, compiles it with and without
--explicit-clifford, parses the "Wrote N instructions" line from qleave, and
writes the results to clifford_overhead.csv at the repo root.

Usage:
    python3 scripts/clifford_overhead.py
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CIRCUITS_DIR = REPO / "circuits"
OUT_CSV = REPO / "clifford_overhead.csv"

WROTE_RE = re.compile(r"Wrote\s+(\d+)\s+instructions")


def compile_count(circuit: Path, explicit: bool) -> int:
    """Run qleave in compile mode and return the instruction count."""
    cmd = [
        "cargo", "run", "--release", "--quiet", "--bin", "qleave",
        "--", "--mode", "compile",
    ]
    if explicit:
        cmd.append("--explicit-clifford")
    cmd.append(str(circuit))

    proc = subprocess.run(
        cmd, cwd=REPO, capture_output=True, text=True,
    )
    output = proc.stdout + proc.stderr
    matches = WROTE_RE.findall(output)
    if not matches:
        raise RuntimeError(
            f"could not find instruction count for {circuit.name} "
            f"(explicit={explicit}); exit={proc.returncode}\n{output}"
        )
    # take the last match in case of multiple lines
    return int(matches[-1])


def main() -> int:
    circuits = sorted(CIRCUITS_DIR.glob("*.qasm"))
    if not circuits:
        print(f"no circuits found in {CIRCUITS_DIR}", file=sys.stderr)
        return 1

    rows = []
    for circuit in circuits:
        name = circuit.name
        print(f"compiling {name} ...", flush=True)
        try:
            baseline = compile_count(circuit, explicit=False)
            explicit = compile_count(circuit, explicit=True)
        except RuntimeError as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            rows.append([name, "", "", "", ""])
            continue

        overhead = explicit - baseline
        ratio = explicit / baseline if baseline else ""
        print(
            f"  baseline={baseline} explicit={explicit} "
            f"overhead={overhead} ratio={ratio:.3f}" if ratio != "" else
            f"  baseline={baseline} explicit={explicit} overhead={overhead}"
        )
        rows.append([
            name, baseline, explicit, overhead,
            f"{ratio:.4f}" if ratio != "" else "",
        ])

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "circuit", "baseline_instructions",
            "explicit_clifford_instructions", "overhead", "ratio",
        ])
        writer.writerows(rows)

    print(f"\nwrote results to {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
