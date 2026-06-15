#!/usr/bin/env python3
"""Count the qubits referenced in a stim-style circuit (e.g. out.circ).

A "qubit" is any qubit index that appears as a gate target: a bare integer
target (R / M / CNOT / ...) or the index of a Pauli term in an MPP product
(`X248`, `Z3`, `!Y7`). Measurement-record (`rec[-N]`) and sweep (`sweep[N]`)
targets are not qubits and are ignored, as are annotation arguments in
parentheses (`OBSERVABLE_INCLUDE(0)`, `DETECTOR(...)`).

Usage:
    scripts/count_qubits.py [FILE]   # defaults to stdin
"""

import re
import sys

# A Pauli target: optional `!` sign, a Pauli letter, then the qubit index.
PAULI = re.compile(r"^!?[XYZ](\d+)$")
# A bare integer qubit target.
BARE = re.compile(r"^\d+$")


def qubits_in_line(line: str):
    """Yield the qubit indices referenced as targets on one circuit line."""
    tokens = line.split()
    if not tokens:
        return
    # tokens[0] is the gate / annotation name (may carry `(args)`); skip it.
    for tok in tokens[1:]:
        if tok.startswith("rec[") or tok.startswith("sweep["):
            continue  # measurement-record / sweep-bit target, not a qubit
        # An MPP target is a `*`-joined product of Pauli terms; plain gates have
        # one part per token, so splitting on `*` handles both uniformly.
        for part in tok.split("*"):
            m = PAULI.match(part)
            if m:
                yield int(m.group(1))
            elif BARE.match(part):
                yield int(part)
            # anything else (stray annotation, unknown syntax) is ignored


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    f = open(path) if path else sys.stdin
    try:
        qubits = set()
        for line in f:
            qubits.update(qubits_in_line(line))
    finally:
        if path:
            f.close()

    if not qubits:
        print("0 qubits (no qubit targets found)")
        return

    distinct = len(qubits)
    hi = max(qubits)
    contiguous = distinct == hi + 1 and min(qubits) == 0
    print(f"distinct qubit indices: {distinct}")
    print(f"max index: {hi}  ->  {hi + 1} qubits if 0..{hi} are all allocated")
    if not contiguous:
        print(f"(indices are not a gap-free 0..{hi} range)")


if __name__ == "__main__":
    main()
