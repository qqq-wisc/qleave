"""Helper invoked by code_to_log_ops.rs to compute a logical Pauli basis for a
CSS code via qldpc's BP+OSD logical-operator reduction.

It is compiled into the Rust binary (`include_str!`) and run as
`python -c <this file>`, with the parity-check matrices fed on stdin. The
interpreter is chosen by Rust from the QLDPC_PYTHON env var (default `python3`);
that environment must have `qldpc` installed.

Wire protocol — whitespace-separated integers throughout:

  stdin    rows cols            # h_x: `rows` checks over `cols` physical qubits
           <rows*cols bits>     # row-major 0/1
           rows cols            # h_z, same layout
           <rows*cols bits>

  stdout   rows cols            # logical-ops matrix: rows = 2k, cols = 2n
           <rows*cols bits>     # cols 0..n = X-type support, n..2n = Z-type
"""

import sys

import numpy as np
from qldpc.codes import CSSCode


def read_matrix(tokens):
    rows = int(next(tokens))
    cols = int(next(tokens))
    data = [int(next(tokens)) for _ in range(rows * cols)]
    return np.array(data, dtype=int).reshape(rows, cols)


def main():
    tokens = iter(sys.stdin.read().split())
    h_x = read_matrix(tokens)
    h_z = read_matrix(tokens)

    code = CSSCode(h_x, h_z)
    code.reduce_logical_ops(with_BP_OSD=True, osd_method="osd_cs", osd_order=10)
    ops = np.asarray(code.get_logical_ops()).astype(int)

    rows, cols = ops.shape
    sys.stdout.write(f"{rows} {cols}\n")
    sys.stdout.write(" ".join(str(b) for b in ops.reshape(-1)))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
