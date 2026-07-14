"""Helper invoked by circuit_to_checks.rs to construct the CSS parity-check
matrices (h_x, h_z) of a named code block via the qldpc package.

It is compiled into the Rust binary (`include_str!`) and run as
`python -c <this file>`, with the code name fed on stdin. The interpreter is
chosen by Rust from the QLDPC_PYTHON env var (default `python3`); that
environment must have `qldpc` installed.

The named codes are the lifted-product (LP) and bivariate bicycle (BB) codes of
Cain et al., "Shor's algorithm is possible with as few as 10,000 reconfigurable
atomic qubits" (arXiv:2603.28627v1), Appendix A:

  bb18      [[248,  10, <=18]]  BB,  a = 1 + x^6 y + x^27, b = y^2 + x^15 y^3 + x^24, (l,m)=(31,4)
  gross     [[144,  12,   12]]  BB,  a = x^3 + y + y^2, b = y^3 + x + x^2, (l,m)=(12,6)
                                     IBM "gross code" (Bravyi et al. 2024); GeneCS benchmark
  two_gross  [[288, 12,   18]]  BB,  a = x^3 + y^2 + y^7, b = y^3 + x + x^2, (l,m)=(12,12)
                                     IBM "two-gross code" (Bravyi et al. 2024); GeneCS benchmark
  lp3_5_20  [[1122, 148, <=20]] LP,  3x5 seed over F2[x]/(x^33 + 1)   (processor, "balanced")
  lp3_7_16  [[2610, 744, <=16]] LP,  3x7 seed over F2[x]/(x^45 + 1)
  lp3_7_20  [[4350, 1224,<=20]] LP,  3x7 seed over F2[x]/(x^75 + 1)   (memory, lp20)
  lp3_7_24  [[5278, 1480,<=24]] LP,  3x7 seed over F2[x]/(x^91 + 1)   (memory, lp24)

Each LP code is LP(A, A^dagger) for the seed matrix A whose (i, j) entry is the
monomial x^e for the exponent e tabulated below; qldpc's single-argument LPCode
reproduces the paper's [[n, k]] for every entry of LP_CODES.

Wire protocol -- whitespace-separated tokens:

  stdin   <code name>                # one token, e.g. "bb18"

  stdout  rows cols                  # h_x: `rows` X-checks over `cols` physical qubits
          <rows*cols bits>           # row-major 0/1
          rows cols                  # h_z, same layout
          <rows*cols bits>
"""

import sys

import numpy as np

# name -> (lift order ell, exponent matrix). A[i, j] = x ** exps[i][j].
LP_CODES = {
    "lp3_5_20": (
        33,
        [[0, 0, 0, 0, 0],
         [0, 14, 19, 11, 26],
         [0, 13, 2, 15, 21]],
    ),
    "lp3_7_16": (
        45,
        [[29, 21, 31, 15, 37, 25, 27],
         [13, 25, 19, 26, 11, 18, 29],
         [31, 2, 27, 32, 41, 41, 18]],
    ),
    "lp3_7_20": (
        75,
        [[0, 71, 73, 68, 33, 50, 47],
         [38, 39, 60, 26, 18, 1, 23],
         [73, 6, 5, 42, 20, 22, 73]],
    ),
    "lp3_7_24": (
        91,
        [[57, 75, 42, 80, 7, 67, 27],
         [57, 73, 34, 12, 27, 50, 87],
         [21, 53, 70, 18, 1, 3, 18]],
    ),
}


def lp_code(ell, exps):
    from qldpc import abstract
    from qldpc.codes import LPCode

    group = abstract.CyclicGroup(ell)
    x = group.generators[0]
    data = np.array([[x ** e for e in row] for row in exps], dtype=object)
    return LPCode(abstract.RingArray.build(data, group))


def bb18():
    import sympy
    from qldpc.codes import BBCode

    x, y = sympy.symbols("x y")
    a = 1 + x ** 6 * y + x ** 27
    b = y ** 2 + x ** 15 * y ** 3 + x ** 24
    return BBCode({x: 31, y: 4}, a, b)


def gross():
    import sympy
    from qldpc.codes import BBCode

    x, y = sympy.symbols("x y")
    a = x ** 3 + y + y ** 2
    b = y ** 3 + x + x ** 2
    return BBCode({x: 12, y: 6}, a, b)

def two_gross():
    import sympy
    from qldpc.codes import BBCode

    x, y = sympy.symbols("x y")
    a = x ** 3 + y ** 2 + y ** 7
    b = y ** 3 + x + x ** 2
    return BBCode({x: 12, y: 12}, a, b)


def build_code(name):
    if name == "bb18":
        return bb18()
    if name == "gross":
        return gross()
    if name == "two_gross":
        return two_gross()
    if name in LP_CODES:
        ell, exps = LP_CODES[name]
        return lp_code(ell, exps)
    raise SystemExit(f"unknown code name: {name!r}")


def write_matrix(out, matrix):
    matrix = np.asarray(matrix).astype(int) % 2
    rows, cols = matrix.shape
    out.write(f"{rows} {cols}\n")
    out.write(" ".join(str(b) for b in matrix.reshape(-1)))
    out.write("\n")


def main():
    name = sys.stdin.read().split()
    if not name:
        raise SystemExit("expected a code name on stdin")
    code = build_code(name[0])
    write_matrix(sys.stdout, code.matrix_x)
    write_matrix(sys.stdout, code.matrix_z)


if __name__ == "__main__":
    main()
