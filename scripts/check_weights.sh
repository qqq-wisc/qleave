#!/usr/bin/env bash
#
# Histogram the weights of MPP (measure-Pauli-product) checks in a stim circuit
# emitted by compile_to_physical, split by the three merged-code check families
# (see src/graph_to_checks.rs):
#
#   X-only  -> cycle checks (faces, bounded by the cellulation degree d_c)
#              and modified X-type code checks (code stabilizer + path X edges)
#   Z-only  -> vertex checks; weight == the vertex's degree in the cellulated graph
#   mixed   -> anything carrying both X and Z (or a code-qubit Pauli alongside edges)
#
# Usage: scripts/check_weights.sh [circuit.circ]   (default: out.circ)
set -euo pipefail

circ="${1:-out.circ}"
if [[ ! -f "$circ" ]]; then
  echo "no such circuit file: $circ" >&2
  exit 1
fi

awk '
/^MPP/ {
  line = $0
  n = gsub(/\*/, "*", line) + 1          # factors = (number of "*") + 1
  hasX = (line ~ /X/)
  hasZ = (line ~ /Z/)
  total++
  if (hasX && !hasZ)      { xonly[n]++; if (n > xmax) xmax = n; xn++ }
  else if (hasZ && !hasX) { zonly[n]++; if (n > zmax) zmax = n; zn++ }
  else                    { mixed[n]++; if (n > mmax) mmax = n; mn++ }
}
END {
  printf "MPP checks: %d total (X-only %d, Z-only %d, mixed %d)\n\n", total, xn, zn, mn

  print "X-only (cycle + modified X code checks):"
  for (w = 1; w <= xmax; w++) if (xonly[w]) printf "  w=%-3d %d\n", w, xonly[w]
  printf "  max weight: %d\n\n", xmax

  print "Z-only (vertex checks; weight == vertex degree):"
  for (w = 1; w <= zmax; w++) if (zonly[w]) printf "  w=%-3d %d\n", w, zonly[w]
  printf "  max weight: %d\n", zmax

  if (mn) {
    print "\nmixed:"
    for (w = 1; w <= mmax; w++) if (mixed[w]) printf "  w=%-3d %d\n", w, mixed[w]
    printf "  max weight: %d\n", mmax
  }
}
' "$circ"
