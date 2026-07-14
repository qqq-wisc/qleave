"""Diagnose the circuit-level distance of a compiled stim memory experiment.

Where `analyze_stim_circuit.py` decodes a circuit and reports a logical error
rate, this script answers a sharper question: *why* is the distance what it is,
and *which physical errors* limit it. For each run it reports

  1. the circuit distance under a phenomenological noise model, via
     `search_for_undetectable_logical_errors` (works on the non-graphlike DEMs these
     weight-6 MPP codes produce, unlike `shortest_graphlike_error`);
  2. the physical error(s) realizing the shortest undetectable logical error --
     for each, whether it is a data Pauli (which Pauli, which qubit) or a
     measurement flip (which record, what it measured), and the instruction it
     sits on; and
  3. every detector-error-model mechanism that flips a logical observable while
     triggering NO detector -- the genuinely undetectable logical errors that
     pin the distance -- grouped by observable and sorted by probability.

Iterating tip: run once at the default `--p-meas`/`--p-data`, then again with
`--p-meas 0` to isolate the *data-error* channels from the *measurement-flip*
channels (a measurement-only failure disappears when measurement noise is off).
The noise model is shared with `analyze_stim_circuit.add_noise`, so the two
scripts always agree on what "noisy" means.

Examples:
    python3 scripts/diagnose_distance.py out.stim
    python3 scripts/diagnose_distance.py test.stim --p-meas 0
    python3 scripts/diagnose_distance.py mem.stim --top 20 --save
"""

import argparse
import os
import sys
from collections import defaultdict

import stim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_stim_circuit import add_noise  # noqa: E402  (shared noise model)


def undetectable_observable_errors(dem):
    """DEM error mechanisms that flip >=1 observable and zero detectors.

    These are the undetectable logical errors: nothing in the syndrome reveals
    them, so each one is a weight-`len(circuit_error_locations)` path to a logical
    flip. stim merges all errors with an identical (detectors, observables)
    signature, so undetectable errors sharing an observable set collapse to a
    single mechanism whose probability is their combined weight.

    Returns a list of (probability, frozenset_of_observable_ids).
    """
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        detectors = 0
        observables = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                detectors += 1
            elif t.is_logical_observable_id():
                observables.append(t.val)
        if observables and detectors == 0:
            out.append((inst.args_copy()[0], frozenset(observables)))
    return out


def describe_error_location(loc):
    """One-line human description of a stim CircuitErrorLocation."""
    parts = []
    for p in loc.flipped_pauli_product:
        gt = p.gate_target
        pauli = getattr(gt, "pauli_type", "?")
        qubit = getattr(gt, "qubit_value", getattr(gt, "value", "?"))
        parts.append(f"data {pauli} on qubit {qubit}")
    fm = loc.flipped_measurement
    if fm is not None:
        measured = ", ".join(
            f"{getattr(g.gate_target, 'pauli_type', '?')}"
            f"{getattr(g.gate_target, 'qubit_value', '?')}"
            for g in (fm.observable or ())
        )
        parts.append(f"measurement flip: record {fm.record_index} (measured {measured})")
    try:
        gate = loc.instruction_targets.gate
    except Exception:
        gate = "?"
    body = "; ".join(parts) if parts else "(unknown)"
    return f"[{gate}] {body}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("circuit", help="path to a compiled .stim circuit")
    ap.add_argument("--p-meas", type=float, default=1e-3, help="measurement flip probability")
    ap.add_argument("--p-data", type=float, default=1e-3, help="data depolarizing probability")
    ap.add_argument("--p-two", type=float, default=1e-2,
                    help="two-qubit (CX/CY/CZ) depolarizing probability (for --ancilla-extraction circuits)")
    ap.add_argument("--top", type=int, default=10, help="max undetectable-error groups to list")
    ap.add_argument("--max-det-set", type=int, default=6,
                    help="search bound: max detection-event-set size to explore")
    ap.add_argument("--max-edge-degree", type=int, default=6,
                    help="search bound: max detectors a single error may flip")
    ap.add_argument("--exhaustive", action="store_true",
                    help="complete (slower) search: also explore symptom-degree-increasing edges")
    ap.add_argument("--save", action="store_true", help="write noisy.stim and dem.txt")
    args = ap.parse_args()

    circuit = stim.Circuit(open(args.circuit).read())
    print(
        f"Loaded {args.circuit}: {circuit.num_qubits} qubits, "
        f"{circuit.num_measurements} measurements, "
        f"{circuit.num_detectors} detectors, {circuit.num_observables} observables"
    )

    noisy = add_noise(circuit, args.p_meas, args.p_data, args.p_two)
    print(f"Noise model: p_meas={args.p_meas}, p_data={args.p_data}, p_two={args.p_two}")

    # 1 + 2: circuit distance and the physical errors that realize it.
    # search_for_undetectable_logical_errors handles non-graphlike DEMs (unlike
    # shortest_graphlike_error), which is what these weight-6 MPP codes produce. It
    # returns the smallest set of noise mechanisms that forms an undetectable logical
    # error; its size is the circuit distance under this noise model. The --max-*
    # bounds prune the search: too low can miss the true minimum (over-report or
    # fail), so raise them (or pass --exhaustive) if a result looks suspicious.
    print(
        f"\nSearching for the smallest undetectable logical error "
        f"(det-set<={args.max_det_set}, edge-degree<={args.max_edge_degree}, "
        f"{'exhaustive' if args.exhaustive else 'greedy'})..."
    )
    # with open("shortest_error.sat", "w") as f:
    #         print("Writing shortest error to shortest_error.sat")
    #         f.write(noisy.shortest_error_sat_problem())
    try:
        errors = noisy.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=args.max_det_set,
            dont_explore_edges_with_degree_above=args.max_edge_degree,
            dont_explore_edges_increasing_symptom_degree=not args.exhaustive,
            canonicalize_circuit_errors=True,
        )

        print(f"Circuit distance: {len(errors)}")
        for i, e in enumerate(errors):
            for loc in e.circuit_error_locations:
                print(f"  [{i}] {describe_error_location(loc)}")
    except ValueError as ex:
        msg = str(ex)
        if "Failed to find any logical errors" in msg:
            # Not a failure: no undetectable logical error exists within the bounds,
            # so the distance simply exceeds them. Good news for a sound circuit;
            # raise --max-det-set/--max-edge-degree (or --exhaustive) to push higher.
            print(
                f"No undetectable logical error within the bounds -> distance > "
                f"{args.max_det_set} (raise --max-det-set/--max-edge-degree or use "
                f"--exhaustive to pin the exact value; exhaustive is expensive for "
                f"high-distance qLDPC codes)."
            )
        else:
            print(f"search failed: {ex}  (needs >=1 observable)")

    # 3: the undetectable logical errors -- the actual bug surface.
    dem = noisy.detector_error_model(flatten_loops=True)
    bad = undetectable_observable_errors(dem)
    aggregated = defaultdict(lambda: [0.0, 0])
    for prob, observables in bad:
        slot = aggregated[observables]
        slot[0] += prob
        slot[1] += 1
    print(
        f"\nUndetectable logical errors (flip an observable, trigger NO detector): "
        f"{len(bad)} mechanism(s)"
    )
    if not bad:
        print("  none -- every logical error lights a detector. Distance is detector-limited,")
        print("  i.e. genuinely set by the code/rounds rather than a missing-detector bug.")
    for observables, (prob, count) in sorted(
        aggregated.items(), key=lambda kv: -kv[1][0]
    )[: args.top]:
        ids = ",".join(f"L{i}" for i in sorted(observables))
        print(f"  {ids}: {count} mechanism(s), total prob ~{prob:.4g}")

    if args.save:
        with open("noisy.stim", "w") as f:
            f.write(str(noisy))
        with open("dem.txt", "w") as f:
            f.write(str(dem))
        print("\nWrote noisy.stim and dem.txt")


if __name__ == "__main__":
    main()
