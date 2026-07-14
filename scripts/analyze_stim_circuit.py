"""Analyze and decode a stim circuit.

The circuits produced by this compiler are qLDPC-style: stabilizers are measured
with weight-6 MPP instructions and there are several logical observables. That
makes them non-graphlike, so MWPM (pymatching) is not appropriate -- we use
BP+OSD from the `ldpc` package instead.

The compiled circuit itself is noiseless, so on its own the detector error model
is empty and there is nothing to decode. To get a meaningful logical error rate
we inject a phenomenological noise model (measurement-flip + depolarizing noise)
before building the detector error model.
"""

import argparse

import numpy as np
import stim
from scipy.sparse import csc_matrix


def parse_circuit(filename):
    with open(filename, "r") as f:
        return stim.Circuit(f.read())


# Measurement / reset gate names we know how to make noisy.
_MEASURE_GATES = {"M", "MX", "MY", "MZ", "MR", "MRX", "MRY", "MRZ", "MPP"}
_RESET_GATES = {"R", "RX", "RY", "RZ"}
# Two-qubit gates emitted by the ancilla syndrome-extraction lowering
# (`--ancilla-extraction`). Note `CNOT` is excluded: the compiler also uses it for
# the rec-controlled byproduct corrections, whose targets pair a measurement record
# with a qubit rather than two qubits -- those are classical feedforward, not a
# physical two-qubit gate, so they get no gate noise.
_TWO_QUBIT_GATES = {"CX", "CY", "CZ"}


def add_noise(circuit, p_meas, p_data, p_two=0.0):
    """Return a copy of `circuit` with a phenomenological noise model applied.

    - Every measurement gets a `p_meas` probability of reporting a flipped result.
    - Every reset is followed by a `DEPOLARIZE1(p_data)` on the reset qubits.
    - Each MPP block is preceded by `DEPOLARIZE1(p_data)` on its data qubits,
      acting as one round of data noise.
    - Every two-qubit gate (`CX`/`CY`/`CZ`, from `--ancilla-extraction`) is
      followed by a `DEPOLARIZE2(p_two)` on each control/target pair.
    """
    noisy = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            body = add_noise(inst.body_copy(), p_meas, p_data, p_two)
            noisy.append(stim.CircuitRepeatBlock(inst.repeat_count, body))
            continue

        name = inst.name
        targets = inst.targets_copy()

        if name == "MPP" and p_data > 0:
            # Data noise on the qubits involved in this MPP round.
            qubits = sorted({t.value for t in targets if t.is_qubit_target})
            if qubits:
                noisy.append("DEPOLARIZE1", qubits, p_data)

        if name in _MEASURE_GATES and p_meas > 0:
            # Re-emit the measurement with a flip probability argument.
            noisy.append(name, targets, p_meas)
        else:
            noisy.append(name, targets, inst.gate_args_copy())

        if name in _RESET_GATES and p_data > 0:
            qubits = [t.value for t in targets if t.is_qubit_target]
            if qubits:
                noisy.append("DEPOLARIZE1", qubits, p_data)

        if name in _TWO_QUBIT_GATES and p_two > 0:
            # Two-qubit gates carry their targets in (control, target) pairs; add
            # DEPOLARIZE2 on each pair where both endpoints are qubits.
            for i in range(0, len(targets) - 1, 2):
                a, b = targets[i], targets[i + 1]
                if a.is_qubit_target and b.is_qubit_target:
                    noisy.append("DEPOLARIZE2", [a.value, b.value], p_two)

    return noisy


def dem_to_matrices(dem):
    """Convert a detector error model into (H, L, priors).

    H: (num_detectors x num_errors) sparse parity check matrix.
    L: (num_observables x num_errors) sparse observable matrix.
    priors: per-error prior probabilities.
    """
    det_rows, det_cols = [], []
    obs_rows, obs_cols = [], []
    priors = []

    col = 0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                det_rows.append(t.val)
                det_cols.append(col)
            elif t.is_logical_observable_id():
                obs_rows.append(t.val)
                obs_cols.append(col)
        priors.append(p)
        col += 1

    num_errors = col
    H = csc_matrix(
        (np.ones(len(det_rows), dtype=np.uint8), (det_rows, det_cols)),
        shape=(dem.num_detectors, num_errors),
    )
    L = csc_matrix(
        (np.ones(len(obs_rows), dtype=np.uint8), (obs_rows, obs_cols)),
        shape=(dem.num_observables, num_errors),
    )
    return H, L, np.array(priors)


def strip_observables(circuit):
    """Return a copy of `circuit` with all OBSERVABLE_INCLUDE instructions removed."""
    stripped = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name == "OBSERVABLE_INCLUDE":
            continue
        stripped.append(inst)
    return stripped


def build_dem(circuit):
    """Build a detector error model, returning (dem, has_observables).

    Compiled algorithm circuits (vs. memory experiments) generally have
    non-deterministic observables -- the logical output isn't pinned to a fixed
    value, so stim can't define a logical error rate against it. The detectors
    are still deterministic, so in that case we fall back to a detector-only DEM
    and report syndrome-decoding statistics instead of a logical error rate.
    """
    kwargs = dict(decompose_errors=False, flatten_loops=True)
    try:
        return circuit.detector_error_model(**kwargs), True
    except ValueError as e:
        if "non-deterministic observable" not in str(e):
            raise
        print(
            "Observables are non-deterministic (this looks like an algorithm "
            "circuit, not a memory experiment).\n"
            "Falling back to detector-only syndrome decoding."
        )
        return strip_observables(circuit).detector_error_model(**kwargs), False


def make_decoder(H, priors, max_iter, osd_order):
    from ldpc.bposd_decoder import BpOsdDecoder

    return BpOsdDecoder(
        H,
        error_channel=list(priors),
        max_iter=max_iter,
        bp_method="minimum_sum",
        ms_scaling_factor=0.625,
        osd_method="osd_cs",
        osd_order=osd_order,
    )


def decode(circuit, shots, max_iter, osd_order, seed):
    dem, has_observables = build_dem(circuit)
    H, L, priors = dem_to_matrices(dem)
    print(
        f"DEM: {dem.num_detectors} detectors, {dem.num_observables} observables, "
        f"{H.shape[1]} error mechanisms"
    )
    if H.shape[1] == 0:
        print("No error mechanisms in the DEM -- did you set a nonzero noise rate?")
        return

    decoder = make_decoder(H, priors, max_iter, osd_order)

    if has_observables:
        L = L.toarray().astype(np.uint8)
        sampler = circuit.compile_detector_sampler(seed=seed)
        det_data, obs_data = sampler.sample(shots, separate_observables=True)

        logical_errors = 0
        for i in range(shots):
            syndrome = det_data[i].astype(np.uint8)
            correction = decoder.decode(syndrome)
            predicted_obs = (L @ correction) % 2
            actual_obs = obs_data[i].astype(np.uint8)
            if not np.array_equal(predicted_obs, actual_obs):
                logical_errors += 1

        print(f"Shots: {shots}")
        print(f"Logical errors (any observable): {logical_errors}")
        print(f"Logical error rate: {logical_errors / shots:.4g}")
    else:
        # No usable observables: report how well BP+OSD explains the syndrome.
        sampler = strip_observables(circuit).compile_detector_sampler(seed=seed)
        det_data = sampler.sample(shots)

        fully_explained = 0
        triggered_total = 0
        residual_total = 0
        for i in range(shots):
            syndrome = det_data[i].astype(np.uint8)
            triggered_total += int(syndrome.sum())
            correction = decoder.decode(syndrome)
            residual = (syndrome ^ ((H @ correction) % 2).astype(np.uint8))
            residual_weight = int(residual.sum())
            residual_total += residual_weight
            if residual_weight == 0:
                fully_explained += 1

        print(f"Shots: {shots}")
        print(f"Mean triggered detectors / shot: {triggered_total / shots:.2f}")
        print(f"Mean residual (unexplained) detectors / shot: {residual_total / shots:.4g}")
        print(
            f"Syndromes fully explained by decoder: {fully_explained}/{shots} "
            f"({fully_explained / shots:.4g})"
        )
        print(
            "Note: no logical error rate -- the circuit's observables are "
            "non-deterministic. Decode a memory-experiment circuit for that."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", nargs="?", default="out.stim")
    parser.add_argument("--p-meas", type=float, default=1e-3,
                        help="measurement flip probability")
    parser.add_argument("--p-data", type=float, default=1e-3,
                        help="data depolarizing probability")
    parser.add_argument("--p-two", type=float, default=0.0,
                        help="two-qubit (CX/CY/CZ) depolarizing probability "
                             "(for --ancilla-extraction circuits)")
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=30,
                        help="BP iterations before OSD")
    parser.add_argument("--osd-order", type=int, default=0,
                        help="OSD order (0 = fast OSD-0; higher is slower)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    circuit = parse_circuit(args.circuit)
    print(
        f"Loaded {args.circuit}: {circuit.num_qubits} qubits, "
        f"{circuit.num_measurements} measurements"
    )

    noisy = add_noise(circuit, args.p_meas, args.p_data, args.p_two)
    shortest = noisy.shortest_graphlike_error()
    print(f"shortest graphlike error: {shortest[0].dem_error_terms}")
    print(f"distance: {len(shortest)}")
    with open("noisy.stim", "w") as f:
        f.write(str(noisy))
    with open("dem.txt", "w") as f:
        f.write(str(noisy.detector_error_model()))
    # decode(noisy, args.shots, args.max_iter, args.osd_order, args.seed)


if __name__ == "__main__":
    main()
