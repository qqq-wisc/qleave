# the standard way to import PySAT:
from pysat.formula import CNF, IDPool
from pysat.solvers import Solver
from pysat.card import *
import re
import threading
import time

def parse_qasm(path):
    '''parse an OpenQASM 2.0 file and return a list of tuples of qubit indices, one per gate'''
    gates = []
    qubit_index = {}  # name+index -> flat int
    next_idx = [0]

    def get_idx(name, i):
        key = (name, i)
        if key not in qubit_index:
            qubit_index[key] = next_idx[0]
            next_idx[0] += 1
        return qubit_index[key]

    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(';')
            if not line or line.startswith('OPENQASM') or line.startswith('include') or line.startswith('//'):
                continue
            if line.startswith('qreg') or line.startswith('creg'):
                continue
            # extract qubit arguments: everything after the gate name
            args_str = re.sub(r'^[a-z_A-Z0-9]+\s+', '', line)
            qubits = tuple(
                get_idx(m.group(1), int(m.group(2)))
                for m in re.finditer(r'(\w+)\[(\d+)\]', args_str)
            )
            if qubits:
                gates.append(qubits)
    return gates

def gate_list_to_dag(gates):
    '''create a list of edges, where an element (u,v) means gate v is a direct successor of u in the dag defined by the gate list'''
    edges = set()
    last_on_qubit = {}  # qubit -> index of most recent gate using it
    for i, gate in enumerate(gates):
        for q in gate:
            if q in last_on_qubit:
                edges.add((last_on_qubit[q], i))
            last_on_qubit[q] = i
    return list(edges)

def all_gates_executed(gates, max_subcircuits, vpool):
    clauses = []
    for i, gate in enumerate(gates):
        lits = [encode_lit((1, "g", i, s), vpool) for s in range(max_subcircuits)]
        exact_one = CardEnc.equals(lits=lits, bound=1, vpool=vpool)
        clauses.extend(exact_one.clauses)
    return clauses

def order_preserved(edges, max_subcircuits, vpool):
    clauses = []
    for (u,v) in edges:
        for i in range(max_subcircuits):
            for j in range(i):
                lits = [(-1, "g", u, i), (-1, "g", v, j)]
                clause = [encode_lit(lit, vpool) for lit in lits]
                clauses.append(clause)
    return clauses


def gates_executable(gates, max_subcircuits, vpool):
    clauses = []
    for index, gate in enumerate(gates):
        for i in range(max_subcircuits):
            for q in gate:
                lits = [(-1, "g", index, i), (1, "m", q, i)]
                clause = [encode_lit(lit, vpool) for lit in lits]
                clauses.append(clause)
    return clauses


def max_capacity(qubits, processor_capacity, max_subcircuits, vpool):
    clauses = []
    for c in range(max_subcircuits):
        lits = [encode_lit((1, "m", q, c), vpool) for q in range(len(qubits))]
        at_most_k = CardEnc.atmost(lits, bound=processor_capacity, vpool=vpool)
        clauses.extend(at_most_k.clauses)
    return clauses

def must_load_store(qubits, max_subcircuits, vpool):
    clauses = []
    for q in range(len(qubits)):
        lits = [(-1, "m", q, 0), (1, "l", q, 0)]
        clause = [encode_lit(lit, vpool) for lit in lits]
        clauses.append(clause)
    for i in range(max_subcircuits-1):
        for q in range(len(qubits)):
            load_lits = [(1, "m", q, i), (-1, "m", q, i+1), (1, "l", q, i+1)]
            load_clause = [encode_lit(lit, vpool) for lit in load_lits]
            clauses.append(load_clause)
            store_lits = [(-1, "m", q, i), (1, "m", q, i+1), (1, "s", q, i+1)]
            store_clause = [encode_lit(lit, vpool) for lit in store_lits]
            clauses.append(store_clause)
    return clauses



def encode_lit(lit, vpool : IDPool):
    return lit[0]*vpool.id(lit[1:])

def no_gaps(gates, max_subcircuits, vpool):
    clauses = []
    n_gates = len(gates)
    # backward: gate(i,j) → used[j]
    for j in range(max_subcircuits):
        for i in range(n_gates):
            clauses.append([encode_lit((-1, "g", i, j), vpool), encode_lit((1, "u", j), vpool)])
    # forward: used[j] → ∨_i gate(i,j)
    for j in range(max_subcircuits):
        clause = [encode_lit((-1, "u", j), vpool)] + [encode_lit((1, "g", i, j), vpool) for i in range(n_gates)]
        clauses.append(clause)
    # no-gaps: used[j+1] → used[j]
    for j in range(max_subcircuits - 1):
        clauses.append([encode_lit((-1, "u", j+1), vpool), encode_lit((1, "u", j), vpool)])
    return clauses

def encode_cnf(gates, processor_capacity, max_subcircuit_count, vpool):
    qubits = set(q for g in gates for q in g)

    edges = gate_list_to_dag(gates)
    gates_executed_clauses  = all_gates_executed(gates, max_subcircuit_count, vpool)
    order_preserved_clauses = order_preserved(edges, max_subcircuit_count, vpool)
    gates_executable_clauses = gates_executable(gates, max_subcircuit_count, vpool)
    capacity_clauses = max_capacity(qubits, processor_capacity, max_subcircuit_count, vpool)
    load_store_clauses = must_load_store(qubits, max_subcircuit_count, vpool)
    no_gaps_clauses = no_gaps(gates, max_subcircuit_count, vpool)
    return gates_executed_clauses + order_preserved_clauses + capacity_clauses + gates_executable_clauses + load_store_clauses + no_gaps_clauses

def solve(path, processor_capacity, max_subcircuit_count):
    vpool = IDPool()
    gates = parse_qasm(path)
    clauses = encode_cnf(gates, processor_capacity, max_subcircuit_count, vpool)
    solver = Solver("cd19", bootstrap_with=clauses)
    res = solver.solve()
    if res:
        raw_model = solver.get_model()
        decoded_model = decode(raw_model, vpool)
        print(decoded_model)
    else: 
        print(res)

def optimize(path, processor_capacity, max_subcircuit_count, ubound=None, timeout=None):
    vpool = IDPool()
    gates = parse_qasm(path)
    qubit_count = len(set(q for g in gates for q in g))

    clauses = encode_cnf(gates, processor_capacity, max_subcircuit_count, vpool)
    solver = Solver("g4", bootstrap_with=clauses)
    load_lits = [encode_lit((1, "l", q, i), vpool) for q in range(qubit_count) for i in range(max_subcircuit_count)]
    store_lits = [encode_lit((1, "s", q, i), vpool) for q in range(qubit_count) for i in range(1, max_subcircuit_count)]
    lits = load_lits + store_lits
    if ubound == None:
        ubound = len(lits)
    itot = ITotalizer(lits=lits, ubound=ubound, top_id=vpool.top)
    vpool.top = itot.top_id
    solver.append_formula(itot.cnf.clauses)

    # itot.rhs[k] is the literal whose negation enforces "sum <= k"
    # i.e., assuming -itot.rhs[k] means "at most k of op_vars are true"

    best_model = None
    best_k = ubound
    deadline = time.time() + timeout if timeout is not None else None

    while True:
        print(best_k)
        assumption = -itot.rhs[best_k - 1]

        timer = None
        if deadline is not None:
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"timeout: best found = {best_k}")
                break
            timer = threading.Timer(remaining, solver.interrupt)
            timer.start()

        sat = solver.solve_limited(assumptions=[assumption])

        if timer is not None:
            timer.cancel()
            solver.clear_interrupt()

        if sat is True:
            model = solver.get_model()
            k = sum(1 for v in load_store_vars(model, vpool) if v[0] == 1)
            best_model, best_k = model, k
        else:
            if sat is None:
                print(f"timeout: best found = {best_k}")
            break

    itot.delete()
    print(len([v for v in load_store_vars(best_model, vpool) if v[0] == 1]))
        

def decode(raw_model, vpool : IDPool):
    decoded = []
    for lit in raw_model:
        if vpool.obj(lit):
                decoded.append((1, *vpool.obj(lit)))
        elif vpool.obj(-lit):
                decoded.append((-1, *vpool.obj(-lit)))
    return decoded

def load_store_vars(raw_model, vpool):
    return [v for v in decode(raw_model, vpool) if v[1] in ['l', 's']]



if __name__ == "__main__":
    optimize("../circuits/gf_2^10_mult.qasm", 10, 30, timeout=60, ubound=278)
