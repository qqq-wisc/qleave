//! OpenQASM 2.0 parser and serializer.

use std::io::BufRead;

use crate::circuit::{Circuit, Gate, Qubit};

pub fn parse(qasm: &str) -> Result<Circuit, String> {
    let qasm = strip_block_comments(qasm);
    let mut registers: Vec<(String, usize, usize)> = Vec::new(); // (name, offset, size)
    let mut num_qubits: usize = 0;
    let mut seen_gate = false;
    let mut c = Circuit::new(0);
    for (line_num, raw_line) in qasm.lines().enumerate() {
        let line_num = line_num + 1;
        // Strip the line comment before splitting on ';', otherwise a semicolon
        // inside a comment splits it into bogus statements.
        let raw_line = strip_line_comment(raw_line);
        for line in raw_line
            .split(';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            if line.is_empty()
                || line.starts_with("OPENQASM")
                || line.starts_with("include")
                || line.starts_with("barrier")
            {
                continue;
            }
            if line.starts_with("qreg") {
                if seen_gate {
                    return Err(format!("line {line_num}: qreg declaration after gate"));
                }
                let rest = line[4..].trim();
                if let (Some(bracket), Some(end)) = (rest.find('['), rest.find(']')) {
                    let name = rest[..bracket].trim().to_string();
                    let size: usize = rest[bracket + 1..end]
                        .parse()
                        .map_err(|e| format!("line {line_num}: bad qreg size: {e}"))?;
                    registers.push((name, num_qubits, size));
                    num_qubits += size;
                }
            } else {
                seen_gate = true;
                parse_gate_stmt(line, line_num, &registers, &mut c)?;
            }
        }
    }
    c.num_qubits = num_qubits;
    Ok(c)
}

pub fn serialize(circuit: &Circuit) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    writeln!(s, "OPENQASM 2.0;").unwrap();
    writeln!(s, "include \"qelib1.inc\";").unwrap();
    writeln!(s, "qreg q[{}];", circuit.num_qubits).unwrap();
    for gate in &circuit.gates {
        match gate {
            Gate::X(q) => writeln!(s, "x q[{}];", q.0),
            Gate::Y(q) => writeln!(s, "y q[{}];", q.0),
            Gate::H(q) => writeln!(s, "h q[{}];", q.0),
            Gate::S(q) => writeln!(s, "s q[{}];", q.0),
            Gate::Sdg(q) => writeln!(s, "sdg q[{}];", q.0),
            Gate::Z(q) => writeln!(s, "z q[{}];", q.0),
            Gate::T(q) => writeln!(s, "t q[{}];", q.0),
            Gate::Tdg(q) => writeln!(s, "tdg q[{}];", q.0),
            Gate::CNOT { control, target } => writeln!(s, "cx q[{}],q[{}];", control.0, target.0),
            Gate::CCZ {
                control1,
                control2,
                target,
            } => writeln!(
                s,
                "ccz q[{}],q[{}],q[{}];",
                control1.0, control2.0, target.0
            ),
        }
        .unwrap();
    }
    s
}

fn parse_gate_stmt(
    line: &str,
    line_num: usize,
    registers: &[(String, usize, usize)],
    c: &mut Circuit,
) -> Result<(), String> {
    if let Some(rest) = line.strip_prefix("cx ") {
        let q = resolve_qubits(rest, registers, line_num)?;
        c.apply(Gate::CNOT {
            control: q[0],
            target: q[1],
        });
    } else if let Some(rest) = line.strip_prefix("ccx ") {
        let q = resolve_qubits(rest, registers, line_num)?;
        c.apply(Gate::H(q[2]));
        c.apply(Gate::CCZ {
            control1: q[0],
            control2: q[1],
            target: q[2],
        });
        c.apply(Gate::H(q[2]));
    } else if let Some(rest) = line.strip_prefix("cz ") {
        let q = resolve_qubits(rest, registers, line_num)?;
        c.apply(Gate::H(q[1]));
        c.apply(Gate::CNOT {
            control: q[0],
            target: q[1],
        });
        c.apply(Gate::H(q[1]));
    } else if let Some(rest) = line.strip_prefix("h ") {
        c.apply(Gate::H(resolve_qubits(rest, registers, line_num)?[0]));
    } else if let Some(rest) = line.strip_prefix("x ") {
        c.apply(Gate::X(resolve_qubits(rest, registers, line_num)?[0]));
    } else if let Some(rest) = line.strip_prefix("y ") {
        c.apply(Gate::Y(resolve_qubits(rest, registers, line_num)?[0]));
    } else if let Some(rest) = line.strip_prefix("s ") {
        c.apply(Gate::S(resolve_qubits(rest, registers, line_num)?[0]));
    } else if let Some(rest) = line.strip_prefix("sdg ") {
        c.apply(Gate::Sdg(resolve_qubits(rest, registers, line_num)?[0]));
    } else if let Some(rest) = line.strip_prefix("tdg ") {
        c.apply(Gate::Tdg(resolve_qubits(rest, registers, line_num)?[0]));
    } else if let Some(rest) = line.strip_prefix("z ") {
        c.apply(Gate::Z(resolve_qubits(rest, registers, line_num)?[0]));
    } else if let Some(rest) = line.strip_prefix("t ") {
        c.apply(Gate::T(resolve_qubits(rest, registers, line_num)?[0]));
    } else {
        return Err(format!("line {line_num}: unsupported: {line}"));
    }
    Ok(())
}

/// Streaming QASM reader that yields gates in batches without loading the whole file into memory.
pub struct StreamingReader<R: BufRead> {
    reader: R,
    pub num_qubits: usize,
    registers: Vec<(String, usize, usize)>,
    line_num: usize,
    in_block_comment: bool,
    done: bool,
    leftover: Vec<Gate<Qubit>>,
}

impl<R: BufRead> StreamingReader<R> {
    pub fn new(mut reader: R) -> Result<Self, String> {
        let mut num_qubits = 0usize;
        let mut registers: Vec<(String, usize, usize)> = Vec::new();
        let mut line_num = 0;
        let mut in_block_comment = false;
        let mut line_buf = String::new();

        loop {
            line_buf.clear();
            let bytes = reader
                .read_line(&mut line_buf)
                .map_err(|e| format!("I/O error: {e}"))?;
            if bytes == 0 {
                if registers.is_empty() {
                    return Err("no qreg declaration found".into());
                }
                return Ok(StreamingReader {
                    reader,
                    num_qubits,
                    registers,
                    line_num,
                    in_block_comment,
                    done: true,
                    leftover: Vec::new(),
                });
            }
            line_num += 1;

            let line = Self::strip_block_comment_line(&line_buf, &mut in_block_comment);
            let line = strip_line_comment(&line).trim().to_string();
            if line.is_empty() {
                continue;
            }

            let mut leftover_c = Circuit::new(0);
            let mut hit_gate = false;

            for stmt in line.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                if stmt.is_empty()
                    || stmt.starts_with("OPENQASM")
                    || stmt.starts_with("include")
                    || stmt.starts_with("barrier")
                {
                    continue;
                }
                if !hit_gate && stmt.starts_with("qreg") {
                    let rest = stmt[4..].trim();
                    if let (Some(bracket), Some(end)) = (rest.find('['), rest.find(']')) {
                        let name = rest[..bracket].trim().to_string();
                        let size: usize = rest[bracket + 1..end]
                            .parse()
                            .map_err(|e| format!("line {line_num}: bad qreg size: {e}"))?;
                        registers.push((name, num_qubits, size));
                        num_qubits += size;
                    }
                    continue;
                }
                hit_gate = true;
                parse_gate_stmt(stmt, line_num, &registers, &mut leftover_c)?;
            }

            if hit_gate {
                if registers.is_empty() {
                    return Err("no qreg declaration found".into());
                }
                return Ok(StreamingReader {
                    reader,
                    num_qubits,
                    registers,
                    line_num,
                    in_block_comment,
                    done: false,
                    leftover: leftover_c.gates,
                });
            }
        }
    }

    /// Read the next batch of up to `batch_size` gates. Returns `None` at EOF.
    pub fn next_batch(&mut self, batch_size: usize) -> Result<Option<Vec<Gate<Qubit>>>, String> {
        if self.done {
            return Ok(None);
        }
        let mut gates = if !self.leftover.is_empty() {
            std::mem::take(&mut self.leftover)
        } else {
            Vec::with_capacity(batch_size.min(1_000_000))
        };
        let mut line_buf = String::new();

        while gates.len() < batch_size {
            line_buf.clear();
            let bytes = self
                .reader
                .read_line(&mut line_buf)
                .map_err(|e| format!("I/O error: {e}"))?;
            if bytes == 0 {
                self.done = true;
                break;
            }
            self.line_num += 1;
            let line_num = self.line_num;

            let line = Self::strip_block_comment_line(&line_buf, &mut self.in_block_comment);
            let line = strip_line_comment(&line).trim().to_string();
            if line.is_empty() {
                continue;
            }

            for stmt in line.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()) {
                if stmt.is_empty()
                    || stmt.starts_with("OPENQASM")
                    || stmt.starts_with("include")
                    || stmt.starts_with("barrier")
                    || stmt.starts_with("qreg")
                {
                    continue;
                }
                let mut tmp = Circuit::new(0);
                parse_gate_stmt(stmt, line_num, &self.registers, &mut tmp)?;
                gates.extend(tmp.gates);
            }
        }

        if gates.is_empty() {
            Ok(None)
        } else {
            Ok(Some(gates))
        }
    }

    fn strip_block_comment_line(line: &str, in_comment: &mut bool) -> String {
        let mut out = String::with_capacity(line.len());
        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if *in_comment {
                if i + 1 < bytes.len() && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    *in_comment = false;
                    i += 2;
                } else {
                    i += 1;
                }
            } else if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                *in_comment = true;
                i += 2;
            } else {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
        out
    }
}

/// Truncate a line at its `//` comment marker.
fn strip_line_comment(line: &str) -> &str {
    match line.find("//") {
        Some(pos) => &line[..pos],
        None => line,
    }
}

fn strip_block_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find("/*") {
        out.push_str(&rest[..start]);
        match rest[start + 2..].find("*/") {
            Some(end) => {
                for c in rest[start..start + 2 + end + 2].chars() {
                    if c == '\n' {
                        out.push('\n');
                    }
                }
                rest = &rest[start + 2 + end + 2..];
            }
            None => {
                for c in rest[start..].chars() {
                    if c == '\n' {
                        out.push('\n');
                    }
                }
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

fn resolve_qubits(
    s: &str,
    registers: &[(String, usize, usize)],
    line_num: usize,
) -> Result<Vec<Qubit>, String> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim().trim_end_matches(';');
        if let (Some(bracket), Some(end)) = (part.find('['), part.find(']')) {
            let name = part[..bracket].trim();
            let idx: usize = part[bracket + 1..end]
                .parse()
                .map_err(|e| format!("line {line_num}: bad qubit index: {e}"))?;
            let (_, offset, size) = registers
                .iter()
                .find(|(n, _, _)| n == name)
                .ok_or_else(|| format!("line {line_num}: unknown register '{name}'"))?;
            if idx >= *size {
                return Err(format!(
                    "line {line_num}: index {idx} out of range for register '{name}' (size {size})"
                ));
            }
            result.push(Qubit(offset + idx));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_from_qasm() {
        let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nz q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 1);
        assert_eq!(c.gates.len(), 1);
        assert!(matches!(&c.gates[0], Gate::Z(Qubit(0))));
    }

    #[test]
    fn sdg_from_qasm() {
        let qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nsdg q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 1);
        assert_eq!(c.gates.len(), 1);
        assert!(matches!(&c.gates[0], Gate::Sdg(Qubit(0))));
    }

    #[test]
    fn z_qasm_roundtrip() {
        let mut c = Circuit::new(2);
        c.apply(Gate::Z(Qubit(0)));
        c.apply(Gate::Z(Qubit(1)));
        let qasm = serialize(&c);
        let c2 = parse(&qasm).unwrap();
        assert_eq!(c2.gates.len(), 2);
        assert!(matches!(&c2.gates[0], Gate::Z(Qubit(0))));
        assert!(matches!(&c2.gates[1], Gate::Z(Qubit(1))));
    }

    #[test]
    fn sdg_qasm_roundtrip() {
        let mut c = Circuit::new(2);
        c.apply(Gate::Sdg(Qubit(0)));
        c.apply(Gate::Sdg(Qubit(1)));
        let qasm = serialize(&c);
        let c2 = parse(&qasm).unwrap();
        assert_eq!(c2.gates.len(), 2);
        assert!(matches!(&c2.gates[0], Gate::Sdg(Qubit(0))));
        assert!(matches!(&c2.gates[1], Gate::Sdg(Qubit(1))));
    }

    #[test]
    fn mixed_gates_qasm_roundtrip() {
        let mut c = Circuit::new(3);
        c.apply(Gate::H(Qubit(0)));
        c.apply(Gate::Z(Qubit(0)));
        c.apply(Gate::Sdg(Qubit(1)));
        c.apply(Gate::S(Qubit(2)));
        c.apply(Gate::T(Qubit(0)));
        c.apply(Gate::Tdg(Qubit(1)));
        c.apply(Gate::CNOT {
            control: Qubit(0),
            target: Qubit(1),
        });
        c.apply(Gate::X(Qubit(2)));
        let qasm = serialize(&c);
        let c2 = parse(&qasm).unwrap();
        assert_eq!(c2.gates.len(), 8);
        assert!(matches!(&c2.gates[1], Gate::Z(Qubit(0))));
        assert!(matches!(&c2.gates[2], Gate::Sdg(Qubit(1))));
    }

    #[test]
    fn z_to_qasm() {
        let mut c = Circuit::new(1);
        c.apply(Gate::Z(Qubit(0)));
        let qasm = serialize(&c);
        assert!(qasm.contains("z q[0];"));
    }

    #[test]
    fn sdg_to_qasm() {
        let mut c = Circuit::new(1);
        c.apply(Gate::Sdg(Qubit(0)));
        let qasm = serialize(&c);
        assert!(qasm.contains("sdg q[0];"));
    }

    #[test]
    fn line_comment_only() {
        let qasm =
            "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n// just a comment\nqreg q[1];\nh q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
    }

    #[test]
    fn semicolon_inside_line_comment() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\n// ancillas; they start in |0>.\nh q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(0))));
    }

    #[test]
    fn semicolon_inside_trailing_line_comment() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nh q[0]; // note; also a semicolon\nt q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 2);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(0))));
        assert!(matches!(&c.gates[1], Gate::T(Qubit(0))));
    }

    #[test]
    fn streaming_semicolon_inside_line_comment() {
        let qasm = "OPENQASM 2.0;\n// ancillas; they start in |0>.\nqreg q[1];\n\
                    // more; commentary\nh q[0];\nt q[0]; // done; really\n";
        let mut r = StreamingReader::new(std::io::Cursor::new(qasm)).unwrap();
        assert_eq!(r.num_qubits, 1);
        let mut gates = Vec::new();
        while let Some(batch) = r.next_batch(64).unwrap() {
            gates.extend(batch);
        }
        assert_eq!(gates.len(), 2);
        assert!(matches!(&gates[0], Gate::H(Qubit(0))));
        assert!(matches!(&gates[1], Gate::T(Qubit(0))));
    }

    #[test]
    fn inline_line_comment() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nh q[0]; // apply hadamard\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(0))));
    }

    #[test]
    fn block_comment_single_line() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\n/* comment */ h q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
    }

    #[test]
    fn block_comment_multiline() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\n/* this is\na multi-line\ncomment */\nh q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
    }

    #[test]
    fn block_comment_inline() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nh /* surprise */ q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
    }

    #[test]
    fn block_comment_between_gates() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\n/* between */\ncx q[0],q[1];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 2);
    }

    #[test]
    fn multiple_block_comments() {
        let qasm = "OPENQASM 2.0;\n/* a */ qreg q[1]; /* b */\n/* c */ h q[0]; /* d */\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
    }

    #[test]
    fn block_comment_spanning_gate() {
        let qasm = "OPENQASM 2.0;\nqreg q[2];\nh q[0];\n/* cx q[0],q[1]; */\nt q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 2);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(0))));
        assert!(matches!(&c.gates[1], Gate::T(Qubit(0))));
    }

    #[test]
    fn block_and_line_comments_mixed() {
        let qasm = "\
OPENQASM 2.0;
qreg q[2];
// line comment
h q[0]; // inline
/* block */ cx q[0],q[1];
/* multi
   line */
t q[0];
";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 3);
    }

    #[test]
    fn unclosed_block_comment_ignores_rest() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nh q[0];\n/* unclosed\nt q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(0))));
    }

    #[test]
    fn empty_block_comment() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\n/**/ h q[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
    }

    #[test]
    fn comment_only_file() {
        let qasm = "OPENQASM 2.0;\n// nothing here\n/* also nothing */\nqreg q[1];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 0);
    }

    #[test]
    fn line_comment_at_end_no_newline() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nh q[0]; // trailing";
        let c = parse(qasm).unwrap();
        assert_eq!(c.gates.len(), 1);
    }

    #[test]
    fn block_comment_preserves_line_numbers() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\n/* skip\nthis\n*/\nh q[0];\nfoo q[0];\n";
        let err = parse(qasm).unwrap_err();
        assert!(
            err.contains("line 7"),
            "expected line 7 in error, got: {err}"
        );
    }

    #[test]
    fn unsupported_gate_error() {
        let qasm = "OPENQASM 2.0;\nqreg q[1];\nry(0.5) q[0];\n";
        let err = parse(qasm).unwrap_err();
        assert!(err.contains("line 3"));
        assert!(err.contains("unsupported"));
        assert!(err.contains("ry"));
    }

    #[test]
    fn two_registers() {
        let qasm = "OPENQASM 2.0;\nqreg a[2];\nqreg b[3];\nh a[0];\nh a[1];\nt b[0];\nt b[2];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 5);
        assert_eq!(c.gates.len(), 4);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(0))));
        assert!(matches!(&c.gates[1], Gate::H(Qubit(1))));
        assert!(matches!(&c.gates[2], Gate::T(Qubit(2))));
        assert!(matches!(&c.gates[3], Gate::T(Qubit(4))));
    }

    #[test]
    fn multi_register_cnot() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nqreg b[1];\ncx a[0],b[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 2);
        assert!(matches!(
            &c.gates[0],
            Gate::CNOT {
                control: Qubit(0),
                target: Qubit(1)
            }
        ));
    }

    #[test]
    fn qreg_after_gate_error() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nh a[0];\nqreg b[1];\n";
        let err = parse(qasm).unwrap_err();
        assert!(err.contains("line 4"));
        assert!(err.contains("qreg declaration after gate"));
    }

    #[test]
    fn unknown_register_error() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nh b[0];\n";
        let err = parse(qasm).unwrap_err();
        assert!(err.contains("unknown register"));
    }

    #[test]
    fn register_index_out_of_range() {
        let qasm = "OPENQASM 2.0;\nqreg a[2];\nh a[5];\n";
        let err = parse(qasm).unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn three_registers_offsets() {
        let qasm = "OPENQASM 2.0;\nqreg x[3];\nqreg y[2];\nqreg z[1];\n\
                     h x[0];\nh x[2];\nh y[0];\nh y[1];\nh z[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 6);
        assert_eq!(c.gates.len(), 5);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(0))));
        assert!(matches!(&c.gates[1], Gate::H(Qubit(2))));
        assert!(matches!(&c.gates[2], Gate::H(Qubit(3))));
        assert!(matches!(&c.gates[3], Gate::H(Qubit(4))));
        assert!(matches!(&c.gates[4], Gate::H(Qubit(5))));
    }

    #[test]
    fn multi_register_ccx() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nqreg b[1];\nqreg c[1];\nccx a[0],b[0],c[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 3);
        assert_eq!(c.gates.len(), 3);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(2))));
        assert!(matches!(
            &c.gates[1],
            Gate::CCZ {
                control1: Qubit(0),
                control2: Qubit(1),
                target: Qubit(2)
            }
        ));
        assert!(matches!(&c.gates[2], Gate::H(Qubit(2))));
    }

    #[test]
    fn multi_register_cz() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nqreg b[1];\ncz a[0],b[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 2);
        assert_eq!(c.gates.len(), 3);
        assert!(matches!(&c.gates[0], Gate::H(Qubit(1))));
        assert!(matches!(
            &c.gates[1],
            Gate::CNOT {
                control: Qubit(0),
                target: Qubit(1)
            }
        ));
        assert!(matches!(&c.gates[2], Gate::H(Qubit(1))));
    }

    #[test]
    fn single_qubit_registers() {
        let qasm = "OPENQASM 2.0;\nqreg r0[1];\nqreg r1[1];\nqreg r2[1];\nqreg r3[1];\n\
                     cx r0[0],r3[0];\nt r2[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 4);
        assert!(matches!(
            &c.gates[0],
            Gate::CNOT {
                control: Qubit(0),
                target: Qubit(3)
            }
        ));
        assert!(matches!(&c.gates[1], Gate::T(Qubit(2))));
    }

    #[test]
    fn multi_register_all_single_qubit_gates() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nqreg b[1];\n\
                     x a[0];\ns b[0];\nsdg a[0];\nz b[0];\ntdg a[0];\nt b[0];\n";
        let c = parse(qasm).unwrap();
        assert_eq!(c.num_qubits, 2);
        assert!(matches!(&c.gates[0], Gate::X(Qubit(0))));
        assert!(matches!(&c.gates[1], Gate::S(Qubit(1))));
        assert!(matches!(&c.gates[2], Gate::Sdg(Qubit(0))));
        assert!(matches!(&c.gates[3], Gate::Z(Qubit(1))));
        assert!(matches!(&c.gates[4], Gate::Tdg(Qubit(0))));
        assert!(matches!(&c.gates[5], Gate::T(Qubit(1))));
    }

    #[test]
    fn qreg_after_gate_on_same_line_error() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\nh a[0]; qreg b[1];\n";
        let err = parse(qasm).unwrap_err();
        assert!(err.contains("qreg declaration after gate"));
    }

    #[test]
    fn register_index_exactly_at_boundary() {
        let qasm = "OPENQASM 2.0;\nqreg a[2];\nh a[2];\n";
        let err = parse(qasm).unwrap_err();
        assert!(err.contains("out of range"));
    }

    #[test]
    fn register_index_max_valid() {
        let qasm = "OPENQASM 2.0;\nqreg a[3];\nh a[2];\n";
        let c = parse(qasm).unwrap();
        assert!(matches!(&c.gates[0], Gate::H(Qubit(2))));
    }

    #[test]
    fn unknown_register_in_cnot() {
        let qasm = "OPENQASM 2.0;\nqreg a[1];\ncx a[0],nosuch[0];\n";
        let err = parse(qasm).unwrap_err();
        assert!(err.contains("unknown register"));
        assert!(err.contains("nosuch"));
    }
}
