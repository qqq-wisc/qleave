use crate::circuit_to_checks::{CSSCodeBlock, css_block};

/// A target architecture from Cain et al. (arXiv:2603.28627v1): a large memory
/// code block, a smaller processor block, and a magic-state factory block. Each
/// block is named by the qldpc-constructible code that realizes it (see
/// [`css_block`]); `*_capacity` is that code's number of logical qubits `k`.
#[derive(Debug, Clone, Copy)]
pub struct Architecture {
    pub memory_capacity: usize,
    pub processor_capacity: usize,
    /// Memory block code name, e.g. `"lp3_7_20"`.
    pub memory_code: &'static str,
    /// Processor block code name, e.g. `"bb18"` or `"lp3_5_20"`.
    pub processor_code: &'static str,
    /// Magic-state factory block code name (the `bb18` factory code).
    pub magic_code: &'static str,
}

impl Architecture {
    /// Construct the three CSS code blocks (memory, processor, magic) for this
    /// architecture via qldpc. Building the large memory codes runs qldpc's
    /// logical-operator reduction and can be slow.
    pub fn css_blocks(&self) -> std::io::Result<(CSSCodeBlock, CSSCodeBlock, CSSCodeBlock)> {
        Ok((   
            css_block(self.memory_code)?,
            css_block(self.processor_code)?,
            css_block(self.magic_code)?,
        ))
    }

    /// Look up a preset by its CLI name (e.g. `"space-efficient-lp20"`).
    pub fn preset(name: &str) -> Option<&'static Architecture> {
        match name {
            "space-efficient-lp20" => Some(&SPACE_EFFICIENT_LP_20),
            "space-efficient-lp24" => Some(&SPACE_EFFICIENT_LP_24),
            "balanced-lp20" => Some(&BALANCED_LP_20),
            "balanced-lp24" => Some(&BALANCED_LP_24),
            "small" => Some(&SMALL),
            "gross" => Some(&GROSS),
            _ => None,
        }
    }
}

// Memory codes: lp3_7_20 = [[4350, 1224, <=20]], lp3_7_24 = [[5278, 1480, <=24]].
// Processor codes: bb18 = [[248, 10, <=18]] (space-efficient),
//                  lp3_5_20 = [[1122, 148, <=20]] (balanced).
// Magic/factory code: bb18 in every architecture.

pub const SPACE_EFFICIENT_LP_20: Architecture = Architecture {
    memory_capacity: 1224,
    processor_capacity: 10,
    memory_code: "lp3_7_20",
    processor_code: "bb18",
    magic_code: "bb18",
};
pub const SPACE_EFFICIENT_LP_24: Architecture = Architecture {
    memory_capacity: 1480,
    processor_capacity: 10,
    memory_code: "lp3_7_24",
    processor_code: "bb18",
    magic_code: "bb18",
};
pub const BALANCED_LP_20: Architecture = Architecture {
    memory_capacity: 1224,
    processor_capacity: 148,
    memory_code: "lp3_7_20",
    processor_code: "lp3_5_20",
    magic_code: "bb18",
};
pub const BALANCED_LP_24: Architecture = Architecture {
    memory_capacity: 1480,
    processor_capacity: 148,
    memory_code: "lp3_7_24",
    processor_code: "lp3_5_20",
    magic_code: "bb18",
};

pub const SMALL: Architecture = Architecture {
    memory_capacity: 10,
    processor_capacity: 10,
    memory_code: "bb18",
    processor_code: "bb18",
    magic_code: "bb18",
};

pub const LP_20_GROSS: Architecture = Architecture {
    memory_capacity: 12,
    processor_capacity: 12,
    memory_code: "lp3_7_20",
    processor_code: "gross",
    magic_code: "gross",
};

pub const LP_20_2GROSS: Architecture = Architecture {
    memory_capacity: 12,
    processor_capacity: 12,
    memory_code: "lp3_7_20",
    processor_code: "two_gross",
    magic_code: "two_gross",
};


// IBM gross code [[144, 12, 12]] in every block — for benchmarking single-operator
// surgery against GeneCS (arXiv:2605.21746) Table 2, which reports 24 ancilla
// qubits for this code.
pub const GROSS: Architecture = Architecture {
    memory_capacity: 12,
    processor_capacity: 12,
    memory_code: "gross",
    processor_code: "gross",
    magic_code: "gross",
};

// IBM gross code [[288, 12, 18]] in every block 
pub const TWO_GROSS: Architecture = Architecture {
    memory_capacity: 12,
    processor_capacity: 12,
    memory_code: "two_gross",
    processor_code: "two_gross",
    magic_code: "two_gross",
};

