pub struct Architecture{
    pub memory_capacity : usize,
    pub processor_capacity : usize
}

pub const SPACE_EFFICIENT_LP_20 : Architecture = Architecture{memory_capacity : 1124, processor_capacity : 10 };
pub const SPACE_EFFICIENT_LP_24 : Architecture =Architecture{memory_capacity : 1480, processor_capacity : 10 };
pub const BALANCED_LP_20 : Architecture = Architecture{memory_capacity : 1124, processor_capacity : 148 };
pub const BALANCED_LP_24 : Architecture =Architecture{memory_capacity : 1480, processor_capacity : 148 };