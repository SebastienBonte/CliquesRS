#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

// Use mimalloc as the custom allocator if enabled
#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

// Make the bron_kerbosch module public as a library
pub mod bron_kerbosch;
