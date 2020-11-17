#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "jemallocator")]
use jemallocator::Jemalloc;

// Use mimalloc as the custom allocator if enabled
#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

// Use mimalloc as the custom allocator if enabled
#[global_allocator]
#[cfg(feature = "jemallocator")]
static GLOBAL: Jemalloc = Jemalloc;

// Make the bron_kerbosch module public as a library
pub mod bron_kerbosch;
