#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[global_allocator]
#[cfg(feature = "mimalloc")]
static GLOBAL: MiMalloc = MiMalloc;

pub mod bron_kerbosch;
