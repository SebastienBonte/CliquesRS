#[derive(Copy, Clone)]
pub struct Options {
    pub num_threads: usize,
    pub verbose: bool,
    pub clique_size: Option<usize>
}

impl Options {
    pub fn new() -> Self {
        Options {
            num_threads: 1,
            verbose: false,
            clique_size: None
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Options {
            #[cfg(feature = "pool")]
            num_threads: num_cpus::get(),
            #[cfg(not(feature = "pool"))]
            num_threads: 1,
            verbose: false,
            clique_size: None
        }
    }
}
