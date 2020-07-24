use clap::{crate_authors, crate_name, crate_version, App, Arg};
use cliques_rs::bron_kerbosch::*;
use std::sync::Arc;

/// List of all implementations
const FUNCS: &[(&str, Algorithm, bool)] = &[
    ("BK-Basic", basic, false),
    ("BK-BasicWithPivot", basic_pivot, false),
    ("BK-BasicWithOrdering", basic_ordering, false),
    ("BK-BasicIterative", basic_iter, false),
    ("BK-BasicIterWithPivot", basic_iter_pivot, false),
    ("BK-BasicIterWithOrdering", basic_iter_ordering, false),
    #[cfg(feature = "rayon")]
    ("BK-BasicParallel", basic_top_par, true),
    #[cfg(feature = "rayon")]
    ("BK-BasicParWithPivot", basic_top_par_pivot, true),
    #[cfg(feature = "rayon")]
    ("BK-BasicParWithOrdering", basic_top_par_ordering, true),
    #[cfg(feature = "pool")]
    ("BK-BasicPool", basic_pool, true),
    #[cfg(feature = "pool")]
    ("BK-BasicPoolWithPivot", basic_pool_pivot, true),
];

fn main() {
    //  Used for arguments parsing and usage
    let matches = App::new(crate_name!())
        .version(crate_version!()).author(crate_authors!())
        .about("Find maximal cliques using different algorithms")
        .arg(Arg::with_name("input-file").short('i').long("input-file")
            .value_name("file").about("Set input-file. Use standard input if absent"))
        .arg(Arg::with_name("alg").short('a').long("algorithm")
            .about("Choose which algorithm to run").takes_value(true)
            .multiple(true).use_delimiter(true)
            .possible_values(&FUNCS.iter().map(|(n, _, _)| n).copied().collect::<Vec<_>>()))
        .arg("-t, --threads [threads] 'Choose number of threads (> 0) for compatible algorithms (Defaults to number of processor cores)'")
        .arg("-k, --max-cliques [clique size] 'Search maximum cliques of specific size'")
        .arg(Arg::with_name("num").short('n').long("repeat")
            .about("Run a specific number of times (default: 1)").value_name("number"))         
        .arg("-v, --verbose")
        .get_matches();

    let mut options = Options::default();
    options.verbose = matches.is_present("verbose");
    if let Ok(max) = matches.value_of_t("max-cliques") {
        options.clique_size = Some(max);
    }
    if matches.is_present("threads") {
        let nb: usize = matches.value_of_t_or_exit("threads");
        assert_ne!(nb, 0, "Cannot have 0 threads");
        options.num_threads = nb;
        #[cfg(feature = "rayon")]
        rayon::ThreadPoolBuilder::new()
            .num_threads(nb)
            .build_global()
            .unwrap();
    }

    let graph: Arc<Graph> = match matches.value_of("input-file") {
        Some(file) => Arc::new(Graph::from_file(file)),
        None => Arc::new(Graph::from_stdin()),
    };
    let repeat: usize = matches.value_of_t("num").unwrap_or(1);
    if options.verbose {
        println!("{:?}", graph);
    }
    match matches.values_of("alg") {
        Some(values) => {
            let values: Vec<_> = values.collect();
            for (name, alg, _) in FUNCS.iter().filter(|(name, _, _)| values.contains(name)) {
                // Run each algorithm passed as argument n times
                let mut out = String::from(*name);
                for _ in 0..repeat {
                    out += &format!(";{}", test_alg(graph.clone(), *alg, options).as_secs_f64());
                    // Returns time take by algorithm
                }
                println!("{}", out);
            }
        }
        None => {
            for (name, alg, _) in FUNCS {
                // Run every algorithm n times
                let mut out = String::from(*name);
                for _ in 0..repeat {
                    out += &format!(";{}", test_alg(graph.clone(), *alg, options).as_secs_f64());
                    // Returns time take by algorithm
                }
                println!("{}", out);
            }
        }
    };
}
