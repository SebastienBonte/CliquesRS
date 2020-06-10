use clap::{crate_authors, crate_name, crate_version, App, Arg};
use cliques_rs::bron_kerbosch::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

const FUNCS: &[(&str, Algorithm)] = &[
    ("BK-BasicMPI", basic_mpi),
    ("BK-BasicMPIWithPivot", basic_mpi_pivot),
];

fn test_alg_mpi(nodes: Arc<Graph>, fun: Algorithm, options: Options) -> Duration {
    let start = Instant::now();
    let p: Set = nodes.nodes().cloned().collect();
    let _cliques = fun(nodes, &mut vec![], p, Set::default(), options);
    start.elapsed()
}

fn main() {
    let matches = App::new(crate_name!())
        .version(crate_version!())
        .author(crate_authors!())
        .about("Find maximal cliques using different algorithms")
        .arg(
            Arg::with_name("input-file")
                .short('i')
                .long("input-file")
                .value_name("FILE")
                .about("Set input-file")
                .required(true),
        )
        .arg(
            Arg::with_name("alg")
                .short('a')
                .long("algorithm")
                .about("Choose which algorithm to run")
                .takes_value(true)
                .use_delimiter(true)
                .required(true)
                .possible_values(&FUNCS.iter().map(|(n, _)| n).copied().collect::<Vec<_>>()),
        )
        .arg("-v, --verbose")
        .get_matches();

    let file: &str = matches.value_of("input-file").expect("Missing input file");
    let mut options = Options::default();
    options.verbose = matches.is_present("verbose");

    let graph = Arc::new(Graph::from_file(file));
    if options.verbose {
        println!("{:?}", graph);
    }
    let value = matches.value_of("alg").unwrap();

    if let Some((name, alg)) = FUNCS.iter().find(|(name, _)| &value == name) {
        println!("{}:{:?}", name, test_alg_mpi(graph, *alg, options));
    }
}
