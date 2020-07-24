use cliques_rs::bron_kerbosch::*;
use criterion::measurement::WallTime;
use criterion::{criterion_group, criterion_main, BenchmarkGroup, Criterion};
use std::sync::Arc;

/// List of implementations
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

/// Benchmark function
fn bencher(
    c: &mut BenchmarkGroup<WallTime>,
    fun: Algorithm,
    name: &str,
    file: &str,
    nb_threads: usize,
) {
    let graph = Arc::new(Graph::from_file(file));
    let r: Set = graph.nodes().copied().collect();
    let mut options = Options::default();
    options.num_threads = nb_threads;
    c.bench_function(name, |b| {
        b.iter(|| {
            fun(
                graph.clone(),
                &mut Default::default(),
                r.clone(),
                Set::default(),
                options,
            );
        })
    });
}

/// Specialised benchmark function for sequential algorithms
fn seq(b: &mut Criterion, file: &str, name: &str) {
    let mut group = b.benchmark_group(format!("{}/{}/1", "Sequential", name));
    for (name, fun, _) in FUNCS.iter().filter(|(_, _, p)| !*p) {
        bencher(&mut group, *fun, name, file, 1);
    }
    group.finish();
}

/// Specialised benchmark function for parallel algorithms
fn par(b: &mut Criterion, file: &str, name: &str) {
    let nb_threads = std::env::var("RAYON_NUM_THREADS")
        .unwrap_or_else(|_| "1".to_owned())
        .parse::<usize>()
        .unwrap();
    let mut group = b.benchmark_group(format!("{}/{}/{}", "Parallel", name, nb_threads));
    for (name, fun, _) in FUNCS.iter().filter(|(_, _, p)| *p) {
        bencher(&mut group, *fun, name, file, nb_threads);
    }
    group.finish();
}

fn seq_small(b: &mut Criterion) {
    seq(b, "data/small.txt", "small");
}

fn par_small(b: &mut Criterion) {
    par(b, "data/small.txt", "small");
}

fn seq_medium(b: &mut Criterion) {
    seq(b, "data/com-amazon.ungraph.txt", "medium");
}

fn par_medium(b: &mut Criterion) {
    par(b, "data/com-amazon.ungraph.txt", "medium");
}

fn seq_big(b: &mut Criterion) {
    seq(b, "data/com-youtube.ungraph.txt", "big");
}

fn par_big(b: &mut Criterion) {
    par(b, "data/com-youtube.ungraph.txt", "big");
}

criterion_group!(benches, seq_small, par_small, seq_medium, par_medium, seq_big, par_big);
criterion_main!(benches);
