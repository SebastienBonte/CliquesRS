use cliques_rs::bron_kerbosch::*;
use std::sync::Arc;

#[test]
fn test_init_graph() {
    let graph = Graph::from_file("./data/minimal.txt");
    assert_eq!(graph.number_nodes(), 12);
    assert_eq!(graph.number_edges(), 10);
}

fn mini_tester(file: &str, count: usize, max_size: usize, fun: Algorithm) {
    let graph = Graph::from_file(file);
    let r: Set = graph.nodes().copied().collect();
    let options = Options::default();
    let cliques = fun(Arc::new(graph), &mut vec![], r, Set::default(), options);
    assert_eq!(cliques.count, count);
    assert_eq!(cliques.max_size, max_size);
}

#[test]
fn test_basic() {
    mini_tester("./data/minimal.txt", 10, 2, basic);
}

#[test]
fn test_basic_iter() {
    mini_tester("./data/minimal.txt", 10, 2, basic_iter);
}

#[test]
#[cfg(feature = "rayon")]
fn test_basic_top_par() {
    mini_tester("./data/minimal.txt", 10, 2, basic_top_par);
}

#[test]
#[cfg(feature = "pool")]
fn test_basic_pool() {
    mini_tester("./data/minimal.txt", 10, 2, basic_pool);
}

#[test]
fn test_basic_pivot() {
    mini_tester("./data/minimal.txt", 10, 2, basic_pivot);
}

#[test]
fn test_basic_iter_pivot() {
    mini_tester("./data/minimal.txt", 10, 2, basic_iter_pivot);
}

#[test]
#[cfg(feature = "rayon")]
fn test_basic_top_par_pivot() {
    mini_tester("./data/minimal.txt", 10, 2, basic_top_par_pivot);
}

#[test]
#[cfg(feature = "pool")]
fn test_basic_pool_pivot() {
    mini_tester("./data/minimal.txt", 10, 2, basic_pool_pivot);
}

#[test]
fn test_basic_ordering() {
    mini_tester("./data/minimal.txt", 10, 2, basic_ordering);
}

#[test]
fn test_basic_iter_ordering() {
    mini_tester("./data/minimal.txt", 10, 2, basic_iter_ordering);
}

#[test]
#[cfg(feature = "rayon")]
fn test_basic_top_par_ordering() {
    mini_tester("./data/minimal.txt", 10, 2, basic_top_par_ordering);
}
