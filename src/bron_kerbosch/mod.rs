use std::sync::Arc;
#[cfg(feature = "rayon")]
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[cfg(feature = "pool")]
use crossbeam_channel::{bounded, Sender};
#[cfg(feature = "mpi")]
use mpi::{initialize, traits::*};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
#[cfg(feature = "pool")]
use threadpool::ThreadPool;

pub use types::*;

mod types;

/// Returns a copy of set P as a vector
#[inline]
fn get_list(_: Arc<Graph>, p: &Set, _: &Set) -> Vec<Node> {
    p.iter().cloned().collect()
}

/// Return the candidates of P that are not neighbours of pivot
#[inline]
fn get_pivot_list(graph: Arc<Graph>, p: &Set, x: &Set) -> Vec<Node> {
    let pivot = p
        .union(x)
        .max_by_key(|x| graph[x].intersection(p).count())
        .unwrap();
    p.difference(&graph[pivot]).cloned().collect()
}
/// Naive implementation of the Bron-Kerbosch algorithm
pub fn basic(
    graph: Arc<Graph>,
    r: &mut Vec<Node>,
    mut p: Set,
    mut x: Set,
    opts: Options,
) -> Report {
    let mut cliques = Report::default(); // Initialise the report
    if p.is_empty() && x.is_empty() {
        // If P and X are empty, report R as clique
        cliques.count = 1;
        if opts.clique_size.is_none() || opts.clique_size.unwrap() >= r.len() {
            cliques.max_size = r.len();
            cliques.cliques = Some(vec![r.clone()]);
        }
    } else if !p.is_empty() {
        for node in get_list(graph.clone(), &p, &x) {
            // For each vertex v in P
            let neighbours: &Set = &graph[&node];
            r.push(node);
            cliques += basic(graph.clone(), r, &p & neighbours, &x & neighbours, opts);
            // Recursion with the & operator performing set intersection, also combines reports
            r.pop();
            p.remove(&node);
            x.insert(node);
        }
    }
    cliques // Returns the report
}

/// Pivot implementation of the Bron-Kerbosch algorithm
pub fn basic_pivot(
    graph: Arc<Graph>,
    r: &mut Vec<Node>,
    mut p: Set,
    mut x: Set,
    opts: Options,
) -> Report {
    let mut cliques = Report::default();
    if p.is_empty() && x.is_empty() {
        cliques.count = 1;
        if opts.clique_size.is_none() || opts.clique_size.unwrap() >= r.len() {
            cliques.max_size = r.len();
            cliques.cliques = Some(vec![r.clone()]);
        }
    } else if !p.is_empty() {
        for node in get_pivot_list(graph.clone(), &p, &x) {
            let neighbours: &Set = &graph[&node];
            r.push(node);
            cliques += basic_pivot(graph.clone(), r, &p & neighbours, &x & neighbours, opts);
            r.pop();
            p.remove(&node);
            x.insert(node);
        }
    }
    cliques
}

/// Returns the list of all vertices ordered by degeneracy
fn degeneracy_ordering(graph: Arc<Graph>, opts: Options) -> Vec<Node> {
    let start = Instant::now();
    let mut output: Vec<Node> = Vec::with_capacity(graph.number_nodes());
    let mut output_set: Set = Default::default();
    let mut degrees: Map<Node, usize> = Default::default();
    let mut degeneracy: Map<usize, Set> = Default::default();

    output_set.reserve(graph.number_nodes());

    for (k, v) in graph.items() {
        let degree = v.len();
        degrees.insert(*k, degree);
        degeneracy.entry(degree).or_default().insert(*k);
    }

    while let Some((_, v)) = degeneracy.iter_mut().find(|(_, v)| !v.is_empty()) {
        v.shrink_to_fit();
        let x = *v.iter().next().unwrap();
        let node = v.take(&x).unwrap();
        output.push(node);
        output_set.insert(node);
        for w in graph[&node].difference(&output_set) {
            let degree = degrees[w];
            degeneracy.get_mut(&degree).unwrap().remove(&w);
            if degree > 0 {
                degrees.insert(*w, degree - 1);
                degeneracy.entry(degree - 1).or_default().insert(*w);
            }
        }
    }
    output.reverse();
    if opts.verbose {
        println!("Degeneracy computed in {:?}", start.elapsed());
    }
    output
}

/// Implementation of the Bron-Kerbosch algorithm using degeneracy ordering
pub fn basic_ordering(
    graph: Arc<Graph>,
    r: &mut Vec<Node>,
    mut p: Set,
    mut x: Set,
    opts: Options,
) -> Report {
    let mut cliques = Default::default();
    for n in degeneracy_ordering(graph.clone(), opts) {
        let neighbours: &Set = &graph[&n];
        r.push(n);
        cliques += basic_pivot(graph.clone(), r, &p & neighbours, &x & neighbours, opts); // Call the pivot version for recursion
        r.pop();
        p.remove(&n);
        x.insert(n);
    }
    cliques
}

/// Base skeleton for the iterative version of the Bron-Kerbosch algorithm
#[inline]
fn base_iter(
    graph: Arc<Graph>,
    r: &mut Vec<Node>,
    mut stack: Vec<(Set, Set, Vec<Node>)>,
    f: Generator,
    opts: Options,
) -> Report {
    let mut cliques = Report::default();
    while let Some((mut p, mut x, mut nodes)) = stack.pop() {
        // While the stack is not empty
        match nodes.pop() {
            // Check if the pop'ed state still contains candidates
            Some(node) => {
                // Perfom algorithm
                let neighbours: &Set = &graph[&node];
                let (np, nx) = (&p & neighbours, &x & neighbours);
                p.remove(&node);
                x.insert(node);
                if np.is_empty() && nx.is_empty() {
                    if r.len() + 1 >= cliques.max_size
                        && (opts.clique_size.is_none() || opts.clique_size.unwrap() > r.len())
                    {
                        r.push(node);
                        cliques += Report {
                            count: 1,
                            max_size: r.len(),
                            cliques: Some(vec![r.clone()]),
                        };
                        r.pop();
                    } else {
                        cliques.count += 1;
                    }
                    stack.push((p, x, nodes)); // Push state back for recursion after reporting
                } else {
                    // Entering false recursion, adding two scopes to the stack
                    let nn = f(graph.clone(), &np, &nx);
                    let new = (np, nx, nn);
                    r.push(node);
                    stack.push((p, x, nodes));
                    stack.push(new);
                }
            }
            None => {
                // No candidates, backtrack
                r.pop();
            }
        }
    }
    cliques // Returns report
}

/// Naive & Iterative version of the Bron-Kerbosch algorithm
pub fn basic_iter(graph: Arc<Graph>, r: &mut Vec<Node>, p: Set, x: Set, opts: Options) -> Report {
    let nn = get_list(graph.clone(), &p, &x);
    base_iter(graph, r, vec![(p, x, nn)], get_list, opts)
}

/// Pivot & Iterative version of the Bron-Kerbosch algorithm
pub fn basic_iter_pivot(
    graph: Arc<Graph>,
    r: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    let nn = get_pivot_list(graph.clone(), &p, &x);
    base_iter(graph, r, vec![(p, x, nn)], get_pivot_list, opts)
}

/// Iterative version of the Bron-Kerbosch algorithm using degeneracy ordering
pub fn basic_iter_ordering(
    graph: Arc<Graph>,
    r: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    base_iter(
        graph.clone(),
        r,
        vec![(p, x, degeneracy_ordering(graph, opts))],
        get_pivot_list, // Uses a pivot as a list of candidates for recursion
        opts,
    )
}

/// Base skeleton for the top-level parallel version of the Bron-Kerbosch algorithm
#[cfg(feature = "rayon")]
#[inline]
fn base_par(
    graph: Arc<Graph>,
    p: Set,
    x: Set,
    fun: Algorithm,
    nodes: &[Node],
    opts: Options,
) -> Report {
    let guard = Mutex::new((p, x));
    nodes
        .into_par_iter() // Parallel iterator to perform a high-order mapping in parallel
        .map(|node| -> Report {
            let (np, nx);
            let neighbours: &Set = &graph[&node];
            // Locked scope to modify P and X without data-races
            {
                let mut data = guard.lock().unwrap();
                let (p, x) = &mut *data;
                np = &*p & neighbours;
                nx = &*x & neighbours;
                p.remove(node);
                x.insert(*node);
            }
            fun(graph.clone(), &mut vec![*node], np, nx, opts)
        })
        .sum() // Collects all sub-tasks
}

/// Top-Level parallel naive version of the Bron-Kerbosch algorithm
#[cfg(feature = "rayon")]
pub fn basic_top_par(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    let nodes: Vec<_> = p.iter().cloned().collect();
    // Use the naive sequential implementation for recursion
    base_par(graph, p, x, basic, &nodes, opts)
}

/// Top-Level parallel pivot version of the Bron-Kerbosch algorithm
#[cfg(feature = "rayon")]
pub fn basic_top_par_pivot(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    let nodes: Vec<_> = p.iter().cloned().collect();
    // Use the pivot sequential implementation for recursion
    base_par(graph, p, x, basic_pivot, &nodes, opts)
}

/// Top-Level parallel version of the Bron-Kerbosch algorithm using degeneracy ordering
#[cfg(feature = "rayon")]
pub fn basic_top_par_ordering(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    base_par(
        graph.clone(),
        p,
        x,
        basic_pivot, // Use the pivot sequential implementation for recursion
        &degeneracy_ordering(graph, opts),
        opts,
    )
}

/// Base skeleton for the deeply parallel version of the Bron-Kerbosch algorithm
#[cfg(feature = "pool")]
fn base_pool(
    graph: Arc<Graph>,
    stack: (Vec<Node>, Set, Set),
    pool: ThreadPool,
    sender: Sender<Report>,
    func: (Generator, Algorithm),
    opts: Options,
) {
    let (r, mut p, mut x) = stack;
    let mut report = Report::default();
    for node in func.0(graph.clone(), &p, &x) {
        // Loop over candidates in P
        let neighbours: &Set = &graph[&node];
        let (mut nr, np, nx) = (r.clone(), &p & neighbours, &x & neighbours);
        nr.push(node);
        p.remove(&node);
        x.insert(node);
        if np.is_empty() && nx.is_empty() {
            // If P and X are empty, report the clique
            if opts.clique_size.is_none() || opts.clique_size.unwrap() >= nr.len() {
                report += Report {
                    count: 1,
                    max_size: nr.len(),
                    cliques: Some(vec![nr]),
                };
            } else {
                report.count += 1;
            }
        } else if nr.len() > 4 || np.len() < nx.len() {
            // If too deep, execute sequentially
            report += func.1(graph.clone(), &mut nr, np, nx, opts);
        } else {
            // Else, generate a new task
            let (graph, pc, sender) = (graph.clone(), pool.clone(), sender.clone());
            pool.execute(move || {
                base_pool(graph, (nr, np, nx), pc, sender, func, opts);
            });
        }
    }
    sender.send(report).unwrap(); // Send the report to the channel
}

/// Deeply parallel version of the naive Bron-Kerbosch algorithm
#[cfg(feature = "pool")]
pub fn basic_pool(graph: Arc<Graph>, _: &mut Vec<Node>, p: Set, x: Set, opts: Options) -> Report {
    let pool = ThreadPool::new((opts.num_threads - 1).max(1)); // Keep one thread for task collection
    let (tx, rx) = bounded::<Report>(pool.max_count() * 1024); // Limit the channel's size to prevent memory exhaustion
    let cp = pool.clone();
    pool.execute(move || {
        base_pool(graph, (vec![], p, x), cp, tx, (get_list, basic), opts); // Execute first task on another thread
    });
    rx.iter().sum() // Collect all sub-tasks and returns the report
}

/// Deeply parallel version of the pivot Bron-Kerbosch algorithm
#[cfg(feature = "pool")]
pub fn basic_pool_pivot(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    let pool = ThreadPool::new((opts.num_threads - 1).max(1)); // Keep one thread for task collection
    let (tx, rx) = bounded::<Report>(pool.max_count() * 1024); // Limit the channel's size to prevent memory exhaustion
    let cp = pool.clone();
    pool.execute(move || {
        base_pool(
            graph,
            (vec![], p, x),
            cp,
            tx,
            (get_pivot_list, basic_pivot),
            opts,
        ); // Execute first task on another thread
    });
    rx.iter().sum() // Collect all sub-tasks and returns the report
}

/// Base skeleton for the distributed memory version of the Bron-Kerbosch algorithm (using MPI)
#[cfg(feature = "mpi")]
#[inline]
fn mpi_base(
    graph: Arc<Graph>,
    mut p: Set,
    mut x: Set,
    opts: Options,
    gen: Generator,
    fun: Algorithm,
) -> Report {
    let universe = initialize().expect("Could not initialize MPI");
    let world = universe.world();
    let size = world.size();
    assert!(
        size > 1,
        "Cannot run on only one node due to master/slave configuration"
    );
    let rank = world.rank();
    let root_process = world.process_at_rank(0);
    let mut cliques = Report::default();

    if rank == 0 {
        let all: Vec<Node> = gen(graph.clone(), &p, &x);
        let step = std::cmp::max(1, all.len() / (size - 1) as usize); // Compute the size of the blocks
        for rank in 0..size - 1 {
            // Send the blocks to each slave
            let process = world.process_at_rank(rank + 1);
            let start = std::cmp::min(all.len(), step * rank as usize);
            let end = std::cmp::min(all.len(), step * (rank + 1) as usize);
            let chunk = if rank != size - 1 {
                // Compute which block to send
                &all[start..end]
            } else {
                &all[start..]
            };
            process.send(&chunk.len());
            for node in chunk {
                // Pre-process the data then sends it to the node
                let neighbours: &Set = &graph[node];
                let np = p.intersection(neighbours).cloned().collect::<Vec<_>>(); // "Marshal" the data
                let nx = x.intersection(neighbours).cloned().collect::<Vec<_>>(); // "Marshal" the data
                p.remove(node);
                x.insert(*node);
                process.send(node);
                let p_len = np.len();
                process.send(&p_len);
                process.send(&np[..]);
                let x_len = nx.len();
                process.send(&x_len);
                process.send(&nx[..]);
            }
        }
        std::mem::drop((p, x)); // Release memory (The sets are no longer useful)
        let (input, mut output) = (
            Vec::from(cliques), // "Marshal" the report
            vec![0 as usize; (2 * size) as usize],
        ); // Prepare an array to receive results
        root_process.gather_into_root(&input[..], &mut output[..]); // Gather results
        cliques = output
            .chunks_exact(2)
            .map(|x| Report::try_from(x).unwrap())
            .sum(); // "Unmarshal" the partial reports
    } else {
        let mut len: usize = 0;
        root_process.receive_into(&mut len);
        let mut all: Vec<(Node, Vec<Node>, Vec<Node>)> = vec![];
        for _ in 0..len {
            // Receive and "Unmarshal" the data
            let mut r: Node = 0;
            let (mut p, mut x);
            root_process.receive_into(&mut r);
            let mut p_len: usize = 0;
            root_process.receive_into(&mut p_len);
            p = vec![0 as Node; p_len];
            root_process.receive_into(&mut p[..]);
            let mut x_len: usize = 0;
            root_process.receive_into(&mut x_len);
            x = vec![0 as Node; x_len];
            root_process.receive_into(&mut x[..]);
            all.push((r, p, x));
        }
        for (r, p, x) in all {
            let p: Set = p.into_iter().collect(); // Unmarshal the data
            let x: Set = x.into_iter().collect(); // Unmarshal the data
            cliques += fun(graph.clone(), &mut vec![r], p, x, opts); // Proceed to run the chosen algorithm
        }
        root_process.gather_into(&Vec::from(cliques.clone())[..]); // Send the results to the root node
    }
    let current_len = cliques.max_size;
    let mut max_len: usize = current_len;
    root_process.broadcast_into(&mut max_len);
    if rank == 0 {
        // Gather cliques
        let mut sizes = vec![0 as usize; size as usize];
        for process in (1..size).map(|x| world.process_at_rank(x)) {
            // Ask for sizes for each node to allocate size later
            process.receive_into(&mut sizes[process.rank() as usize]);
        }
        let total_size = sizes.iter().sum();
        let mut max_cliques = vec![0 as Node; total_size]; // Allocate array for all results
        for process in (1..size).map(|x| world.process_at_rank(x)) {
            // gather all cliques
            if sizes[process.rank() as usize] == 0 {
                continue;
            }
            process
                .receive_into(&mut max_cliques[sizes.iter().take(process.rank() as usize).sum()..]);
        }
        cliques.cliques = Some(
            max_cliques
                .chunks_exact(max_len)
                .map(|x| x.to_vec())
                .collect::<Vec<Vec<Node>>>(),
        ) // Unmarshal the results to compute the final report
    } else {
        if max_len == current_len {
            // Send the cliques to the root node
            match cliques.cliques.as_ref() {
                Some(cl) => {
                    let flat: Vec<Node> = cl.iter().flatten().cloned().collect();
                    root_process.send(&flat.len());
                    root_process.send(&flat[..]);
                }
                None => root_process.send(&0),
            };
        } else {
            root_process.send(&0);
        };
    }
    if rank == 0 && opts.verbose {
        println!("{}", cliques);
    }
    cliques
}

/// Distributed memory parallel version of the naive Bron-Kerbosch algorithm
#[cfg(feature = "mpi")]
pub fn basic_mpi(graph: Arc<Graph>, _: &mut Vec<Node>, p: Set, x: Set, opts: Options) -> Report {
    mpi_base(graph, p, x, opts, get_list, basic)
}

/// Distributed memory parallel version of the pivot Bron-Kerbosch algorithm
#[cfg(feature = "mpi")]
pub fn basic_mpi_pivot(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    mpi_base(graph, p, x, opts, get_pivot_list, basic_pivot)
}

/// Utility function to measure performance of an implementation, returns the run-time
pub fn test_alg(graph: Arc<Graph>, fun: Algorithm, options: Options) -> Duration {
    let start = Instant::now();
    let p: Set = graph.nodes().cloned().collect();
    let cliques = fun(graph, &mut vec![], p, Set::default(), options);
    let elapsed = start.elapsed();
    if options.verbose {
        println!("{}", cliques);
    }
    elapsed
}
