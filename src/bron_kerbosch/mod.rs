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

#[inline]
fn get_list(_: Arc<Graph>, p: &Set, _: &Set) -> Vec<Node> {
    p.iter().cloned().collect()
}

#[inline]
fn get_pivot_list(graph: Arc<Graph>, p: &Set, x: &Set) -> Vec<Node> {
    let pivot = p
        .union(x)
        .max_by_key(|x| graph[x].intersection(p).count())
        .unwrap();
    p.difference(&graph[pivot]).cloned().collect()
}

pub fn basic(
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
        for node in get_list(graph.clone(), &p, &x) {
            let neighbours: &Set = &graph[&node];
            r.push(node);
            cliques += basic(graph.clone(), r, &p & neighbours, &x & neighbours, opts);
            r.pop();
            p.remove(&node);
            x.insert(node);
        }
    }
    cliques
}

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
        cliques += basic_pivot(graph.clone(), r, &p & neighbours, &x & neighbours, opts);
        r.pop();
        p.remove(&n);
        x.insert(n);
    }
    cliques
}

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
        match nodes.pop() {
            Some(node) => {
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
                    stack.push((p, x, nodes));
                } else {
                    let nn = f(graph.clone(), &np, &nx);
                    let new = (np, nx, nn);
                    r.push(node);
                    stack.push((p, x, nodes));
                    stack.push(new);
                }
            }
            None => {
                r.pop();
            }
        }
    }
    cliques
}

pub fn basic_iter(graph: Arc<Graph>, r: &mut Vec<Node>, p: Set, x: Set, opts: Options) -> Report {
    let nn = get_list(graph.clone(), &p, &x);
    base_iter(graph, r, vec![(p, x, nn)], get_list, opts)
}

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
        get_pivot_list,
        opts,
    )
}

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
        .into_par_iter()
        .map(|node| -> Report {
            let (np, nx);
            let neighbours: &Set = &graph[&node];
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
        .sum()
}

#[cfg(feature = "rayon")]
pub fn basic_top_par(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    let nodes: Vec<_> = p.iter().cloned().collect();
    base_par(graph, p, x, basic, &nodes, opts)
}

#[cfg(feature = "rayon")]
pub fn basic_top_par_pivot(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    let nodes: Vec<_> = p.iter().cloned().collect();
    base_par(graph, p, x, basic_pivot, &nodes, opts)
}

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
        basic_pivot,
        &degeneracy_ordering(graph, opts),
        opts,
    )
}

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
        let neighbours: &Set = &graph[&node];
        let (mut nr, np, nx) = (r.clone(), &p & neighbours, &x & neighbours);
        nr.push(node);
        p.remove(&node);
        x.insert(node);
        if np.is_empty() && nx.is_empty() {
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
            report += func.1(graph.clone(), &mut nr, np, nx, opts);
        } else {
            let (graph, pc, sender) = (graph.clone(), pool.clone(), sender.clone());
            pool.execute(move || {
                base_pool(graph, (nr, np, nx), pc, sender, func, opts);
            });
        }
    }
    sender.send(report).unwrap();
}

#[cfg(feature = "pool")]
pub fn basic_pool(graph: Arc<Graph>, _: &mut Vec<Node>, p: Set, x: Set, opts: Options) -> Report {
    let pool = ThreadPool::new((opts.num_threads - 1).max(1));
    let (tx, rx) = bounded::<Report>(pool.max_count() * 1024);
    let cp = pool.clone();
    pool.execute(move || {
        base_pool(graph, (vec![], p, x), cp, tx, (get_list, basic), opts);
    });
    rx.iter().sum()
}

#[cfg(feature = "pool")]
pub fn basic_pool_pivot(
    graph: Arc<Graph>,
    _: &mut Vec<Node>,
    p: Set,
    x: Set,
    opts: Options,
) -> Report {
    let pool = ThreadPool::new((opts.num_threads - 1).max(1));
    let (tx, rx) = bounded::<Report>(pool.max_count() * 1024);
    let cp = pool.clone();
    pool.execute(move || {
        base_pool(
            graph,
            (vec![], p, x),
            cp,
            tx,
            (get_pivot_list, basic_pivot),
            opts,
        );
    });
    rx.iter().sum()
}

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
        let step = std::cmp::max(1, all.len() / (size - 1) as usize);
        for rank in 0..size - 1 {
            let process = world.process_at_rank(rank + 1);
            let start = std::cmp::min(all.len(), step * rank as usize);
            let end = std::cmp::min(all.len(), step * (rank + 1) as usize);
            let chunk = if rank != size - 1 {
                &all[start..end]
            } else {
                &all[start..]
            };
            process.send(&chunk.len());
            for node in chunk {
                let neighbours: &Set = &graph[node];
                let np = p.intersection(neighbours).cloned().collect::<Vec<_>>();
                let nx = x.intersection(neighbours).cloned().collect::<Vec<_>>();
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
        std::mem::drop((p, x));
        let (input, mut output) = (Vec::from(cliques), vec![0 as usize; (2 * size) as usize]);
        root_process.gather_into_root(&input[..], &mut output[..]);
        cliques = output
            .chunks_exact(2)
            .map(|x| Report::try_from(x).unwrap())
            .sum();
    } else {
        let mut len: usize = 0;
        root_process.receive_into(&mut len);
        let mut all: Vec<(Node, Vec<Node>, Vec<Node>)> = vec![];
        for _ in 0..len {
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
            let p: Set = p.into_iter().collect();
            let x: Set = x.into_iter().collect();
            cliques += fun(graph.clone(), &mut vec![r], p, x, opts);
        }
        root_process.gather_into(&Vec::from(cliques.clone())[..]);
    }
    let current_len = cliques.max_size;
    let mut max_len: usize = current_len;
    root_process.broadcast_into(&mut max_len);
    if rank == 0 {
        let mut sizes = vec![0 as usize; size as usize];
        for process in (1..size).map(|x| world.process_at_rank(x)) {
            process.receive_into(&mut sizes[process.rank() as usize]);
        }
        let total_size = sizes.iter().sum();
        let mut max_cliques = vec![0 as Node; total_size];
        for process in (1..size).map(|x| world.process_at_rank(x)) {
            if *&sizes[process.rank() as usize] == 0 {
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
        )
    } else {
        if max_len == current_len {
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

#[cfg(feature = "mpi")]
pub fn basic_mpi(graph: Arc<Graph>, _: &mut Vec<Node>, p: Set, x: Set, opts: Options) -> Report {
    mpi_base(graph, p, x, opts, get_list, basic)
}

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
