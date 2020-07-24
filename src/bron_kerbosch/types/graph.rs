use std::collections::hash_map as base;
use std::fmt;
use std::hash::Hash;
use std::ops::Index;
use std::str::FromStr;
use std::time::{Duration, Instant};

#[cfg(feature = "fasthash")]
pub use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
#[cfg(not(feature = "fasthash"))]
pub use std::collections::{HashMap, HashSet};

use std::fmt::Formatter;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

/// Type alias for a vertex
pub type Node = u32;

/// Type alias for a set of vertices
pub type Set<N = Node> = HashSet<N>;

/// Type alias for a map of vertices to neighbours
pub type Map<K, V = Set<K>> = HashMap<K, V>;

/// Data-structure representing a graph
#[derive(Clone)]
pub struct Graph<N = Node>
where
    N: Eq + Hash + Copy, // Node type must be a primitive type (like integers)
{
    inner: Map<N, Set<N>>,
    parsing_duration: Duration,
}

impl<N> Graph<N>
where
    N: Eq + Hash + Copy,
{
    /// Returns the neighbours of a given vertex, or None if the vertex is not in the Graph
    #[inline]
    pub fn neighbours(&self, node: &N) -> Option<&Set<N>> {
        self.inner.get(node)
    }

    /// Add a vertex to the graph if it does not exist and returns its neighbours set
    #[inline]
    pub fn add_node(&mut self, node: N) -> &Set<N> {
        self.inner.entry(node).or_insert_with(Default::default)
    }

    /// Returns an iterator of pairs of (vertex, neighbours)
    #[inline]
    pub fn items(&self) -> base::Iter<N, Set<N>> {
        self.inner.iter()
    }

    /// Returns a mutable iterator of pairs of (vertex, neighbours)
    #[inline]
    pub fn items_mut(&mut self) -> base::IterMut<N, Set<N>> {
        self.inner.iter_mut()
    }

    /// Returns an iterator over all vertices in the Graph
    #[inline]
    pub fn nodes(&self) -> base::Keys<N, Set<N>> {
        self.inner.keys()
    }

    /// Returns an iterator over all edges in the Graph
    #[inline]
    pub fn edges(&self) -> base::Values<N, Set<N>> {
        self.inner.values()
    }

    /// Returns a mutable iterator over all edges in the Graph
    #[inline]
    pub fn edges_mut(&mut self) -> base::ValuesMut<N, Set<N>> {
        self.inner.values_mut()
    }

    /// Returns true if the Graph is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the number of vertices in the Graph
    #[inline]
    pub fn number_nodes(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of edges in the Graph
    #[inline]
    pub fn number_edges(&self) -> usize {
        self.inner.values().map(Set::len).sum::<usize>() / 2
    }

    /// Initialise a graph from an edge-list
    pub fn from_edges(edges: &[(N, N)]) -> Self {
        let mut inner = Map::<N, Set<N>>::default();
        let st = Instant::now();
        for r in edges {
            if r.0 == r.1 {
                continue;
            }
            inner.entry(r.0).or_default().insert(r.1);
            inner.entry(r.1).or_default().insert(r.0);
        }
        Graph {
            inner,
            parsing_duration: st.elapsed(),
        }
    }

    /// Add edges to the Graph
    pub fn add_edges(&mut self, edges: &[(N, N)]) {
        edges.iter().for_each(|&(a, b)| {
            if a == b {
                return;
            }
            self.inner.entry(a).or_default().insert(b);
            self.inner.entry(b).or_default().insert(a);
        });
    }
}

impl<N> Default for Graph<N>
where
    N: Eq + Hash + Copy,
{
    fn default() -> Self {
        Graph {
            inner: Default::default(),
            parsing_duration: Default::default(),
        }
    }
}

impl<N> IntoIterator for Graph<N>
where
    N: Eq + Hash + Copy,
{
    type Item = (N, Set<N>);
    type IntoIter = base::IntoIter<N, Set<N>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, N> IntoIterator for &'a Graph<N>
where
    N: Eq + Hash + Copy,
{
    type Item = (&'a N, &'a Set<N>);
    type IntoIter = base::Iter<'a, N, Set<N>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, N> IntoIterator for &'a mut Graph<N>
where
    N: Eq + Hash + Copy,
{
    type Item = (&'a N, &'a mut Set<N>);
    type IntoIter = base::IterMut<'a, N, Set<N>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

impl<N> Graph<N>
where
    N: Eq + Hash + Copy + FromStr,
    <N as std::str::FromStr>::Err: std::fmt::Debug,
{
    /// Skeleton for initialising a Graph from an input
    fn read<T>(reader: BufReader<T>) -> Self
    where
        T: std::io::Read,
    {
        let st = Instant::now();
        let mut inner: HashMap<N, Set<N>> = Default::default();

        reader
            .lines()
            .map(Result::unwrap)
            .filter(|x| !x.starts_with('#'))
            .for_each(|line| {
                let mut content = line.split_whitespace().map(|s| s.parse::<N>().unwrap());
                let (a, b) = (content.next().unwrap(), content.next().unwrap());
                if a != b {
                    inner.entry(a).or_default().insert(b);
                    inner.entry(b).or_default().insert(a);
                }
            });
        Graph {
            inner,
            parsing_duration: st.elapsed(),
        }
    }

    /// Initialise a Graph from the standard input
    pub fn from_stdin() -> Self {
        let reader = BufReader::new(std::io::stdin());
        Self::read(reader)
    }

    /// Initialise a Graph from a file
    pub fn from_file(file_path: &str) -> Self {
        let reader = BufReader::new(File::open(file_path).unwrap());
        Self::read(reader)
    }
}

impl<N> Index<&N> for Graph<N>
where
    N: Eq + Hash + Copy,
{
    type Output = Set<N>;

    #[inline]
    fn index(&self, key: &N) -> &Self::Output {
        self.inner.get(key).expect("no entry found for key")
    }
}

impl<'a, N> Index<&N> for &'a Graph<N>
where
    N: Eq + Hash + Copy,
{
    type Output = Set<N>;

    #[inline]
    fn index(&self, key: &N) -> &Self::Output {
        self.inner.get(key).expect("no entry found for key")
    }
}

impl<N> fmt::Debug for Graph<N>
where
    N: Eq + Hash + Copy,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("ParsingDuration", &self.parsing_duration)
            .field("Nodes", &self.number_nodes())
            .field("Edges", &self.number_edges())
            .finish()
    }
}

impl<N> fmt::Display for Graph<N>
where
    N: Eq + Hash + Copy,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "Graph of {} nodes and {} edges (parsed in {:?})",
            self.number_nodes(),
            self.number_edges(),
            self.parsing_duration
        ))
    }
}
