use std::collections::hash_map as base;
use std::fmt;
use std::hash::Hash;
use std::ops::Index;
use std::time::{Duration, Instant};

#[cfg(feature = "rustc-hash")]
pub use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
#[cfg(not(feature = "rustc-hash"))]
pub use std::collections::{HashMap, HashSet};

#[cfg(not(feature = "csv"))]
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

pub type Node = u32;
pub type Set<N = Node> = HashSet<N>;

#[cfg(feature = "csv")]
type Record<N = Node> = (N, N);

pub type Map<K, V = Set<K>> = HashMap<K, V>;

#[derive(Clone)]
pub struct Graph<N = Node>
where
    N: Eq + Hash,
{
    inner: Map<N, Set<N>>,
    debug_duration: Duration,
}

impl<N> Graph<N>
where
    N: Eq + Hash + Copy,
{
    #[inline]
    pub fn neighbours(&self, node: &N) -> Option<&Set<N>> {
        self.inner.get(node)
    }

    #[inline]
    pub fn compact(&mut self) {
        self.inner.shrink_to_fit();
    }

    #[inline]
    pub fn add_node(&mut self, node: N) -> &Set<N> {
        self.inner.entry(node).or_insert_with(Default::default)
    }

    #[inline]
    pub fn items(&self) -> base::Iter<N, Set<N>> {
        self.inner.iter()
    }

    #[inline]
    pub fn items_mut(&mut self) -> base::IterMut<N, Set<N>> {
        self.inner.iter_mut()
    }

    #[inline]
    pub fn nodes(&self) -> base::Keys<N, Set<N>> {
        self.inner.keys()
    }

    #[inline]
    pub fn edges(&self) -> base::Values<N, Set<N>> {
        self.inner.values()
    }

    #[inline]
    pub fn edges_mut(&mut self) -> base::ValuesMut<N, Set<N>> {
        self.inner.values_mut()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn number_nodes(&self) -> usize {
        self.inner.len()
    }

    pub fn number_edges(&self) -> usize {
        self.inner.values().map(Set::len).sum::<usize>() / 2
    }
}

impl<N> Default for Graph<N>
where
    N: Eq + Hash,
{
    fn default() -> Self {
        Graph {
            inner: Default::default(),
            debug_duration: Default::default(),
        }
    }
}

impl<N> Graph<N>
where
    N: Eq + Hash + Copy,
{
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
            debug_duration: st.elapsed(),
        }
    }

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

impl<N> IntoIterator for Graph<N>
where
    N: Eq + Hash,
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
    N: Eq + Hash,
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
    N: Eq + Hash,
{
    type Item = (&'a N, &'a mut Set<N>);
    type IntoIter = base::IterMut<'a, N, Set<N>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

#[cfg(feature = "csv")]
impl Graph<Node> {
    fn read<T>(mut reader: csv::Reader<T>) -> Self
    where
        T: std::io::Read,
    {
        let st = Instant::now();
        let mut inner: Map<Node, Set<Node>> = Default::default();
        reader
            .deserialize::<Record>()
            .map(Result::unwrap)
            .for_each(|r| {
                if r.0 == r.1 {
                    return;
                }
                inner.entry(r.0).or_default().insert(r.1);
                inner.entry(r.1).or_default().insert(r.0);
            });
        for list in inner.values_mut() {
            list.shrink_to_fit();
        }
        inner.shrink_to_fit();
        Graph {
            inner,
            debug_duration: st.elapsed(),
        }
    }

    pub fn from_stdin() -> Self {
        let reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .comment(Some(b'#'))
            .has_headers(false)
            .from_reader(std::io::stdin());
        Self::read(reader)
    }

    pub fn from_file(file_path: &str) -> Self {
        let reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .comment(Some(b'#'))
            .has_headers(false)
            .from_path(file_path)
            .unwrap();
        Self::read(reader)
    }
}

#[cfg(not(feature = "csv"))]
impl Graph<Node> {
    fn read<T>(reader: BufReader<T>) -> Self
    where
        T: std::io::Read,
    {
        let st = Instant::now();
        let mut inner: HashMap<Node, Set<Node>> = Default::default();
        let mut buf: Vec<Node> = Vec::with_capacity(2);

        reader
            .lines()
            .map(Result::unwrap)
            .filter(|x| !x.starts_with('#'))
            .for_each(|line| {
                buf.extend(
                    line.split_whitespace()
                        .map(|s| s.parse::<Node>().unwrap())
                        .take(2),
                );
                inner.entry(buf[0]).or_default().insert(buf[1]);
                inner.entry(buf[1]).or_default().insert(buf[0]);
                buf.clear();
            });
        Graph {
            inner,
            debug_duration: st.elapsed(),
        }
    }

    pub fn from_stdin() -> Self {
        let reader = BufReader::new(std::io::stdin());
        Self::read(reader)
    }

    pub fn from_file(file_path: &str) -> Self {
        let reader = BufReader::new(File::open(file_path).unwrap());
        Self::read(reader)
    }
}

impl<N> Index<&N> for Graph<N>
where
    N: Eq + Hash,
{
    type Output = Set<N>;

    #[inline]
    fn index(&self, key: &N) -> &Self::Output {
        self.inner.get(key).expect("no entry found for key")
    }
}

impl<'a, N> Index<&N> for &'a Graph<N>
where
    N: Eq + Hash,
{
    type Output = Set<N>;

    #[inline]
    fn index(&self, key: &N) -> &Self::Output {
        self.inner.get(key).expect("no entry found for key")
    }
}

impl<N> fmt::Debug for Graph<N>
where
    N: Eq + Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("GenerationTime", &self.debug_duration)
            .field("Nodes", &self.inner.len())
            .field(
                "Edges",
                &(self.inner.values().map(Set::len).sum::<usize>() / 2),
            )
            .finish()
    }
}
