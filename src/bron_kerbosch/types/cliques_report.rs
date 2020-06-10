use crate::bron_kerbosch::types::graph::*;
use std::cmp::Ordering;
use std::fmt;
use std::iter::Sum;
use std::ops::{Add, AddAssign};

#[cfg(feature = "mpi")]
pub use std::convert::{From, TryFrom};

#[derive(Debug, Clone)]
pub struct Report<N = Node> {
    pub count: usize,
    pub max_size: usize,
    pub cliques: Option<Vec<Vec<N>>>,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            count: 0,
            max_size: 0,
            cliques: None,
        }
    }
}

impl Add for Report {
    type Output = Report;

    fn add(self, rhs: Self) -> Self::Output {
        Report {
            count: self.count.add(rhs.count),
            max_size: self.max_size.max(rhs.max_size),
            cliques: match self.max_size.cmp(&rhs.max_size) {
                Ordering::Equal => {
                    if let Some(c) = self.cliques {
                        if let Some(d) = rhs.cliques {
                            Some(c.into_iter().chain(d.into_iter()).collect())
                        } else {
                            None
                        }
                    } else if let Some(d) = rhs.cliques {
                        Some(d)
                    } else {
                        None
                    }
                }
                Ordering::Less => rhs.cliques,
                Ordering::Greater => self.cliques,
            },
        }
    }
}

impl AddAssign for Report {
    fn add_assign(&mut self, rhs: Self) {
        self.count.add_assign(rhs.count);
        match self.max_size.cmp(&rhs.max_size) {
            Ordering::Equal => {
                if let Some(c) = &mut self.cliques {
                    if let Some(d) = rhs.cliques {
                        c.extend(d);
                    }
                } else if let Some(d) = rhs.cliques {
                    self.cliques = Some(d);
                }
            }
            Ordering::Less => self.cliques = rhs.cliques,
            Ordering::Greater => {}
        }
        self.max_size = self.max_size.max(rhs.max_size);
    }
}

impl Sum for Report {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Default::default(), Add::add)
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "Found {} maximal cliques (Maximum size: {} - Number: {})",
            self.count,
            self.max_size,
            self.cliques.as_ref().map_or(0, Vec::len)
        ))
    }
}

#[cfg(feature = "mpi")]
impl<N> From<Report<N>> for Vec<usize> {
    fn from(report: Report<N>) -> Vec<usize> {
        vec![report.count, report.max_size]
    }
}

#[cfg(feature = "mpi")]
impl<N> TryFrom<Vec<usize>> for Report<N> {
    type Error = &'static str;

    fn try_from(vec: Vec<usize>) -> Result<Self, Self::Error> {
        match vec.len() {
            2 => Ok(Report::<N> {
                count: vec[0],
                max_size: vec[1],
                cliques: None,
            }),
            _ => Err("CliquesReport only accepts vectors of size 2"),
        }
    }
}

#[cfg(feature = "mpi")]
impl<N> TryFrom<&[usize]> for Report<N> {
    type Error = &'static str;

    fn try_from(vec: &[usize]) -> Result<Self, Self::Error> {
        match vec.len() {
            2 => Ok(Report::<N> {
                count: vec[0],
                max_size: vec[1],
                cliques: None,
            }),
            _ => Err("CliquesReport only accepts vectors of size 2"),
        }
    }
}
