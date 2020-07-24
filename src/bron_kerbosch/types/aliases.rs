use crate::bron_kerbosch::types::*;
use std::sync::Arc;

/// Generic function type, all algorithms must follow this prototype
pub type Algorithm<N = Node> = fn(Arc<Graph>, &mut Vec<Node>, Set<N>, Set<N>, Options) -> Report;

/// A Generator is a function that returns a list of candidates,
/// useful to have a generic implementation of an algorithm
pub type Generator<N = Node> = fn(Arc<Graph>, &Set<N>, &Set<N>) -> Vec<Node>;
