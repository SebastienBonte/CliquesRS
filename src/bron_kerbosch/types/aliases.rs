use crate::bron_kerbosch::types::*;
use std::sync::Arc;

pub type Algorithm<N = Node> = fn(Arc<Graph>, &mut Vec<Node>, Set<N>, Set<N>, Options) -> Report;
pub type Generator<N = Node> = fn(Arc<Graph>, &Set<N>, &Set<N>) -> Vec<Node>;
