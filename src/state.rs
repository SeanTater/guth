//! Streaming state helpers shared across the TTS pipeline.
//!
//! This module defines tiny, generic building blocks used by streaming layers to
//! track position and cache per-layer state as audio is generated frame by frame.

use burn::tensor::backend::Backend;
use std::collections::HashMap;

/// Trait implemented by streaming modules that keep per-request state.
pub trait StreamingModule<B: Backend> {
    /// Concrete state type for this module.
    type State;

    /// Allocate a fresh state for a given batch and sequence length.
    fn init_state(&self, batch_size: usize, sequence_length: usize) -> Self::State;
    /// Advance internal position tracking by `increment`.
    fn increment_step(&self, state: &mut Self::State, increment: usize);
}

/// Named collection of streaming step counters for nested modules.
pub type StateTree = HashMap<String, usize>;

#[cfg(test)]
mod tests {
    use super::StateTree;

    #[test]
    fn state_tree_gets_and_sets() {
        let mut tree = StateTree::default();
        assert!(tree.get("missing").is_none());

        tree.insert("flow".to_string(), 5);
        assert_eq!(tree.get("flow").copied(), Some(5));

        if let Some(step) = tree.get_mut("flow") {
            *step = step.saturating_add(3);
        }
        assert_eq!(tree.get("flow").copied(), Some(8));
    }
}
