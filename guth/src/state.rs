use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct StreamStep {
    pub index: usize,
}

impl StreamStep {
    pub fn new() -> Self {
        Self { index: 0 }
    }

    pub fn increment(&mut self, by: usize) {
        self.index = self.index.saturating_add(by);
    }
}

impl Default for StreamStep {
    fn default() -> Self {
        Self::new()
    }
}

pub trait StreamingModule {
    type State;

    fn init_state(&self, batch_size: usize, sequence_length: usize) -> Self::State;
    fn increment_step(&self, state: &mut Self::State, increment: usize);
}

#[derive(Debug, Default)]
pub struct StateTree {
    steps: HashMap<String, StreamStep>,
}

impl StateTree {
    pub fn set_step(&mut self, key: impl Into<String>, step: StreamStep) {
        self.steps.insert(key.into(), step);
    }

    pub fn step_mut(&mut self, key: &str) -> Option<&mut StreamStep> {
        self.steps.get_mut(key)
    }

    pub fn step(&self, key: &str) -> Option<&StreamStep> {
        self.steps.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::{StateTree, StreamStep};

    #[test]
    fn stream_step_increments() {
        let mut step = StreamStep::new();
        step.increment(2);
        assert_eq!(step.index, 2);
        step.increment(0);
        assert_eq!(step.index, 2);
    }

    #[test]
    fn state_tree_gets_and_sets() {
        let mut tree = StateTree::default();
        assert!(tree.step("missing").is_none());

        tree.set_step("flow", StreamStep { index: 5 });
        assert_eq!(tree.step("flow").map(|s| s.index), Some(5));

        if let Some(step) = tree.step_mut("flow") {
            step.increment(3);
        }
        assert_eq!(tree.step("flow").map(|s| s.index), Some(8));
    }
}
