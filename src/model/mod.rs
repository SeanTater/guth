//! High-level model components for FlowLM, Mimi, and the combined TTS stack.
//!
//! Each submodule implements a piece of the pipeline: FlowLM generates latent
//! audio codes, Mimi decodes them into waveforms, and `tts` orchestrates both.

pub mod flow_lm;
pub mod mimi;
pub mod tts;
