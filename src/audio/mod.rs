//! Audio I/O and resampling utilities for the TTS CLI and tests.
//!
//! These helpers keep audio handling separate from the model itself, focusing on
//! reading/writing waveforms and converting sample rates or channel counts.

pub mod io;
pub mod resample;
