# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial public release of `guth` TTS library
- `TtsModel` for text-to-speech generation with FlowLM and Mimi components
- Streaming audio generation via `generate_audio_stream`
- Voice cloning support via `condition_on_audio`
- HuggingFace Hub integration for automatic weight downloading
- Comprehensive documentation with usage examples

### Internal
- Visibility audit: internal modules marked `#[doc(hidden)]`
- Added rustfmt and clippy to CI pipeline
- Added pre-commit hooks for code quality
