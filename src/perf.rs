//! Lightweight performance aggregation utilities.
//!
//! This module provides coarse-grained timing and counter tracking with
//! minimal overhead. It is always enabled and intended for end-of-run
//! summaries rather than fine-grained profiling.

use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricKind {
    Duration,
    Counter,
}

#[derive(Debug, Clone, Copy)]
struct MetricInfo {
    name: &'static str,
    kind: MetricKind,
}

/// Named metrics tracked by the perf collector.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Metric {
    RuntimeFromConfigPath,
    RuntimeFromConfig,
    RuntimePrepareTokens,
    RuntimeConditionOnAudioPath,
    RuntimeConditionOnAudioSamples,
    RuntimeConditioningFromAudioPath,
    RuntimeConditioningFromAudioSamples,
    TtsGenerateBatch,
    TtsGenerateStream,
    FlowLmTotal,
    MimiDecodeTotal,
    TtsTokens,
    MimiFrames,
    MimiSamples,
    MhaAppendKv,
    MhaApplyRope,
    MhaBuildMask,
    MhaAttention,
    FlowLmInputLinear,
    FlowLmBackbone,
    FlowLmEos,
    FlowNetLsdDecode,
    TransformerAttn,
    TransformerFfn,
}

impl Metric {
    const COUNT: usize = 24;

    fn index(self) -> usize {
        self as usize
    }
}

const METRICS: [MetricInfo; Metric::COUNT] = [
    MetricInfo {
        name: "runtime.from_config_path",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "runtime.from_config",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "runtime.prepare_tokens",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "runtime.condition_on_audio_path",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "runtime.condition_on_audio_samples",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "runtime.conditioning_from_audio_path",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "runtime.conditioning_from_audio_samples",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "tts.generate.batch",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "tts.generate.stream",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "flow_lm.total",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "mimi.decode.total",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "tts.tokens",
        kind: MetricKind::Counter,
    },
    MetricInfo {
        name: "mimi.frames",
        kind: MetricKind::Counter,
    },
    MetricInfo {
        name: "mimi.samples",
        kind: MetricKind::Counter,
    },
    MetricInfo {
        name: "mha.append_kv",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "mha.apply_rope",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "mha.build_mask",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "mha.attention",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "flow_lm.input_linear",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "flow_lm.backbone",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "flow_lm.eos",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "flow_net.lsd_decode",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "transformer.attn",
        kind: MetricKind::Duration,
    },
    MetricInfo {
        name: "transformer.ffn",
        kind: MetricKind::Duration,
    },
];

struct PerfCollector {
    start: Instant,
    totals_us: [AtomicU64; Metric::COUNT],
    counts: [AtomicU64; Metric::COUNT],
}

impl PerfCollector {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            totals_us: std::array::from_fn(|_| AtomicU64::new(0)),
            counts: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    fn add_duration(&self, metric: Metric, duration: Duration) {
        let micros = duration.as_micros();
        let micros = if micros > u64::MAX as u128 {
            u64::MAX
        } else {
            micros as u64
        };
        let index = metric.index();
        self.totals_us[index].fetch_add(micros, Ordering::Relaxed);
        self.counts[index].fetch_add(1, Ordering::Relaxed);
    }

    fn add_count(&self, metric: Metric, delta: u64) {
        let index = metric.index();
        self.counts[index].fetch_add(delta, Ordering::Relaxed);
    }

    fn snapshot(&self) -> PerfSnapshot {
        let mut totals_us = [0u64; Metric::COUNT];
        let mut counts = [0u64; Metric::COUNT];
        for idx in 0..Metric::COUNT {
            totals_us[idx] = self.totals_us[idx].load(Ordering::Relaxed);
            counts[idx] = self.counts[idx].load(Ordering::Relaxed);
        }
        PerfSnapshot {
            uptime: self.start.elapsed(),
            totals_us,
            counts,
        }
    }
}

static COLLECTOR: OnceLock<PerfCollector> = OnceLock::new();

fn collector() -> &'static PerfCollector {
    COLLECTOR.get_or_init(PerfCollector::new)
}

/// A RAII timer that records its duration when dropped.
pub struct PerfSpan {
    metric: Metric,
    start: Instant,
}

impl Drop for PerfSpan {
    fn drop(&mut self) {
        collector().add_duration(self.metric, self.start.elapsed());
    }
}

/// Begin a named timing span.
pub fn span(metric: Metric) -> PerfSpan {
    PerfSpan {
        metric,
        start: Instant::now(),
    }
}

/// Record a duration for a named metric.
pub fn add_duration(metric: Metric, duration: Duration) {
    collector().add_duration(metric, duration);
}

/// Record a counter delta for a named metric.
pub fn add_count(metric: Metric, delta: u64) {
    collector().add_count(metric, delta);
}

/// Snapshot of collected performance data.
#[derive(Debug)]
pub struct PerfSnapshot {
    uptime: Duration,
    totals_us: [u64; Metric::COUNT],
    counts: [u64; Metric::COUNT],
}

impl PerfSnapshot {
    /// Format a human-readable report.
    pub fn format(&self) -> String {
        let mut duration_rows: Vec<(usize, u64, u64)> = Vec::new();
        let mut counter_rows: Vec<(usize, u64)> = Vec::new();

        for (idx, metric) in METRICS.iter().enumerate() {
            let total_us = self.totals_us[idx];
            let count = self.counts[idx];
            match metric.kind {
                MetricKind::Duration => {
                    if count > 0 || total_us > 0 {
                        duration_rows.push((idx, total_us, count));
                    }
                }
                MetricKind::Counter => {
                    if count > 0 {
                        counter_rows.push((idx, count));
                    }
                }
            }
        }

        duration_rows.sort_by(|a, b| b.1.cmp(&a.1));
        counter_rows.sort_by(|a, b| b.1.cmp(&a.1));

        let mut output = String::new();
        let _ = writeln!(
            &mut output,
            "Performance summary (uptime: {:.3}s)",
            self.uptime.as_secs_f64()
        );

        if duration_rows.is_empty() && counter_rows.is_empty() {
            let _ = writeln!(&mut output, "No performance data recorded.");
            return output;
        }

        if !duration_rows.is_empty() {
            let _ = writeln!(&mut output, "Durations:");
            let _ = writeln!(
                &mut output,
                "  {:<32} {:>10} {:>8} {:>10}",
                "name", "total", "count", "avg"
            );
            for (idx, total_us, count) in duration_rows {
                let avg_ms = if count == 0 {
                    0.0
                } else {
                    (total_us as f64) / (count as f64) / 1000.0
                };
                let _ = writeln!(
                    &mut output,
                    "  {:<32} {:>10.3}s {:>8} {:>10.3}ms",
                    METRICS[idx].name,
                    (total_us as f64) / 1_000_000.0,
                    count,
                    avg_ms
                );
            }
        }

        if !counter_rows.is_empty() {
            let _ = writeln!(&mut output, "Counters:");
            for (idx, value) in counter_rows {
                let _ = writeln!(&mut output, "  {:<32} {}", METRICS[idx].name, value);
            }
        }

        output
    }
}

/// Format a report of all collected metrics.
pub fn report() -> String {
    collector().snapshot().format()
}
