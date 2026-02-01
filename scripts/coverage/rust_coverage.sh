#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/coverage/rust_coverage.sh [--min <percent>] [--baseline-file <path>] [--quiet]

Computes Rust line coverage using cargo-llvm-cov (src-only .rs files).

Options:
  --min <percent>         Fail if coverage is below this percent.
  --baseline-file <path>  Fail if coverage is below the value stored in this file.
  --quiet                 Suppress the coverage summary line.
USAGE
}

min=""
baseline_file=""
quiet=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min)
      min="$2"
      shift 2
      ;;
    --baseline-file)
      baseline_file="$2"
      shift 2
      ;;
    --quiet)
      quiet=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found in PATH" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to parse coverage output" >&2
  exit 1
fi

if ! cargo llvm-cov --version >/dev/null 2>&1; then
  echo "cargo-llvm-cov is required. Install with: cargo install cargo-llvm-cov" >&2
  exit 1
fi

if ! rustup component list --installed 2>/dev/null | grep -q '^llvm-tools'; then
  echo "llvm-tools-preview is required. Install with: rustup component add llvm-tools-preview" >&2
  exit 1
fi

report_tmp="$(mktemp)"
run_log="$(mktemp)"
cleanup() {
  rm -f "$report_tmp" "$run_log"
}
trap cleanup EXIT

cov_args=(
  llvm-cov
  --json
  --summary-only
  --ignore-filename-regex '^(?!.*\.rs$).*'
  --output-path "$report_tmp"
)

if [[ "$quiet" == true ]]; then
  if ! cargo "${cov_args[@]}" >"$run_log" 2>&1; then
    cat "$run_log" >&2
    exit 1
  fi
else
  cargo "${cov_args[@]}"
fi

coverage=$(jq -r '[.data[0].files[] | select(.filename | test("/src/.*\\.rs$")) | .summary.lines] | {count:(map(.count)|add // 0), covered:(map(.covered)|add // 0)} | if .count == 0 then 0 else (.covered / .count * 100) end' "$report_tmp")
coverage_fmt=$(awk -v c="$coverage" 'BEGIN {printf "%.2f", c}')

if [[ "$quiet" == false ]]; then
  echo "Rust line coverage (src-only): ${coverage_fmt}%"
fi

if [[ -n "$baseline_file" ]]; then
  if [[ ! -f "$baseline_file" ]]; then
    echo "Baseline file not found: $baseline_file" >&2
    exit 1
  fi
  baseline=$(tr -d ' %\n\r' < "$baseline_file")
  if ! awk -v c="$coverage" -v b="$baseline" 'BEGIN {exit !(c+0 >= b+0)}'; then
    echo "Coverage ${coverage_fmt}% is below baseline ${baseline}%" >&2
    exit 1
  fi
fi

if [[ -n "$min" ]]; then
  if ! awk -v c="$coverage" -v m="$min" 'BEGIN {exit !(c+0 >= m+0)}'; then
    echo "Coverage ${coverage_fmt}% is below minimum ${min}%" >&2
    exit 1
  fi
fi
