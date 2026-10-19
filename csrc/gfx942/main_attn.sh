#!/usr/bin/env bash
set -euo pipefail

if [[ -f /opt/venv/bin/activate ]]; then
    source /opt/venv/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
BIN="$SCRIPT_DIR/build/attention_tester"
PROF_DIR="$SCRIPT_DIR/rocprof_results"

mkdir -p "/run/user/$(id -u)"

RUN_ATT=${RUN_ATT:-1}
RUN_CORRECTNESS=${RUN_CORRECTNESS:-0}
RUN_BENCHMARK=${RUN_BENCHMARK:-0}
NO_TAIL=${MOONMATH_NO_TAIL:-1}
if [[ "$NO_TAIL" == "1" ]]; then
    export MOONMATH_NO_TAIL=1
else
    unset MOONMATH_NO_TAIL
fi

PERF_B=${PERF_B:-2}
PERF_H=${PERF_H:-24}
PERF_S=${PERF_S:-16384}
PERF_D=${PERF_D:-128}
PERF_WARMUP=${PERF_WARMUP:-10}
PERF_ITERS=${PERF_ITERS:-10}

CHECK_B=${CHECK_B:-1}
CHECK_H=${CHECK_H:-2}
CHECK_S=${CHECK_S:-256}
CHECK_D=${CHECK_D:-128}
CHECK_WARMUP=${CHECK_WARMUP:-2}
CHECK_ITERS=${CHECK_ITERS:-5}

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: tester not found or not executable: $BIN" >&2
    echo "Build it first with: $SCRIPT_DIR/setup.sh" >&2
    exit 1
fi

PERF_MID_ITER=$(( PERF_ITERS / 2 ))
if (( PERF_MID_ITER < 1 )); then
    PERF_MID_ITER=1
fi
PERF_CHECK_WORK=$(( PERF_B * PERF_H * PERF_S * PERF_S * PERF_D ))
PERF_CHECK_THRESHOLD=$(( 1 * 2 * 512 * 512 * 128 ))
if (( PERF_CHECK_WORK <= PERF_CHECK_THRESHOLD )); then
    PERF_CHECK_MODE="ENABLED (auto: small config)"
else
    PERF_CHECK_MODE="DISABLED (performance only)"
fi

echo "========================================================================"
echo "moonmath_attention run configuration"
echo "  Correctness pass : $([[ "$RUN_CORRECTNESS" == "1" ]] && echo enabled || echo disabled)"
echo "  ATT pass         : $([[ "$RUN_ATT" == "1" ]] && echo enabled || echo disabled)"
echo "  Benchmark pass   : $([[ "$RUN_BENCHMARK" == "1" ]] && echo enabled || echo disabled)"
echo "  Tail workspace   : $([[ "$NO_TAIL" == "1" ]] && echo disabled || echo enabled)"
echo "========================================================================"

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    echo ""
    echo "--- Pass 1: correctness ENABLED (B=$CHECK_B H=$CHECK_H S=$CHECK_S D=$CHECK_D) ---"
    "$BIN" "$CHECK_B" "$CHECK_H" "$CHECK_S" "$CHECK_D" "$CHECK_WARMUP" "$CHECK_ITERS" --force-check
fi

if [[ "$RUN_ATT" == "1" ]]; then
    rm -rf "$PROF_DIR"
    mkdir -p "$PROF_DIR"
    echo ""
    echo "--- Pass 2: ATT profiling; correctness $PERF_CHECK_MODE (B=$PERF_B H=$PERF_H S=$PERF_S D=$PERF_D) ---"
    echo "--- Capturing middle timed kernel (iter $PERF_MID_ITER/$PERF_ITERS) ---"
    rocprofv3 --att \
        --att-activity 8 \
        -d "${PROF_DIR}/att_trace" \
        --kernel-include-regex "attention_forward" \
        --kernel-iteration-range "$PERF_MID_ITER" \
        -- "$BIN" "$PERF_B" "$PERF_H" "$PERF_S" "$PERF_D" "$PERF_WARMUP" "$PERF_ITERS" \
        || echo "WARNING: ATT pass failed"
fi

if [[ "$RUN_BENCHMARK" == "1" ]]; then
    echo ""
    echo "--- Pass 3: Benchmark only, no profiling (B=$PERF_B H=$PERF_H S=$PERF_S D=$PERF_D) ---"
    "$BIN" "$PERF_B" "$PERF_H" "$PERF_S" "$PERF_D" "$PERF_WARMUP" "$PERF_ITERS"
fi

echo ""
echo "========================================================================"
echo "=== Profiling complete"
echo "========================================================================"
