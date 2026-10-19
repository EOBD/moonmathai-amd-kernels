#!/usr/bin/env bash
set -euo pipefail

if [[ -f /opt/venv/bin/activate ]]; then
    source /opt/venv/bin/activate
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIN="$SCRIPT_DIR/build/mla_decode_a16w8_tester"
PROF_DIR="$SCRIPT_DIR/rocprof_results/mla_att_trace"

mkdir -p "/run/user/$(id -u)"

RUN_CORRECTNESS=${RUN_CORRECTNESS:-1}
RUN_ATT=${RUN_ATT:-1}
RUN_BENCHMARK=${RUN_BENCHMARK:-1}
NO_WORKSPACE=${MLA_NO_WORKSPACE:-${MOONMATH_NO_TAIL:-0}}
if [[ "$NO_WORKSPACE" == "1" ]]; then
    export MLA_NO_WORKSPACE=1
else
    unset MLA_NO_WORKSPACE
    unset MOONMATH_NO_TAIL
fi

PERF_B=${PERF_B:-2}
PERF_H=${PERF_H:-16}
PERF_S=${PERF_S:-16384}
PERF_WARMUP=${PERF_WARMUP:-10}
PERF_ITERS=${PERF_ITERS:-10}

CHECK_B=${CHECK_B:-1}
CHECK_H=${CHECK_H:-16}
CHECK_S=${CHECK_S:-256}
CHECK_WARMUP=${CHECK_WARMUP:-2}
CHECK_ITERS=${CHECK_ITERS:-5}

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: tester not found or not executable: $BIN" >&2
    echo "Build it first with: make mla" >&2
    exit 1
fi

PERF_MID_ITER=$(( PERF_ITERS / 2 ))
if (( PERF_MID_ITER < 1 )); then
    PERF_MID_ITER=1
fi
# rocprof counts matched launches from process start, including tester warmups.
PERF_CAPTURE_ITER=$(( PERF_WARMUP + PERF_MID_ITER ))
PERF_CHECK_WORK=$(( PERF_B * PERF_H * PERF_S * (576 + 512) ))
PERF_CHECK_THRESHOLD=$(( 1 * 16 * 512 * (576 + 512) ))
if (( PERF_CHECK_WORK <= PERF_CHECK_THRESHOLD )); then
    PERF_CHECK_MODE="ENABLED (auto: small config)"
else
    PERF_CHECK_MODE="DISABLED (performance only)"
fi

echo "========================================================================"
echo "moonmath MLA A16W8 run configuration"
echo "  Correctness pass : $([[ "$RUN_CORRECTNESS" == "1" ]] && echo enabled || echo disabled)"
echo "  ATT pass         : $([[ "$RUN_ATT" == "1" ]] && echo enabled || echo disabled)"
echo "  Benchmark pass   : $([[ "$RUN_BENCHMARK" == "1" ]] && echo enabled || echo disabled)"
echo "  Split workspace  : $([[ "$NO_WORKSPACE" == "1" ]] && echo disabled || echo enabled)"
echo "========================================================================"

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    echo ""
    echo "--- Pass 1: correctness ENABLED (B=$CHECK_B H=$CHECK_H S=$CHECK_S) ---"
    "$BIN" "$CHECK_B" "$CHECK_H" "$CHECK_S" "$CHECK_WARMUP" "$CHECK_ITERS" --force-check
fi

if [[ "$RUN_ATT" == "1" ]]; then
    rm -rf "$PROF_DIR"
    mkdir -p "$PROF_DIR"
    echo ""
    echo "--- Pass 2: ATT profiling; correctness $PERF_CHECK_MODE (B=$PERF_B H=$PERF_H S=$PERF_S) ---"
    echo "--- Capturing middle timed kernel (matched launch $PERF_CAPTURE_ITER) ---"
    rocprofv3 --att \
        --att-activity 8 \
        -d "$PROF_DIR" \
        --kernel-include-regex "mla_decode_a16w8_kernel" \
        --kernel-iteration-range "$PERF_CAPTURE_ITER" \
        -- "$BIN" "$PERF_B" "$PERF_H" "$PERF_S" "$PERF_WARMUP" "$PERF_ITERS" \
        || echo "WARNING: MLA ATT pass failed"
fi

if [[ "$RUN_BENCHMARK" == "1" ]]; then
    echo ""
    echo "--- Pass 3: benchmark only, no profiling (B=$PERF_B H=$PERF_H S=$PERF_S) ---"
    "$BIN" "$PERF_B" "$PERF_H" "$PERF_S" "$PERF_WARMUP" "$PERF_ITERS"
fi

echo ""
echo "========================================================================"
echo "=== MLA run complete"
echo "========================================================================"
