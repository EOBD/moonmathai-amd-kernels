#!/usr/bin/env bash
# Profile the CDNA3 attention HIP kernel using rocprof-compute (PMC) and rocprofv3 (ATT).
# Emits ISA analysis HTML matched to HIP source, then packages the results into a zip.
#
# Usage (run from the repo root):
#   ./tools/rocprof_profile.sh
#   WORKLOAD_NAME=attn_run1 ./tools/rocprof_profile.sh          # pin the workload name
#   WARMUP_ITERS=5 ./tools/rocprof_profile.sh
#   ROCPROF_NO_UI_TRACE=1 ./tools/rocprof_profile.sh             # skip ATT (faster)
#   ROCPROF_NO_SUMMARY=1 ./tools/rocprof_profile.sh              # skip analyze + summary CSV
#   NO_ZIP=1 ./tools/rocprof_profile.sh                          # skip the final zip
#   ROCPROF_UI_ATT_LIBRARY_PATH=/path/to/decoder ./tools/rocprof_profile.sh
#   PYTHON=/path/to/python ./tools/rocprof_profile.sh
#
# Needs rocprof-compute, rocprofv3 and libamdhip64.so (ROCm).
# librocprof-trace-decoder.so is auto-detected from:
#   1) /opt/rocm/lib
#   2) ../../rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64 (sibling checkout)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Prepend the conda lib dir so the rocprofiler-sdk env reset keeps libstdc++ visible.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Locate librocprof-trace-decoder.so (needed by ATT).
if [[ -z "${ROCPROF_UI_ATT_LIBRARY_PATH:-}" ]]; then
  _decoder_found=""
  for _d in /opt/rocm/lib /opt/rocm-*/lib; do
    if [[ -f "${_d}/librocprof-trace-decoder.so" ]]; then
      _decoder_found="${_d}"; break
    fi
  done
  if [[ -z "${_decoder_found}" ]]; then
    _att_default="$(cd "${ROOT}/.." && pwd)/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64"
    [[ -f "${_att_default}/librocprof-trace-decoder.so" ]] && _decoder_found="${_att_default}"
  fi
  if [[ -n "${_decoder_found}" ]]; then
    ROCPROF_UI_ATT_LIBRARY_PATH="${_decoder_found}"
    echo "Auto-detected decoder: ${_decoder_found}/librocprof-trace-decoder.so"
  fi
fi

# Pick the Python interpreter.
PYTHON="${PYTHON:-python}"
if [[ "${PYTHON}" == "python" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
fi
if ! PYTHON="$(command -v "${PYTHON}")"; then
  echo "error: PYTHON not found" >&2; exit 1
fi

# Find rocprof-compute.
ROCPROF_COMPUTE="${ROCPROF_COMPUTE:-}"
if [[ -z "${ROCPROF_COMPUTE}" ]]; then
  if command -v rocprof-compute &>/dev/null; then
    ROCPROF_COMPUTE="$(command -v rocprof-compute)"
  elif [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/bin/rocprof-compute" ]]; then
    ROCPROF_COMPUTE="${ROCM_PATH}/bin/rocprof-compute"
  else
    echo "error: rocprof-compute not found (set ROCPROF_COMPUTE or add to PATH)" >&2; exit 1
  fi
fi

if ! command -v rocprofv3 &>/dev/null; then
  echo "error: rocprofv3 not found on PATH" >&2; exit 1
fi

# Build the kernel shared library if it is not already present.
if ! compgen -G "${ROOT}/libattention"*.so >/dev/null; then
  echo "Building libattention*.so..."
  make -C "${ROOT}"
fi

# Workload output paths.
ROC_OUT_DIR="${ROC_OUT_DIR:-${ROOT}/rocprof_out}"
WORKLOAD_NAME="${WORKLOAD_NAME:-$(date +%Y%m%d_%H%M%S)}"
WARMUP_ITERS="${WARMUP_ITERS:-3}"
WORKLOAD_PATH="${ROC_OUT_DIR}/${WORKLOAD_NAME}"
export WORKLOAD_PATH

PMC_DISPATCH_INFO="${WORKLOAD_PATH}/pmc_dispatch_info.csv"
PMC_PERF="${WORKLOAD_PATH}/pmc_perf.csv"
export PMC_DISPATCH_INFO PMC_PERF

# Find the ISA HTML generator: local copy first, then the sibling conv3amd repo.
FORMAT_ISA_PY="${SCRIPT_DIR}/rocprof_att_stats_to_isa_html.py"
if [[ ! -f "${FORMAT_ISA_PY}" ]]; then
  _sibling="$(cd "${ROOT}/.." && pwd)/conv3amd/rocprof_att_stats_to_isa_html.py"
  [[ -f "${_sibling}" ]] && FORMAT_ISA_PY="${_sibling}"
fi

MERGE_PY="${SCRIPT_DIR}/rocprof_merge_percentage_summary.py"
if [[ ! -f "${MERGE_PY}" ]]; then
  _sibling="$(cd "${ROOT}/.." && pwd)/conv3amd/rocprof_merge_percentage_summary.py"
  [[ -f "${_sibling}" ]] && MERGE_PY="${_sibling}"
fi

# Build ISA HTML from the ATT stats CSVs.
generate_isa_html() {
  local _utt="${ROCPROF_UI_TRACE_DIR:-${WORKLOAD_PATH}/ui_thread_trace}"
  [[ -f "${FORMAT_ISA_PY}" ]] || { echo "warning: ${FORMAT_ISA_PY} not found; skipping ISA HTML" >&2; return 0; }
  [[ -d "${_utt}" ]] || return 0
  local -a _files
  shopt -s nullglob
  _files=( "${_utt}"/stats_ui_output_agent_*_dispatch_*.csv )
  shopt -u nullglob
  (( ${#_files[@]} )) || return 0
  echo "Generating ISA HTML (${#_files[@]} stats CSV(s))..."
  "${PYTHON}" "${FORMAT_ISA_PY}" "${_files[@]}" || echo "warning: ISA HTML generation failed" >&2
}

# Check that rocprof-compute's Python dependencies are importable.
if ! "${PYTHON}" - <<'PY'
from importlib import metadata
for name in ("pyyaml", "PyYAML"):
    try: metadata.distribution(name); break
    except metadata.PackageNotFoundError: continue
else: raise SystemExit(1)
import pandas, pytz
PY
then
  echo "error: rocprof-compute Python deps missing for ${PYTHON}" >&2
  echo "  ${PYTHON} -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt" >&2
  exit 1
fi

export ROC_PROFILE_PYTHON="${PYTHON}"
export ROC_PROFILE_SCRIPT="${ROOT}/runner.py"
if [[ -z "${ROC_PROFILE_LD_PREFIX:-}" && -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export ROC_PROFILE_LD_PREFIX="${CONDA_PREFIX}/lib"
fi

if [[ -n "${ROC_PROFILE_RUNNER_ARGS:-}" ]]; then
  read -r -a ROC_PROFILE_RUNNER_ARGV <<<"${ROC_PROFILE_RUNNER_ARGS}"
else
  ROC_PROFILE_RUNNER_ARGV=(--warmup-iters "${WARMUP_ITERS}")
fi

# Write a throwaway app launcher: rocprofiler-sdk execs a fresh process and resets
# LD_LIBRARY_PATH, so a real script must re-prepend the conda lib path before runner.py.
_APP_SH="${SCRIPT_DIR}/.rocprof_app_$$.sh"
trap 'rm -f "${_APP_SH}"' EXIT
cat > "${_APP_SH}" <<'APPSH'
#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${ROC_PROFILE_LD_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${ROC_PROFILE_LD_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
exec "${ROC_PROFILE_PYTHON}" "${ROC_PROFILE_SCRIPT}" "$@"
APPSH
chmod +x "${_APP_SH}"

mkdir -p "${ROC_OUT_DIR}"

# --- PMC profile (every counter section) ---
echo "=== rocprof-compute PMC profile ==="
echo "  workload: ${WORKLOAD_NAME}  path: ${WORKLOAD_PATH}"
"${PYTHON}" "${ROCPROF_COMPUTE}" profile \
  -n "${WORKLOAD_NAME}" \
  -p "${WORKLOAD_PATH}" \
  -- \
  bash "${_APP_SH}" "${ROC_PROFILE_RUNNER_ARGV[@]}"

echo "PMC profile done: ${WORKLOAD_PATH}"

# --- ATT (Advanced Thread Trace), consumed by ROCprof Compute Viewer ---
if [[ -z "${ROCPROF_NO_UI_TRACE:-}" ]]; then
  RCV_UI_PARENT="${ROCPROF_UI_TRACE_DIR:-${WORKLOAD_PATH}/ui_thread_trace}"
  mkdir -p "${RCV_UI_PARENT}"
  echo "=== rocprofv3 ATT -> ${RCV_UI_PARENT} ==="
  # --att-serialize-all defaults on for CDNA3 to capture reliably; set ROCPROF_UI_ATT_SERIALIZE_ALL=0 to turn it off.
  _rv3=(rocprofv3 --advanced-thread-trace 1 --kernel-trace 1 --att-serialize-all 1 -d "${RCV_UI_PARENT}")
  if [[ -n "${ROCPROF_UI_ATT_LIBRARY_PATH:-}" ]]; then
    read -r -a _att_lib_dirs <<<"${ROCPROF_UI_ATT_LIBRARY_PATH//:/ }"
    _rv3+=(--att-library-path "${_att_lib_dirs[@]}")
  fi
  [[ -n "${ROCPROF_UI_ATT_TARGET_CU:-}" ]]   && _rv3+=(--att-target-cu  "${ROCPROF_UI_ATT_TARGET_CU}")
  [[ -n "${ROCPROF_UI_ATT_GPU_INDEX:-}" ]]   && _rv3+=(--att-gpu-index  "${ROCPROF_UI_ATT_GPU_INDEX}")
  [[ "${ROCPROF_UI_ATT_SERIALIZE_ALL:-1}" == "0" ]] && _rv3=(${_rv3[@]/--att-serialize-all 1/})

  _rv3_ok=0
  if "${_rv3[@]}" -- bash "${_APP_SH}" "${ROC_PROFILE_RUNNER_ARGV[@]}"; then
    _rv3_ok=1
  else
    echo "warning: rocprofv3 ATT failed (missing librocprof-trace-decoder.so?)" >&2
    echo "  Set ROCPROF_UI_ATT_LIBRARY_PATH or install ROCprof Trace Decoder." >&2
  fi

  _ui_dirs="$(find "${RCV_UI_PARENT}" -type d -name 'ui_output_agent_*' 2>/dev/null | head -5 || true)"
  if [[ -n "${_ui_dirs}" ]]; then
    echo "  RCV import dirs:"; while IFS= read -r _d; do [[ -n "${_d}" ]] && echo "    ${_d}"; done <<<"${_ui_dirs}"
  else
    echo "warning: no ui_output_agent_* found under ${RCV_UI_PARENT}" >&2
  fi
fi

# --- rocprof-compute analyze, then a summary CSV ---
if [[ -z "${ROCPROF_NO_SUMMARY:-}" ]]; then
  if [[ -f "${PMC_DISPATCH_INFO}" || -f "${PMC_PERF}" ]] && [[ -f "${MERGE_PY}" ]]; then
    # Derive the dispatch id of the attention_forward kernel.
    DISPATCH_ID="${ROCPROF_SUMMARY_DISPATCH:-}"
    if [[ -z "${DISPATCH_ID}" ]]; then
      DISPATCH_ID="$(
        "${PYTHON}" <<'PY'
import csv, os
from pathlib import Path
wp = Path(os.environ["WORKLOAD_PATH"])

def from_dispatch_info():
    p = wp / "pmc_dispatch_info.csv"
    if not p.is_file(): return ""
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
hits = [int(r["Dispatch_ID"]) for r in rows if any(k in (r.get("Kernel_Name") or "") for k in ("attention_forward", "attn_fwd_"))]
    if hits: return str(max(hits))
    if rows:
        try: return str(int(rows[-1]["Dispatch_ID"]))
        except (KeyError, ValueError): pass
    return ""

def from_pmc_perf():
    p = wp / "pmc_perf.csv"
    if not p.is_file(): return ""
    rows = list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    by_id = {int(r["Dispatch_ID"]): r.get("Kernel_Name","") for r in rows if r.get("Dispatch_ID","").isdigit()}
hits = [d for d,n in by_id.items() if any(k in n for k in ("attention_forward", "attn_fwd_"))]
    if hits: return str(max(hits))
    if by_id: return str(max(by_id.keys()))
    return ""

print((from_dispatch_info() or from_pmc_perf()), end="")
PY
      )"
    fi

    if [[ -n "${DISPATCH_ID}" ]]; then
      ANALYZE_DIR="${WORKLOAD_PATH}/analyze_tables"
      rm -rf "${ANALYZE_DIR}"
      echo "=== rocprof-compute analyze (dispatch ${DISPATCH_ID}, blocks 2 10 11) ==="
      if (
        cd "${WORKLOAD_PATH}"
        "${PYTHON}" "${ROCPROF_COMPUTE}" analyze \
          -p . -d "${DISPATCH_ID}" -b 2 10 11 \
          --output-format csv --output-name analyze_tables -q
      ); then
        SUMMARY_CSV="${WORKLOAD_PATH}/summary_percentages.csv"
        "${PYTHON}" "${MERGE_PY}" "${ANALYZE_DIR}" "${SUMMARY_CSV}"
        echo "  summary: ${SUMMARY_CSV}"
        [[ -n "${ROCPROF_RM_ANALYZE_TABLES:-}" ]] && rm -rf "${ANALYZE_DIR}"
      else
        echo "warning: rocprof-compute analyze failed; no summary_percentages.csv" >&2
      fi
    else
      echo "warning: could not find attention_forward dispatch id; skipping summary" >&2
    fi
  fi
fi

# --- ISA HTML report ---
generate_isa_html

# --- Package everything ---
if [[ -z "${NO_ZIP:-}" ]]; then
  ZIP_PATH="${ROC_OUT_DIR}/${WORKLOAD_NAME}.zip"
  echo "=== Packaging results -> ${ZIP_PATH} ==="
  # Bundle the workload dir plus the HIP source for viewer reference
  (
    cd "${ROC_OUT_DIR}"
    zip -r "${ZIP_PATH}" "${WORKLOAD_NAME}/"
  )
  # Keep the HIP source alongside so ISA HTML source references stay portable
  zip -j "${ZIP_PATH}" "${ROOT}/csrc/gfx942/attention_kernel.hip"
  echo "Done: ${ZIP_PATH}"
  echo "  Extract and open ui_thread_trace/stats_*_isa.html in a browser."
  echo "  For ROCprof Compute Viewer: File -> Import -> Rocprofv3 UI -> pick ui_thread_trace/"
fi
