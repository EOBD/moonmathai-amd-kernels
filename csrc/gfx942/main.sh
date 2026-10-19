#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ATTN=${RUN_ATTN:-1}
RUN_MLA=${RUN_MLA:-1}

echo "================================================================================"
echo "moonmath — full test suite"
echo "  attention        : $([[ "$RUN_ATTN" == "1" ]] && echo enabled || echo disabled)"
echo "  MLA A16W8        : $([[ "$RUN_MLA"  == "1" ]] && echo enabled || echo disabled)"
echo "================================================================================"

rc=0

if [[ "$RUN_ATTN" == "1" ]]; then
    echo ""
    echo "────── attention ──────"
    if ! "$SCRIPT_DIR/main_attn.sh"; then
        echo "ERROR: attention suite failed" >&2
        rc=1
    fi
fi

# if [[ "$RUN_MLA" == "1" ]]; then
#     echo ""
#     echo "────── MLA A16W8 ──────"
#     if ! "$SCRIPT_DIR/main_mla.sh"; then
#         echo "ERROR: MLA suite failed" >&2
#         rc=1
#     fi
# fi

echo ""
echo "================================================================================"
echo "=== Full test suite complete (exit $rc)"
echo "================================================================================"
exit $rc
