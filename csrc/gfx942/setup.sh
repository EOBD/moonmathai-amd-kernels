#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building attention_tester ==="
make -j"$(nproc)"

ATTN_BIN="$SCRIPT_DIR/build/attention_tester"
MLA_BIN="$SCRIPT_DIR/build/mla_decode_a16w8_tester"
echo "=== Build OK: $ATTN_BIN ==="
echo "=== Build OK: $MLA_BIN ==="

echo ""
"audit_spills.py" "$ATTN_BIN" || {
  echo "=== audit_spills.sh reported spills (or failed). See output above. ==="
}
"audit_spills.py" "$MLA_BIN" || {
  echo "=== audit_spills.sh reported spills (or failed). See output above. ==="
}