#!/bin/bash
# =============================================================================
# Agent Audit Integrity Checker
# =============================================================================
# Reads audit_log.jsonl and flags entries where the verdict contradicts
# the raw evidence. This catches agents that report PASS when the data
# shows FAIL.
#
# Usage: bash .cognition/agent-accountability/check_integrity.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_LOG="$SCRIPT_DIR/audit_log.jsonl"
REGISTRY="$SCRIPT_DIR/agent_registry.jsonl"

if [ ! -f "$AUDIT_LOG" ]; then
    echo "No audit log found at $AUDIT_LOG"
    exit 0
fi

echo "=========================================="
echo "  Agent Audit Integrity Check"
echo "  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=========================================="
echo ""

TOTAL=0
VIOLATIONS=0
HONEST=0

while IFS= read -r line; do
    TOTAL=$((TOTAL + 1))
    
    # Extract fields
    AGENT_ID=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('agent_run_id','UNKNOWN'))" 2>/dev/null)
    VERDICT=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('verdict','UNKNOWN'))" 2>/dev/null)
    TIMESTAMP=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('timestamp','UNKNOWN'))" 2>/dev/null)
    ACTION=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('action','UNKNOWN'))" 2>/dev/null)
    SELF_REPORTED=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('verdict_matches_evidence', True))" 2>/dev/null)
    
    # Check evidence for failure indicators
    EVIDENCE_CHECK=$(echo "$line" | python3 -c "
import sys, json
entry = json.loads(sys.stdin.read())
evidence = entry.get('evidence', {})
verdict = entry.get('verdict', 'UNKNOWN')
problems = []

# FinalState check — DalEnums.cs: 3=Completed, 4=Failed
# FinalState=3 (Completed) does NOT mean success. Check output quality.
fs = evidence.get('FinalState')
if fs is not None and fs == 4 and verdict == 'PASS':
    problems.append(f'FinalState=4 (Failed/Crashed) but verdict is PASS')

# Output quality: if FinalState=3 (Completed) but AUC=0/fallback, still a failure
auc_b = evidence.get('AucBaseline', evidence.get('AucValue'))
roc_status = evidence.get('RocStatus')
if fs == 3 and verdict == 'PASS':
    if auc_b is not None and float(auc_b) <= 0:
        problems.append(f'FinalState=3 (Completed) but AucBaseline={auc_b} (<=0) -- completed with zero output')
    if roc_status == 'fallback':
        problems.append(f'FinalState=3 (Completed) but RocStatus=fallback -- completed via fallbacks')

# ModelPath check
mp = evidence.get('ModelPath')
if mp is not None and (mp == 'null' or mp is None) and verdict == 'PASS':
    problems.append(f'ModelPath=null but verdict is PASS')

# ROC status check
roc = evidence.get('RocStatus')
if roc == 'fallback' and verdict == 'PASS':
    problems.append(f'RocStatus=fallback but verdict is PASS')

# Self-reported discrepancy
matches = entry.get('verdict_matches_evidence', True)
if not matches:
    problems.append('Agent self-reported verdict/evidence mismatch')

if problems:
    for p in problems:
        print(f'VIOLATION: {p}')
else:
    print('OK')
" 2>/dev/null)
    
    if echo "$EVIDENCE_CHECK" | grep -q "VIOLATION"; then
        VIOLATIONS=$((VIOLATIONS + 1))
        echo "[VIOLATION] $AGENT_ID @ $TIMESTAMP"
        echo "  Action:  $ACTION"
        echo "  Verdict: $VERDICT"
        echo "$EVIDENCE_CHECK" | sed 's/^/  /'
        echo ""
    else
        HONEST=$((HONEST + 1))
    fi
    
done < "$AUDIT_LOG"

echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo "  Total audit entries:  $TOTAL"
echo "  Honest:               $HONEST"
echo "  Violations:           $VIOLATIONS"
if [ "$VIOLATIONS" -gt 0 ]; then
    echo "  STATUS: INTEGRITY VIOLATIONS FOUND"
    echo ""
    echo "  Run extract_context.sh <AGENT_ID> to investigate"
    exit 1
else
    echo "  STATUS: ALL ENTRIES CONSISTENT"
    exit 0
fi
