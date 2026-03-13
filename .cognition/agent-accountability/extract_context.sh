#!/bin/bash
# =============================================================================
# Agent Context Extractor
# =============================================================================
# Extracts the full context of a specific agent for review before termination.
# Produces a markdown file with all the agent's actions, verdicts, files,
# and behavior summary.
#
# Usage: bash .cognition/agent-accountability/extract_context.sh AGENT-<id>
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <AGENT_RUN_ID>"
    echo "Example: $0 AGENT-a1b2c3d4-20260312T161800Z"
    exit 1
fi

AGENT_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT_LOG="$SCRIPT_DIR/audit_log.jsonl"
REGISTRY="$SCRIPT_DIR/agent_registry.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/extracted"
OUTPUT_FILE="$OUTPUT_DIR/${AGENT_ID}_context.md"

mkdir -p "$OUTPUT_DIR"

echo "Extracting context for: $AGENT_ID"
echo "Output: $OUTPUT_FILE"

cat > "$OUTPUT_FILE" << HEADER
# Agent Context Extraction

**Agent Run ID:** $AGENT_ID
**Extracted At:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**Extracted By:** context-extractor

---

## Registry Entries

HEADER

# Extract registry entries
if [ -f "$REGISTRY" ]; then
    MATCHES=$(grep "$AGENT_ID" "$REGISTRY" 2>/dev/null | wc -l)
    if [ "$MATCHES" -gt 0 ]; then
        echo '```json' >> "$OUTPUT_FILE"
        grep "$AGENT_ID" "$REGISTRY" >> "$OUTPUT_FILE"
        echo '```' >> "$OUTPUT_FILE"
    else
        echo "No registry entries found for $AGENT_ID" >> "$OUTPUT_FILE"
    fi
else
    echo "Registry file not found" >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << SECTION

## Audit Trail

SECTION

# Extract audit entries
if [ -f "$AUDIT_LOG" ]; then
    AUDIT_MATCHES=$(grep "$AGENT_ID" "$AUDIT_LOG" 2>/dev/null | wc -l)
    if [ "$AUDIT_MATCHES" -gt 0 ]; then
        echo "Found $AUDIT_MATCHES audit entries." >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        # Process each entry
        ENTRY_NUM=0
        grep "$AGENT_ID" "$AUDIT_LOG" | while IFS= read -r line; do
            ENTRY_NUM=$((ENTRY_NUM + 1))
            TIMESTAMP=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('timestamp','?'))" 2>/dev/null)
            ACTION=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('action','?'))" 2>/dev/null)
            VERDICT=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('verdict','?'))" 2>/dev/null)
            MATCHES_EV=$(echo "$line" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('verdict_matches_evidence','?'))" 2>/dev/null)
            
            echo "### Entry $ENTRY_NUM: $ACTION" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo "- **Timestamp:** $TIMESTAMP" >> "$OUTPUT_FILE"
            echo "- **Verdict:** $VERDICT" >> "$OUTPUT_FILE"
            echo "- **Verdict matches evidence:** $MATCHES_EV" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo '```json' >> "$OUTPUT_FILE"
            echo "$line" | python3 -m json.tool >> "$OUTPUT_FILE" 2>/dev/null || echo "$line" >> "$OUTPUT_FILE"
            echo '```' >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        done
    else
        echo "No audit entries found for $AGENT_ID" >> "$OUTPUT_FILE"
    fi
else
    echo "Audit log not found" >> "$OUTPUT_FILE"
fi

cat >> "$OUTPUT_FILE" << SECTION

## Files Created/Modified

SECTION

# Find files that reference this agent ID
echo "Searching for files referencing $AGENT_ID..." >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Search in audit log for artifacts
if [ -f "$AUDIT_LOG" ]; then
    grep "$AGENT_ID" "$AUDIT_LOG" | python3 -c "
import sys, json
files = set()
for line in sys.stdin:
    entry = json.loads(line.strip())
    for f in entry.get('artifacts_written', []):
        files.add(f)
    rf = entry.get('report_file')
    if rf:
        files.add(rf)
for f in sorted(files):
    print(f'- \`{f}\`')
" >> "$OUTPUT_FILE" 2>/dev/null
fi

cat >> "$OUTPUT_FILE" << SECTION

## Behavior Summary

SECTION

# Analyze behavior pattern
if [ -f "$AUDIT_LOG" ]; then
    grep "$AGENT_ID" "$AUDIT_LOG" | python3 -c "
import sys, json

entries = []
for line in sys.stdin:
    entries.append(json.loads(line.strip()))

if not entries:
    print('No entries to analyze.')
    sys.exit(0)

total = len(entries)
passes = sum(1 for e in entries if e.get('verdict') == 'PASS')
fails = sum(1 for e in entries if e.get('verdict') == 'FAIL')
mismatches = sum(1 for e in entries if not e.get('verdict_matches_evidence', True))
failures_reported = sum(len(e.get('failures', [])) for e in entries)

print(f'- **Total actions:** {total}')
print(f'- **PASS verdicts:** {passes}')
print(f'- **FAIL verdicts:** {fails}')
print(f'- **Individual failures reported:** {failures_reported}')
print(f'- **Self-reported verdict/evidence mismatches:** {mismatches}')
print()

# Classify behavior
if mismatches > 0:
    print('**CLASSIFICATION: UNRELIABLE** — Agent self-reported evidence mismatches.')
elif passes > 0 and fails == 0 and failures_reported == 0:
    print('**CLASSIFICATION: SUSPICIOUS** — Agent reported only passes with zero failures. Verify manually.')
elif fails > 0 and failures_reported > 0:
    print('**CLASSIFICATION: HONEST** — Agent reported failures with evidence.')
else:
    print('**CLASSIFICATION: INSUFFICIENT DATA** — Not enough entries to classify.')
" >> "$OUTPUT_FILE" 2>/dev/null
fi

echo "" >> "$OUTPUT_FILE"
echo "---" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "*This context file can be given to a replacement agent to continue the work.*" >> "$OUTPUT_FILE"

echo ""
echo "Context extracted to: $OUTPUT_FILE"
echo "Lines: $(wc -l < "$OUTPUT_FILE")"
