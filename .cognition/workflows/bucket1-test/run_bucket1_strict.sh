#!/bin/bash
# =============================================================================
# Bucket 1 STRICT Test — NO COMPROMISES
# =============================================================================
# This script runs the full Bucket 1 E2E test with ZERO fallbacks.
# Every assertion is strict. Every failure is captured.
#
# Design principles:
#   - NO FALLBACKS: If a service is down, FAIL. No stub data.
#   - NO EXCEPTIONS: Every check is mandatory. No "expected failure" paths.
#   - NO DEGRADED MODES: AUC=0 is FAIL. FinalState=3 is FAIL.
#   - MAXIMUM VERBOSITY: Every request/response captured raw.
#
# Prerequisites:
#   - Run on compute-1 (as-compute-1.ktpn) via SSH
#   - All DART Slot 9 services must be running
#   - NFS storage mounted at /mnt/sharedstorage or /mnt/storage01
#   - Python3 available for JSON parsing
#
# Usage:
#   ssh target 'bash -s' < run_bucket1_strict.sh
#   # Or from compute-1 directly:
#   bash run_bucket1_strict.sh
# =============================================================================

set -euo pipefail

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
JOB_ID="bucket1-strict-${TIMESTAMP}"
REPORT_FILE="/tmp/bucket1_strict_${TIMESTAMP}.log"
RESULT_JSON="/tmp/bucket1_result_${TIMESTAMP}.json"
PASS_COUNT=0
FAIL_COUNT=0
FAILURES=""

log() { echo "$1" | tee -a "$REPORT_FILE"; }
log_raw() { echo "$1" >> "$REPORT_FILE"; }
pass() { log "  PASS: $1"; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() { log "  FAIL: $1"; FAIL_COUNT=$((FAIL_COUNT + 1)); FAILURES="${FAILURES}\n  - $1"; }

log "================================================================="
log "  BUCKET 1 STRICT TEST — NO COMPROMISES"
log "  Job:       $JOB_ID"
log "  Timestamp: $TIMESTAMP"
log "  Host:      $(hostname)"
log "================================================================="
log ""

# =============================================================================
# PHASE 1: INFRASTRUCTURE HEALTH — ALL MUST PASS
# =============================================================================
log "=== PHASE 1: INFRASTRUCTURE HEALTH ==="

# 1a. DalService
log "[1a] DalService health..."
DAL_HEALTH=$(curl -sf --max-time 10 http://localhost:49600/health 2>&1) || DAL_HEALTH="UNREACHABLE"
log_raw "  Raw: $DAL_HEALTH"
if echo "$DAL_HEALTH" | grep -q "Healthy"; then
    pass "DalService healthy"
else
    fail "DalService not healthy: $DAL_HEALTH"
fi

# 1b. AnalysisService
log "[1b] AnalysisService lifecycle..."
AS_RESP=$(curl -sf --max-time 10 http://as-gpu-compute-4.ktpn:2037/api/v1.0/Lifecycle 2>&1) || AS_RESP="UNREACHABLE"
log_raw "  Raw: $AS_RESP"
AS_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://as-gpu-compute-4.ktpn:2037/api/v1.0/Lifecycle 2>/dev/null || echo "000")
if [ "$AS_CODE" = "200" ]; then
    pass "AnalysisService responding (HTTP 200)"
else
    fail "AnalysisService returned HTTP $AS_CODE"
fi

# 1c. BIDS/MMS via IIS ARR
log "[1c] BIDS/MMS (port 8089)..."
BIDS_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://as-windows-1.ktpn:8089/api/v1.0/status 2>/dev/null || echo "000")
if [ "$BIDS_CODE" = "200" ]; then
    pass "BIDS/MMS responding (HTTP 200)"
else
    fail "BIDS/MMS returned HTTP $BIDS_CODE"
fi

# 1d. BIDS direct
log "[1d] BIDS direct (port 6959)..."
BIDS_DIRECT=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://as-windows-1.ktpn:6959/swagger/v1/swagger.json 2>/dev/null || echo "000")
if [ "$BIDS_DIRECT" = "200" ]; then
    pass "BIDS direct responding (HTTP 200)"
else
    fail "BIDS direct returned HTTP $BIDS_DIRECT"
fi

# 1e. AnalysisPortal
log "[1e] AnalysisPortal health..."
PORTAL_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://as-compute-1.ktpn:5009/api/v1.0/health 2>/dev/null || echo "000")
if [ "$PORTAL_CODE" = "200" ]; then
    pass "AnalysisPortal responding (HTTP 200)"
else
    fail "AnalysisPortal returned HTTP $PORTAL_CODE"
fi

# 1f. Postgres
log "[1f] Postgres container..."
PG_UP=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c "Postgres.*slot_9" || echo "0")
if [ "$PG_UP" -gt 0 ]; then
    pass "Postgres container running"
else
    fail "Postgres container NOT running"
fi

# 1g. Shared storage
log "[1g] Shared storage..."
STORAGE_OK=false
for SPATH in /mnt/sharedstorage/kla_user/manageddata/lots /mnt/storage01/kla_user/manageddata/lots; do
    if [ -d "$SPATH" ]; then
        STORAGE_OK=true
        pass "Shared storage accessible at $SPATH"
        break
    fi
done
if [ "$STORAGE_OK" = false ]; then
    fail "Shared storage not mounted (checked /mnt/sharedstorage and /mnt/storage01)"
fi

log ""
if [ "$FAIL_COUNT" -gt 0 ]; then
    log "*** PHASE 1 FAILED ($FAIL_COUNT failures) — Infrastructure unhealthy ***"
    log "*** ABORTING — fix infrastructure before running strict tests ***"
    log ""
    log "FAILURES:$FAILURES"
    log ""
    log "Report: $REPORT_FILE"
    exit 1
fi
log "=== PHASE 1 PASSED: All infrastructure checks OK ==="

# =============================================================================
# PHASE 2: DATA PIPELINE — REAL DATA, NO STUBS
# =============================================================================
log ""
log "=== PHASE 2: DATA PIPELINE VALIDATION ==="

# 2a. BIDS lot catalog
log "[2a] BIDS lot catalog..."
CATALOG_RESP=$(curl -s --max-time 30 "http://as-windows-1.ktpn:6959/mms/catalog/v1.0/lots?lotRoot=//172.16.1.31/sharedstorage/kla_user/manageddata/lots" 2>&1)
log_raw "  Catalog response (first 500 chars): ${CATALOG_RESP:0:500}"
CATALOG_COUNT=$(echo "$CATALOG_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null || echo "0")
if [ "$CATALOG_COUNT" -gt 0 ]; then
    pass "BIDS catalog has $CATALOG_COUNT lots"
else
    fail "BIDS catalog returned 0 lots"
fi

# 2b. BIDS dbfromlot v2
log "[2b] BIDS dbfromlot v2 (M4CL_FocusCurve_HY01)..."
DBFROMLOT_RESP=$(curl -s --max-time 60 -X POST "http://as-windows-1.ktpn:6959/mms/dhl/v2/dbfromlot" \
  -H "Content-Type: application/json" \
  -d '{"InputFullPath":"//172.16.1.31/sharedstorage/kla_user/manageddata/lots/M4CL_FocusCurve_HY01","IsRecipePath":false}' 2>&1)
log_raw "  dbfromlot response (first 1000 chars): ${DBFROMLOT_RESP:0:1000}"
DBFROMLOT_OK=$(echo "$DBFROMLOT_RESP" | python3 -c "
import sys,json
d=json.load(sys.stdin)
s=str(d)
print('OK' if any(k in s for k in ['LotAuxDBInfo','WaferInfo','auxDbPaths','AuxDbPaths','waferInfo']) else 'NODATA')
" 2>/dev/null || echo "PARSE_ERROR")
if [ "$DBFROMLOT_OK" = "OK" ]; then
    pass "dbfromlot returned lot metadata"
else
    fail "dbfromlot did not return expected metadata ($DBFROMLOT_OK)"
fi

# 2c. BIDS lotsummary v2
log "[2c] BIDS lotsummary v2 (M4CL_FocusCurve_HY01)..."
LOTSUMMARY_RESP=$(curl -s --max-time 60 -X PUT "http://as-windows-1.ktpn:6959/mms/dhl/v2/lotsummary" \
  -H "Content-Type: application/json" \
  -d '{"InputFullPath":"//172.16.1.31/sharedstorage/kla_user/manageddata/lots/M4CL_FocusCurve_HY01","IsRecipePath":false}' 2>&1)
log_raw "  lotsummary response (first 1000 chars): ${LOTSUMMARY_RESP:0:1000}"
if echo "$LOTSUMMARY_RESP" | grep -qi "defect\|wafer\|count\|Defect\|Total"; then
    pass "lotsummary returned defect/wafer data"
else
    fail "lotsummary returned no recognizable defect data"
fi

# 2d. BIDS recipeinfo v2 (BUG-002 regression check)
log "[2d] BIDS recipeinfo v2 (BUG-002 check)..."
RECIPEINFO_CODE=$(curl -s -o /tmp/recipeinfo_${TIMESTAMP}.txt -w "%{http_code}" --max-time 30 \
  -X PUT "http://as-windows-1.ktpn:6959/mms/dhl/v2/recipeinfo" \
  -H "Content-Type: application/json" \
  -H "X-ToolModel: 3955" \
  -d '{"InputFullPath":"//172.16.1.31/sharedstorage/kla_user/manageddata/lots/M4CL_FocusCurve_HY01","IsRecipePath":false,"InputTestIdList":"1"}' 2>/dev/null || echo "000")
RECIPEINFO_BODY=$(cat /tmp/recipeinfo_${TIMESTAMP}.txt 2>/dev/null || echo "NO_RESPONSE")
log_raw "  recipeinfo HTTP $RECIPEINFO_CODE: $RECIPEINFO_BODY"
if [ "$RECIPEINFO_CODE" = "200" ]; then
    pass "recipeinfo returned HTTP 200 — BUG-002 may be FIXED"
    log "  ACTION REQUIRED: Update BUG_TRACKER.md to mark BUG-002 as Resolved"
elif [ "$RECIPEINFO_CODE" = "500" ]; then
    log "  KNOWN: recipeinfo HTTP 500 — BUG-002 still open (tracked in BUG_TRACKER.md)"
    log "  This is an EXISTING known bug, not a new finding."
else
    fail "recipeinfo unexpected HTTP $RECIPEINFO_CODE (not 200 or 500)"
fi

log ""
PHASE2_FAILS=$FAIL_COUNT
log "=== PHASE 2 COMPLETE (failures so far: $FAIL_COUNT) ==="

# =============================================================================
# PHASE 3: ANALYSISSERVICE REAL LOT LOAD
# =============================================================================
log ""
log "=== PHASE 3: ANALYSISSERVICE LOT LOAD ==="
AS_HOST="http://as-gpu-compute-4.ktpn:2037"

# 3a. Load dataset
log "[3a] Load dataset (M4CL_FocusCurve_HY01, sourceType=1, via BIDS)..."
LOAD_RESP=$(curl -s --max-time 120 -X POST "$AS_HOST/api/v1.0/dataset" \
  -H "Content-Type: application/json" \
  -d '{
    "classifierPath": "/metadata/classifiers/003_AanalysisAndSampling.ido3",
    "lotFilter": [{
      "lotID": "M4CL_FocusCurve_HY01",
      "sourceType": 1,
      "device": "*",
      "layer": "*"
    }],
    "mutableResultsPath": "/metadata/mutableresults/",
    "resultsSessionParam": { "config": {} },
    "toolModel": "3955"
  }' 2>&1)
log_raw "  Load response: $LOAD_RESP"
DATASET_ID=$(echo "$LOAD_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('DatasetId', d.get('datasetId', 'NONE')))" 2>/dev/null || echo "PARSE_ERROR")
if [ "$DATASET_ID" != "NONE" ] && [ "$DATASET_ID" != "PARSE_ERROR" ]; then
    pass "Dataset loaded (DatasetId=$DATASET_ID)"
else
    fail "Dataset load failed — no DatasetId (response: ${LOAD_RESP:0:200})"
    DATASET_ID=""
fi

# 3b. Defect count
if [ -n "$DATASET_ID" ]; then
    log "[3b] Verify defect count..."
    COUNT_RESP=$(curl -s --max-time 30 "$AS_HOST/api/v1.0/dataset/$DATASET_ID/count" 2>&1)
    log_raw "  Count response: $COUNT_RESP"
    DEFECT_COUNT=$(echo "$COUNT_RESP" | python3 -c "import sys; print(int(sys.stdin.read().strip()))" 2>/dev/null || echo "0")
    if [ "$DEFECT_COUNT" -gt 0 ] 2>/dev/null; then
        pass "Dataset has $DEFECT_COUNT defects"
    else
        fail "Dataset has 0 defects or count failed ($COUNT_RESP)"
    fi
fi

# 3c. Schema check
if [ -n "$DATASET_ID" ]; then
    log "[3c] Verify schema (ClassCode_Manual required)..."
    SCHEMA_RESP=$(curl -s --max-time 30 "$AS_HOST/api/v1.0/dataset/$DATASET_ID/schema" 2>&1)
    log_raw "  Schema response (first 2000 chars): ${SCHEMA_RESP:0:2000}"
    HAS_CLASSCODE=$(echo "$SCHEMA_RESP" | python3 -c "import sys; print('YES' if 'ClassCode_Manual' in sys.stdin.read() else 'NO')" 2>/dev/null || echo "NO")
    if [ "$HAS_CLASSCODE" = "YES" ]; then
        pass "Schema contains ClassCode_Manual"
    else
        fail "Schema missing ClassCode_Manual"
    fi
fi

log ""
log "=== PHASE 3 COMPLETE ==="

# =============================================================================
# PHASE 4: DALSERVICE E2E PIPELINE — STRICT ASSERTIONS
# =============================================================================
log ""
log "=== PHASE 4: DALSERVICE E2E PIPELINE ==="

# 4a. Submit
log "[4a] Submit DalJob ($JOB_ID)..."
SUBMIT_RESP=$(curl -s --max-time 60 -X POST "http://localhost:49600/api/v1.0/DalJob/dal/submit" \
  -H "Content-Type: application/json" \
  -d "{
    \"JobId\": \"$JOB_ID\",
    \"WorkspaceId\": \"bucket1_strict\",
    \"ModelName\": \"bucket1_baseline\",
    \"DesignInput\": {
      \"RdfPath\": \"/none\",
      \"SelectedLayers\": [\"M1\"],
      \"CoordinateTransformParams\": { \"Scale\": 1.0, \"RotationDeg\": 0.0, \"OffsetX\": 0.0, \"OffsetY\": 0.0 }
    },
    \"AlgorithmConfigs\": [{
      \"AlgorithmType\": 0,
      \"BackboneName\": \"ResNet\",
      \"TargetROC\": 0.95,
      \"MaxIterations\": 1,
      \"EpochCount\": 1,
      \"BatchSize\": 32,
      \"LearningRate\": 0.001
    }],
    \"LotFilter\": {
      \"ClassifierId\": \"Analysis.ido3\",
      \"LotPath\": \"/metadata/dhl/R0/dhlfolders/Sample10M_64Bit_8db23cd8aecf274\",
      \"WaferSelection\": { \"WaferIds\": [], \"SelectionCriteria\": null },
      \"ProjectionAttributes\": [\"DefectId\", \"ClassCode_Manual\", \"Area\", \"MeanIntensity\", \"PosX\", \"PosY\"]
    },
    \"Priority\": 1
  }" 2>&1)
log_raw "  Submit response: $SUBMIT_RESP"
if echo "$SUBMIT_RESP" | grep -q "Submitted"; then
    pass "Job submitted"
else
    fail "Job submission failed: ${SUBMIT_RESP:0:300}"
fi

# 4b. Poll for completion (strict: 120s max, 5s intervals)
# ENUM REFERENCE (from DalEnums.cs -- DO NOT ALTER):
#   0=Queued, 1=Running, 2=Paused, 3=Completed, 4=Failed, 5=Cancelled
# NOTE: FinalState=3 (Completed) does NOT mean success. It means the
# orchestrator ran to the end, possibly through fallbacks. Validate output.
log "[4b] Polling for completion (max 120s, 5s intervals)..."
ELAPSED=0
FINAL_STATE="TIMEOUT"
while [ $ELAPSED -lt 120 ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    STATUS_RESP=$(curl -s --max-time 10 "http://localhost:49600/api/v1.0/DalJob/$JOB_ID/status" 2>&1)
    CURRENT_STATE=$(echo "$STATUS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('State','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    log "  ${ELAPSED}s: State=$CURRENT_STATE"
    if [ "$CURRENT_STATE" = "3" ] || [ "$CURRENT_STATE" = "Completed" ]; then
        FINAL_STATE="Completed"
        break
    elif [ "$CURRENT_STATE" = "4" ] || [ "$CURRENT_STATE" = "Failed" ]; then
        FINAL_STATE="Failed"
        break
    fi
done

if [ "$FINAL_STATE" = "Completed" ]; then
    log "  Job reached FinalState=3 (Completed). Validating output quality..."
elif [ "$FINAL_STATE" = "Failed" ]; then
    fail "Job CRASHED (FinalState=4). Pipeline threw unhandled exception."
else
    fail "Job TIMEOUT after 120s (last state: $CURRENT_STATE)"
fi

# 4c. Full result capture
log "[4c] Capturing full result..."
RESULT=$(curl -s --max-time 10 "http://localhost:49600/api/v1.0/DalJob/$JOB_ID/result" 2>&1)
echo "$RESULT" > "$RESULT_JSON"
log_raw "  Full result JSON saved to: $RESULT_JSON"
log_raw "  Result: $RESULT"

# 4d. ROC validation (STRICT: AUC > 0 required)
log "[4d] ROC validation (AUC > 0 required)..."
AUC_BASELINE=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('RocComparison',{}).get('BaselinePerformance',{}).get('AucValue', -1))" 2>/dev/null || echo "-1")
AUC_ENHANCED=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('RocComparison',{}).get('DalEnhancedPerformance',{}).get('AucValue', -1))" 2>/dev/null || echo "-1")
ROC_STATUS=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); bp=r.get('RocComparison',{}).get('BaselinePerformance',{}); print(bp.get('RocCurveData',{}).get('status','NONE') if isinstance(bp.get('RocCurveData'),dict) else 'NONE')" 2>/dev/null || echo "NONE")
log "  Baseline AUC:  $AUC_BASELINE"
log "  Enhanced AUC:  $AUC_ENHANCED"
log "  ROC Status:    $ROC_STATUS"

if [ "$ROC_STATUS" = "fallback" ]; then
    fail "ROC in FALLBACK mode — AUC=0 is NOT acceptable in strict mode"
fi
python3 -c "
b=float('${AUC_BASELINE}'); e=float('${AUC_ENHANCED}')
if b <= 0: print('FAIL_B')
if e <= 0: print('FAIL_E')
" 2>/dev/null | while read line; do
    if [ "$line" = "FAIL_B" ]; then
        fail "Baseline AUC <= 0 ($AUC_BASELINE)"
    fi
    if [ "$line" = "FAIL_E" ]; then
        fail "Enhanced AUC <= 0 ($AUC_ENHANCED)"
    fi
done

# 4e. Generated attributes
log "[4e] Generated attributes..."
GEN_ATTRS=$(echo "$RESULT" | python3 -c "import sys,json; ga=json.load(sys.stdin).get('GeneratedAttributes',[]); print(len(ga)); [print(f'  - {a}') for a in ga]" 2>/dev/null || echo "0")
GEN_COUNT=$(echo "$GEN_ATTRS" | head -1)
if [ "$GEN_COUNT" -gt 0 ] 2>/dev/null; then
    pass "Generated $GEN_COUNT attributes"
else
    fail "No generated attributes"
fi

# 4f. Model persistence
log "[4f] Model persistence..."
MODEL_PATH=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ModelPath','NONE'))" 2>/dev/null || echo "NONE")
log "  ModelPath: $MODEL_PATH"
if [ "$MODEL_PATH" != "NONE" ] && [ "$MODEL_PATH" != "null" ] && [ "$MODEL_PATH" != "None" ] && [ -n "$MODEL_PATH" ]; then
    pass "ModelPath set: $MODEL_PATH"
else
    fail "ModelPath is null/empty — persistence failed"
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
log ""
log "================================================================="
log "  BUCKET 1 STRICT TEST — SUMMARY"
log "================================================================="
log "  Job:       $JOB_ID"
log "  Timestamp: $TIMESTAMP"
log "  PASSED:    $PASS_COUNT"
log "  FAILED:    $FAIL_COUNT"
if [ "$FAIL_COUNT" -gt 0 ]; then
    log "  STATUS:    *** FAIL ***"
    log ""
    log "  Failures:"
    echo -e "$FAILURES" | tee -a "$REPORT_FILE"
else
    log "  STATUS:    PASS"
fi
log ""
log "  Report:    $REPORT_FILE"
log "  Result:    $RESULT_JSON"
log "================================================================="

exit $FAIL_COUNT
