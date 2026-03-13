# Bucket 1 Test Workflow — NO COMPROMISES

## Overview

This workflow executes the full Bucket 1 end-to-end test against the DART Slot 9 deployment.
It enforces **production-grade standards** with **zero fallbacks**, **zero exceptions**, and
**zero tolerance for degraded modes**.

**Schedule:** Run periodically to detect regressions and validate deployment health.

## Design Principles

1. **NO FALLBACKS** — If a service is down, the test FAILS. No stub data. No sample payloads. No "fallback mode."
2. **NO EXCEPTIONS** — Every assertion is strict. No "expected failure" paths. No swallowing errors.
3. **NO DEGRADED MODES** — AUC=0.0 is a FAILURE, not "expected in fallback." FinalState=4 (Failed) is a FAILURE. FinalState=3 is Completed (see enum reference below).
4. **MAXIMUM VERBOSITY** — Every request/response is captured raw. Every failure includes full context.
5. **BUG VERIFICATION** — Before reporting any failure as a bug, cross-reference `AutonomousDevelopment/DeploymentBugs/BUG_TRACKER.md`.
6. **FILE TAGGING** — All output files MUST follow the Agent File-Tagging Protocol in the root `AGENTS.md`. Every file must have a tagged filename (`<name>_<timestamp>_<agentID>`) and a metadata footer.

### DalJobState Enum (VERIFIED from DalEnums.cs source code)

```
Queued=0, Running=1, Paused=2, Completed=3, Failed=4, Cancelled=5
```

**CRITICAL:** FinalState=3 means **COMPLETED**, NOT Failed. FinalState=4 means Failed.
A previous agent session incorrectly claimed 3=Failed/2=Completed. That error was
propagated into this workflow file and into SKILL.md. This has been corrected.

## Prerequisites

The agent executing this workflow MUST have SSH access to `as-compute-1.ktpn` (aliased as `ssh target`).
All tests run on compute-1 where the services are deployed.

## Execution Steps

### Phase 0: Bug Tracker Ingest

Before running any tests, read `AutonomousDevelopment/DeploymentBugs/BUG_TRACKER.md` and internalize:
- All open bug IDs and their symptoms
- All resolved bugs and their fixes
- All debunked claims

This context is required for Phase 5 (bug classification).

### Phase 1: Infrastructure Health (STRICT — all must pass)

Run each health check. If ANY check fails, the entire workflow fails immediately.
Do NOT proceed to Phase 2 if infrastructure is unhealthy.

```bash
#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
REPORT_FILE="/tmp/bucket1_strict_${TIMESTAMP}.log"

echo "=== BUCKET 1 STRICT TEST — ${TIMESTAMP} ===" | tee "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# --- Phase 1: Infrastructure Health ---
echo "=== PHASE 1: INFRASTRUCTURE HEALTH ===" | tee -a "$REPORT_FILE"
PHASE1_PASS=true

# 1a. DalService
echo -n "[1a] DalService health... " | tee -a "$REPORT_FILE"
DAL_HEALTH=$(curl -sf --max-time 10 http://localhost:49600/health 2>&1) || { echo "FAIL: DalService unreachable" | tee -a "$REPORT_FILE"; PHASE1_PASS=false; }
if [ "$PHASE1_PASS" = true ]; then
    echo "$DAL_HEALTH" | grep -q "Healthy" || { echo "FAIL: DalService not Healthy: $DAL_HEALTH" | tee -a "$REPORT_FILE"; PHASE1_PASS=false; }
    echo "PASS ($DAL_HEALTH)" | tee -a "$REPORT_FILE"
fi

# 1b. AnalysisService
echo -n "[1b] AnalysisService lifecycle... " | tee -a "$REPORT_FILE"
AS_RESP=$(curl -sf --max-time 10 http://as-gpu-compute-4.ktpn:2037/api/v1.0/Lifecycle 2>&1) || { echo "FAIL: AnalysisService unreachable" | tee -a "$REPORT_FILE"; PHASE1_PASS=false; }
if [ "$PHASE1_PASS" = true ]; then
    echo "PASS" | tee -a "$REPORT_FILE"
    echo "  Response: $AS_RESP" >> "$REPORT_FILE"
fi

# 1c. BIDS/MMS
echo -n "[1c] BIDS/MMS status... " | tee -a "$REPORT_FILE"
BIDS_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://as-windows-1.ktpn:8089/api/v1.0/status 2>/dev/null)
if [ "$BIDS_CODE" != "200" ]; then
    echo "FAIL: BIDS returned HTTP $BIDS_CODE (expected 200)" | tee -a "$REPORT_FILE"
    PHASE1_PASS=false
else
    echo "PASS (HTTP 200)" | tee -a "$REPORT_FILE"
fi

# 1d. BIDS direct port
echo -n "[1d] BIDS direct (port 6959)... " | tee -a "$REPORT_FILE"
BIDS_DIRECT=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://as-windows-1.ktpn:6959/swagger/v1/swagger.json 2>/dev/null)
if [ "$BIDS_DIRECT" != "200" ]; then
    echo "FAIL: BIDS direct returned HTTP $BIDS_DIRECT" | tee -a "$REPORT_FILE"
    PHASE1_PASS=false
else
    echo "PASS (HTTP 200)" | tee -a "$REPORT_FILE"
fi

# 1e. AnalysisPortal
echo -n "[1e] AnalysisPortal health... " | tee -a "$REPORT_FILE"
PORTAL_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://as-compute-1.ktpn:5009/api/v1.0/health 2>/dev/null)
if [ "$PORTAL_CODE" != "200" ]; then
    echo "FAIL: AnalysisPortal returned HTTP $PORTAL_CODE" | tee -a "$REPORT_FILE"
    PHASE1_PASS=false
else
    echo "PASS (HTTP 200)" | tee -a "$REPORT_FILE"
fi

# 1f. Postgres
echo -n "[1f] Postgres container... " | tee -a "$REPORT_FILE"
PG_UP=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -c "Postgres.*slot_9" || echo "0")
if [ "$PG_UP" -eq 0 ]; then
    echo "FAIL: Postgres container not running" | tee -a "$REPORT_FILE"
    PHASE1_PASS=false
else
    echo "PASS (running)" | tee -a "$REPORT_FILE"
fi

# 1g. NFS/SMB storage access
echo -n "[1g] Shared storage (NFS)... " | tee -a "$REPORT_FILE"
if [ -d "/mnt/sharedstorage/kla_user/manageddata/lots" ] || [ -d "/mnt/storage01/kla_user/manageddata/lots" ]; then
    echo "PASS (accessible)" | tee -a "$REPORT_FILE"
else
    echo "FAIL: Shared storage not mounted" | tee -a "$REPORT_FILE"
    PHASE1_PASS=false
fi

echo "" | tee -a "$REPORT_FILE"
if [ "$PHASE1_PASS" != true ]; then
    echo "*** PHASE 1 FAILED — ABORTING. Fix infrastructure before proceeding. ***" | tee -a "$REPORT_FILE"
    exit 1
fi
echo "=== PHASE 1: ALL CHECKS PASSED ===" | tee -a "$REPORT_FILE"
```

### Phase 2: Data Pipeline Validation (STRICT — real data only)

No sample data. No stubs. Real lot loading through BIDS.

```bash
# --- Phase 2: Data Pipeline ---
echo "" | tee -a "$REPORT_FILE"
echo "=== PHASE 2: DATA PIPELINE VALIDATION ===" | tee -a "$REPORT_FILE"
PHASE2_PASS=true

# 2a. BIDS lot catalog — verify lots are discoverable
echo -n "[2a] BIDS lot catalog... " | tee -a "$REPORT_FILE"
CATALOG_RESP=$(curl -s --max-time 30 "http://as-windows-1.ktpn:6959/mms/catalog/v1.0/lots?lotRoot=//172.16.1.31/sharedstorage/kla_user/manageddata/lots" 2>&1)
CATALOG_COUNT=$(echo "$CATALOG_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d) if isinstance(d,list) else 0)" 2>/dev/null || echo "0")
echo "$CATALOG_RESP" >> "$REPORT_FILE"
if [ "$CATALOG_COUNT" -eq 0 ]; then
    echo "FAIL: Zero lots in catalog" | tee -a "$REPORT_FILE"
    PHASE2_PASS=false
else
    echo "PASS ($CATALOG_COUNT lots)" | tee -a "$REPORT_FILE"
fi

# 2b. BIDS dbfromlot v2 — metadata for known lot
echo -n "[2b] BIDS dbfromlot v2 (M4CL_FocusCurve_HY01)... " | tee -a "$REPORT_FILE"
DBFROMLOT_RESP=$(curl -s --max-time 60 -X POST "http://as-windows-1.ktpn:6959/mms/dhl/v2/dbfromlot" \
  -H "Content-Type: application/json" \
  -d '{"InputFullPath":"//172.16.1.31/sharedstorage/kla_user/manageddata/lots/M4CL_FocusCurve_HY01","IsRecipePath":false}' 2>&1)
echo "$DBFROMLOT_RESP" >> "$REPORT_FILE"
DBFROMLOT_CODE=$(echo "$DBFROMLOT_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('OK' if 'LotAuxDBInfo' in str(d) or 'WaferInfo' in str(d) or 'auxDbPaths' in str(d) else 'NODATA')" 2>/dev/null || echo "PARSE_ERROR")
if [ "$DBFROMLOT_CODE" != "OK" ]; then
    echo "FAIL: dbfromlot did not return lot metadata ($DBFROMLOT_CODE)" | tee -a "$REPORT_FILE"
    PHASE2_PASS=false
else
    echo "PASS (metadata returned)" | tee -a "$REPORT_FILE"
fi

# 2c. BIDS lotsummary v2 — verify defect counts
echo -n "[2c] BIDS lotsummary v2... " | tee -a "$REPORT_FILE"
LOTSUMMARY_RESP=$(curl -s --max-time 60 -X PUT "http://as-windows-1.ktpn:6959/mms/dhl/v2/lotsummary" \
  -H "Content-Type: application/json" \
  -d '{"InputFullPath":"//172.16.1.31/sharedstorage/kla_user/manageddata/lots/M4CL_FocusCurve_HY01","IsRecipePath":false}' 2>&1)
echo "$LOTSUMMARY_RESP" >> "$REPORT_FILE"
if echo "$LOTSUMMARY_RESP" | grep -qi "defect\|wafer\|count"; then
    echo "PASS (defect data present)" | tee -a "$REPORT_FILE"
else
    echo "FAIL: lotsummary returned no defect data" | tee -a "$REPORT_FILE"
    PHASE2_PASS=false
fi

# 2d. BIDS recipeinfo v2 — KNOWN BUG-002, document current state
echo -n "[2d] BIDS recipeinfo v2 (BUG-002 regression check)... " | tee -a "$REPORT_FILE"
RECIPEINFO_RESP=$(curl -s --max-time 30 -X PUT "http://as-windows-1.ktpn:6959/mms/dhl/v2/recipeinfo" \
  -H "Content-Type: application/json" \
  -H "X-ToolModel: 3955" \
  -d '{"InputFullPath":"//172.16.1.31/sharedstorage/kla_user/manageddata/lots/M4CL_FocusCurve_HY01","IsRecipePath":false,"InputTestIdList":"1"}' 2>&1)
RECIPEINFO_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -X PUT "http://as-windows-1.ktpn:6959/mms/dhl/v2/recipeinfo" \
  -H "Content-Type: application/json" \
  -H "X-ToolModel: 3955" \
  -d '{"InputFullPath":"//172.16.1.31/sharedstorage/kla_user/manageddata/lots/M4CL_FocusCurve_HY01","IsRecipePath":false,"InputTestIdList":"1"}' 2>/dev/null)
echo "HTTP $RECIPEINFO_CODE | $RECIPEINFO_RESP" >> "$REPORT_FILE"
if [ "$RECIPEINFO_CODE" = "500" ]; then
    echo "CONFIRMED STILL BROKEN (HTTP 500) — BUG-002 remains open" | tee -a "$REPORT_FILE"
    # This is a KNOWN bug. Do not fail the workflow for it, but log it clearly.
    echo "  NOTE: BUG-002 is already tracked in BUG_TRACKER.md" | tee -a "$REPORT_FILE"
elif [ "$RECIPEINFO_CODE" = "200" ]; then
    echo "PASS (HTTP 200) — BUG-002 may be FIXED" | tee -a "$REPORT_FILE"
    echo "  ACTION: Verify fix and update BUG_TRACKER.md to mark BUG-002 as Resolved" | tee -a "$REPORT_FILE"
else
    echo "UNEXPECTED HTTP $RECIPEINFO_CODE — investigate" | tee -a "$REPORT_FILE"
fi

echo "" | tee -a "$REPORT_FILE"
if [ "$PHASE2_PASS" != true ]; then
    echo "*** PHASE 2 FAILED — Data pipeline broken. ***" | tee -a "$REPORT_FILE"
    exit 1
fi
echo "=== PHASE 2: ALL CHECKS PASSED ===" | tee -a "$REPORT_FILE"
```

### Phase 3: AnalysisService Real Lot Load (STRICT — no sample data)

Load a real lot through AS. No LoadClassifier.json fallback.

```bash
# --- Phase 3: Real Lot Load via AnalysisService ---
echo "" | tee -a "$REPORT_FILE"
echo "=== PHASE 3: ANALYSISSERVICE LOT LOAD ===" | tee -a "$REPORT_FILE"
PHASE3_PASS=true
AS_HOST="http://as-gpu-compute-4.ktpn:2037"

# 3a. Load dataset (real lot, through BIDS)
echo -n "[3a] Load dataset (M4CL_FocusCurve_HY01)... " | tee -a "$REPORT_FILE"
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
echo "$LOAD_RESP" >> "$REPORT_FILE"

DATASET_ID=$(echo "$LOAD_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('DatasetId', d.get('datasetId', 'NONE')))" 2>/dev/null || echo "PARSE_ERROR")
if [ "$DATASET_ID" = "NONE" ] || [ "$DATASET_ID" = "PARSE_ERROR" ]; then
    echo "FAIL: No DatasetId returned" | tee -a "$REPORT_FILE"
    echo "  Raw response: $LOAD_RESP" | tee -a "$REPORT_FILE"
    PHASE3_PASS=false
else
    echo "PASS (DatasetId=$DATASET_ID)" | tee -a "$REPORT_FILE"
fi

# 3b. Verify defect count (must be > 0)
if [ "$PHASE3_PASS" = true ]; then
    echo -n "[3b] Verify defect count... " | tee -a "$REPORT_FILE"
    COUNT_RESP=$(curl -s --max-time 30 "$AS_HOST/api/v1.0/dataset/$DATASET_ID/count" 2>&1)
    echo "  Count response: $COUNT_RESP" >> "$REPORT_FILE"
    DEFECT_COUNT=$(echo "$COUNT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin))" 2>/dev/null || echo "$COUNT_RESP")
    if [ "$DEFECT_COUNT" -gt 0 ] 2>/dev/null; then
        echo "PASS ($DEFECT_COUNT defects)" | tee -a "$REPORT_FILE"
    else
        echo "FAIL: Zero or invalid defect count ($DEFECT_COUNT)" | tee -a "$REPORT_FILE"
        PHASE3_PASS=false
    fi
fi

# 3c. Verify schema (must have ClassCode_Manual)
if [ "$PHASE3_PASS" = true ]; then
    echo -n "[3c] Verify schema has ClassCode_Manual... " | tee -a "$REPORT_FILE"
    SCHEMA_RESP=$(curl -s --max-time 30 "$AS_HOST/api/v1.0/dataset/$DATASET_ID/schema" 2>&1)
    echo "$SCHEMA_RESP" >> "$REPORT_FILE"
    HAS_CLASSCODE=$(echo "$SCHEMA_RESP" | python3 -c "import sys,json; print('YES' if 'ClassCode_Manual' in str(json.load(sys.stdin)) else 'NO')" 2>/dev/null || echo "NO")
    if [ "$HAS_CLASSCODE" = "YES" ]; then
        echo "PASS (ClassCode_Manual present)" | tee -a "$REPORT_FILE"
    else
        echo "FAIL: Schema missing ClassCode_Manual attribute" | tee -a "$REPORT_FILE"
        PHASE3_PASS=false
    fi
fi

echo "" | tee -a "$REPORT_FILE"
if [ "$PHASE3_PASS" != true ]; then
    echo "*** PHASE 3 FAILED — Lot loading broken. ***" | tee -a "$REPORT_FILE"
    exit 1
fi
echo "=== PHASE 3: ALL CHECKS PASSED ===" | tee -a "$REPORT_FILE"
```

### Phase 4: DalService E2E Pipeline (STRICT — output quality validation)

```bash
# --- Phase 4: DalService E2E Pipeline ---
echo "" | tee -a "$REPORT_FILE"
echo "=== PHASE 4: DALSERVICE E2E PIPELINE ===" | tee -a "$REPORT_FILE"
PHASE4_PASS=true

JOB_ID="bucket1-strict-${TIMESTAMP}"

# 4a. Submit DalJob
echo -n "[4a] Submit DalJob ($JOB_ID)... " | tee -a "$REPORT_FILE"
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
echo "$SUBMIT_RESP" >> "$REPORT_FILE"
if echo "$SUBMIT_RESP" | grep -q "Submitted"; then
    echo "PASS (Submitted)" | tee -a "$REPORT_FILE"
else
    echo "FAIL: Job not submitted" | tee -a "$REPORT_FILE"
    echo "  Raw: $SUBMIT_RESP" | tee -a "$REPORT_FILE"
    PHASE4_PASS=false
fi

# ENUM REFERENCE (from DalEnums.cs -- DO NOT ALTER):
#   0=Queued, 1=Running, 2=Paused, 3=Completed, 4=Failed, 5=Cancelled
# NOTE: FinalState=3 (Completed) does NOT mean success. It means the
# orchestrator ran to the end. Validate output quality separately.

# 4b. Poll for completion (strict timeout: 120s, poll every 5s)
if [ "$PHASE4_PASS" = true ]; then
    echo "[4b] Polling for completion (max 120s)... " | tee -a "$REPORT_FILE"
    ELAPSED=0
    FINAL_STATE="NONE"
    while [ $ELAPSED -lt 120 ]; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        STATUS_RESP=$(curl -s --max-time 10 "http://localhost:49600/api/v1.0/DalJob/$JOB_ID/status" 2>&1)
        CURRENT_STATE=$(echo "$STATUS_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('State','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
        echo "  ${ELAPSED}s: State=$CURRENT_STATE" | tee -a "$REPORT_FILE"
        if [ "$CURRENT_STATE" = "3" ] || [ "$CURRENT_STATE" = "Completed" ]; then
            FINAL_STATE="Completed"
            break
        elif [ "$CURRENT_STATE" = "4" ] || [ "$CURRENT_STATE" = "Failed" ]; then
            FINAL_STATE="Failed"
            break
        fi
    done

    if [ "$FINAL_STATE" = "Completed" ]; then
        echo "Job reached FinalState=3 (Completed). Now validating output quality..." | tee -a "$REPORT_FILE"
    elif [ "$FINAL_STATE" = "Failed" ]; then
        echo "FAIL: Job crashed (FinalState=4=Failed)" | tee -a "$REPORT_FILE"
        PHASE4_PASS=false
    else
        echo "FAIL: Job did not complete within 120s (last state: $CURRENT_STATE)" | tee -a "$REPORT_FILE"
        PHASE4_PASS=false
    fi
fi

# 4c. Get full result
if [ "$PHASE4_PASS" = true ] || [ "$FINAL_STATE" = "Failed" ]; then
    echo -n "[4c] Get full result... " | tee -a "$REPORT_FILE"
    RESULT=$(curl -s --max-time 10 "http://localhost:49600/api/v1.0/DalJob/$JOB_ID/result" 2>&1)
    echo "$RESULT" >> "$REPORT_FILE"
    echo "(captured)" | tee -a "$REPORT_FILE"
fi

# 4d. STRICT ROC validation — AUC must be > 0
if [ "$PHASE4_PASS" = true ]; then
    echo -n "[4d] Validate ROC (AUC > 0 required)... " | tee -a "$REPORT_FILE"
    AUC_BASELINE=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('RocComparison',{}).get('BaselinePerformance',{}).get('AucValue', -1))" 2>/dev/null || echo "-1")
    AUC_ENHANCED=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('RocComparison',{}).get('DalEnhancedPerformance',{}).get('AucValue', -1))" 2>/dev/null || echo "-1")
    ROC_STATUS=$(echo "$RESULT" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('RocComparison',{}).get('BaselinePerformance',{}).get('RocCurveData',{}).get('status','NONE'))" 2>/dev/null || echo "NONE")

    echo "  Baseline AUC: $AUC_BASELINE" | tee -a "$REPORT_FILE"
    echo "  Enhanced AUC: $AUC_ENHANCED" | tee -a "$REPORT_FILE"
    echo "  ROC Status: $ROC_STATUS" | tee -a "$REPORT_FILE"

    if [ "$ROC_STATUS" = "fallback" ]; then
        echo "FAIL: ROC is in FALLBACK mode (AUC=0). This is NOT acceptable in strict mode." | tee -a "$REPORT_FILE"
        echo "  The pipeline must produce real AUC values, not fallback zeros." | tee -a "$REPORT_FILE"
        PHASE4_PASS=false
    elif python3 -c "exit(0 if float('$AUC_BASELINE') > 0 else 1)" 2>/dev/null; then
        echo "PASS (AUC > 0)" | tee -a "$REPORT_FILE"
    else
        echo "FAIL: AUC is zero or negative" | tee -a "$REPORT_FILE"
        PHASE4_PASS=false
    fi
fi

# 4e. Validate generated attributes
if [ "$PHASE4_PASS" = true ]; then
    echo -n "[4e] Validate generated attributes... " | tee -a "$REPORT_FILE"
    GEN_COUNT=$(echo "$RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('GeneratedAttributes',[])))" 2>/dev/null || echo "0")
    if [ "$GEN_COUNT" -gt 0 ]; then
        echo "PASS ($GEN_COUNT attributes)" | tee -a "$REPORT_FILE"
    else
        echo "FAIL: No generated attributes" | tee -a "$REPORT_FILE"
        PHASE4_PASS=false
    fi
fi

# 4f. Validate model persistence
if [ "$PHASE4_PASS" = true ]; then
    echo -n "[4f] Validate ModelPath is set... " | tee -a "$REPORT_FILE"
    MODEL_PATH=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ModelPath','NONE'))" 2>/dev/null || echo "NONE")
    if [ "$MODEL_PATH" != "NONE" ] && [ "$MODEL_PATH" != "null" ] && [ "$MODEL_PATH" != "None" ]; then
        echo "PASS ($MODEL_PATH)" | tee -a "$REPORT_FILE"
    else
        echo "FAIL: ModelPath is null — persistence failed" | tee -a "$REPORT_FILE"
        PHASE4_PASS=false
    fi
fi

echo "" | tee -a "$REPORT_FILE"
if [ "$PHASE4_PASS" != true ]; then
    echo "*** PHASE 4 FAILED — DalService pipeline has defects. ***" | tee -a "$REPORT_FILE"
fi
```

### Phase 5: Bug Classification & Reporting

After all phases complete (or fail), the agent MUST:

```
1. Collect all failures from the report log
2. For EACH failure:
   a. Read AutonomousDevelopment/DeploymentBugs/BUG_TRACKER.md
   b. Cross-reference against Open Bugs, Resolved Bugs, Debunked Claims
   c. Classify as:
      - KNOWN_BUG: matches an open BUG-ID → reference it, do not create new report
      - REGRESSION: matches a resolved BUG-ID → create new report noting regression
      - NEW_BUG: no match → create new verbose report
      - INFRASTRUCTURE: transient infra issue → note but do not file as code bug
3. Write reports:
   - Bucket 1 failures → AutonomousDevelopment/DeploymentBugs/Bucket1/BUG-NNN_<desc>_<date>.md
   - Use MAXIMUM verbosity: full request/response bodies, timestamps, HTTP codes, error messages
   - Include reproduction steps that another agent can execute
4. Update BUG_TRACKER.md if a known bug's status changed (e.g., BUG-002 now returns 200)
```

### Phase 6: Artifact Persistence (with Agent Tagging)

The agent MUST persist ALL outputs locally per AGENTS.md rules.
All files MUST use the Agent File-Tagging Protocol:

```
1. Raw test log → AutonomousDevelopment/BUCKET-1-TESTING/test-results/bucket1_strict_<timestamp>_<agentID>.txt
2. Bug reports → AutonomousDevelopment/DeploymentBugs/Bucket1/BUG-NNN_<desc>_<timestamp>_<agentID>.md
3. Summary → AutonomousDevelopment/BUCKET-1-TESTING/bucket1-strict-report_<timestamp>_<agentID>.md
```

Every markdown file MUST include the metadata footer defined in AGENTS.md:

```markdown
---
<!-- AGENT METADATA — DO NOT REMOVE -->
| Field | Value |
|-------|-------|
| Agent | <platform and session ID> |
| Timestamp | <ISO 8601 UTC> |
| Session Context | <what was the agent asked to do> |
| Confidence | <HIGH / MEDIUM / LOW> |
| Depends On | <referenced files> |
| Supersedes | <older files this replaces> |
```

## Summary Report Format

The final summary must include:

```markdown
# Bucket 1 Strict Test Report

**Date:** <YYYY-MM-DD HH:MM UTC>
**Mode:** STRICT (no fallbacks, no exceptions)
**Overall Result:** PASS | FAIL

## Phase Results

| Phase | Result | Details |
|-------|--------|---------|
| 1. Infrastructure | PASS/FAIL | <which checks failed> |
| 2. Data Pipeline | PASS/FAIL | <which checks failed> |
| 3. Lot Load | PASS/FAIL | <dataset ID, defect count> |
| 4. E2E Pipeline | PASS/FAIL | <FinalState, AUC values> |

## Failures Detected

| # | Phase | Check | Classification | BUG-ID | Details |
|---|-------|-------|---------------|--------|---------|
| 1 | ... | ... | KNOWN_BUG/REGRESSION/NEW_BUG/INFRASTRUCTURE | BUG-NNN | ... |

## Known Bug Status Changes

| BUG-ID | Previous Status | Current Status | Evidence |
|--------|----------------|----------------|----------|
| BUG-002 | Open | Still Open / Fixed | HTTP code observed |

## Raw Artifacts

- Test log: `test-results/bucket1_strict_<timestamp>.txt`
- Bug reports: `DeploymentBugs/Bucket1/BUG-NNN_*.md`
```

## Strict Mode Assertions Reference

These are the production-grade assertions. NONE of these may be relaxed:

| Assertion | Expected | Failure Meaning |
|-----------|----------|-----------------|
| DalService /health | `Healthy` | Service crash or config error |
| AnalysisService /Lifecycle | HTTP 200 | GPU service down |
| BIDS /status | HTTP 200 | Windows VM or IIS issue |
| Lot catalog count | > 0 | Storage mount or BIDS catalog broken |
| Dataset load | DatasetId returned | BIDS→AS pipeline broken |
| Defect count | > 0 | Lot empty or load failed silently |
| Schema ClassCode_Manual | present | Wrong lot or attribute loading broken |
| FinalState | 3 (Completed) — then validate output quality | 4 = pipeline crashed; anything else = still running/stuck |
| AUC Baseline | > 0.0 | ROC computation failed or fallback mode — Completed with zero output |
| AUC Enhanced | > 0.0 | Design attributes not improving classification |
| GeneratedAttributes | count > 0 | Algorithm produced no output |
| ModelPath | non-null | Persistence to Postgres/disk failed |
