#!/bin/bash
# =============================================================================
# Agent Bootstrap Script
# =============================================================================
# MUST be sourced (not executed) at the start of every agent session.
# Creates the agent's unique ID, branch, working directory, and registration.
#
# Usage:
#   source .cognition/agent-accountability/agent_bootstrap.sh "<task-slug>" "<parent-task>"
#
# Example:
#   source .cognition/agent-accountability/agent_bootstrap.sh "fix-roc-fallback" "Fix ROC fallback to return real AUC values"
#
# After sourcing, these environment variables are available:
#   $AGENT_RUN_ID   -- unique agent identifier
#   $AGENT_BRANCH   -- the agent's dedicated git branch
#   $AGENT_WORKDIR  -- the agent's dedicated working directory (relative)
#   $AGENT_TASK     -- the parent task description
# =============================================================================

# --- Parse arguments ---
TASK_SLUG="${1:-unnamed-task}"
PARENT_TASK="${2:-No task description provided}"

# --- Determine repo root ---
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
    echo "[BOOTSTRAP ERROR] Not inside a git repository."
    return 1 2>/dev/null || exit 1
fi

# --- Generate unique Agent Run ID ---
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
# Try /dev/urandom first, fall back to $RANDOM
HEX=$(head -c 4 /dev/urandom 2>/dev/null | xxd -p 2>/dev/null)
if [ -z "$HEX" ] || [ ${#HEX} -lt 8 ]; then
    HEX=$(printf '%08x' $((RANDOM * RANDOM + RANDOM)))
fi
export AGENT_RUN_ID="AGENT-${HEX}-${TIMESTAMP}"

# --- Determine parent branch and commit ---
PARENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
PARENT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null)

# --- Create dedicated branch ---
# Sanitize task slug: lowercase, replace spaces with hyphens, strip non-alnum
TASK_SLUG_CLEAN=$(echo "$TASK_SLUG" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | sed 's/[^a-z0-9\-]//g' | head -c 50)
export AGENT_BRANCH="agent/${AGENT_RUN_ID}/${TASK_SLUG_CLEAN}"

echo "[BOOTSTRAP] Agent Run ID:  $AGENT_RUN_ID"
echo "[BOOTSTRAP] Task:          $PARENT_TASK"
echo "[BOOTSTRAP] Task slug:     $TASK_SLUG_CLEAN"
echo "[BOOTSTRAP] Parent branch: $PARENT_BRANCH ($PARENT_COMMIT)"
echo "[BOOTSTRAP] Agent branch:  $AGENT_BRANCH"

# Create the branch from current HEAD (which should be main)
git checkout -b "$AGENT_BRANCH" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[BOOTSTRAP WARNING] Could not create branch $AGENT_BRANCH (may already exist). Checking out..."
    git checkout "$AGENT_BRANCH" 2>/dev/null || {
        echo "[BOOTSTRAP ERROR] Failed to create or checkout branch."
        return 1 2>/dev/null || exit 1
    }
fi

# --- Create dedicated working directory ---
export AGENT_WORKDIR="AutonomousDevelopment/AgentWork/${AGENT_RUN_ID}"
mkdir -p "${REPO_ROOT}/${AGENT_WORKDIR}/scratch"
mkdir -p "${REPO_ROOT}/${AGENT_WORKDIR}/logs"
mkdir -p "${REPO_ROOT}/${AGENT_WORKDIR}/reports"

echo "[BOOTSTRAP] Working dir:   $AGENT_WORKDIR"

# --- Write SESSION.md ---
SESSION_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "${REPO_ROOT}/${AGENT_WORKDIR}/SESSION.md" << SESSIONEOF
# Agent Session

Agent-Run-ID: ${AGENT_RUN_ID}
Branch: ${AGENT_BRANCH}
Session-Start: ${SESSION_START}
Parent-Branch: ${PARENT_BRANCH}
Parent-Commit: ${PARENT_COMMIT}
Parent-Task: ${PARENT_TASK}
Status: active

## Task

${PARENT_TASK}

## Progress

_(agent updates this as it works)_
SESSIONEOF

export AGENT_TASK="$PARENT_TASK"

# --- Register in agent_registry.jsonl ---
REGISTRY="${REPO_ROOT}/.cognition/agent-accountability/agent_registry.jsonl"
mkdir -p "$(dirname "$REGISTRY")"

# Build JSON registration entry
python3 -c "
import json, sys
entry = {
    'agent_run_id': '${AGENT_RUN_ID}',
    'registered_at': '${SESSION_START}',
    'agent_type': 'devin-cli',
    'parent_task': $(python3 -c "import json; print(json.dumps('${PARENT_TASK}'))" 2>/dev/null || echo "\"${PARENT_TASK}\""),
    'workflow': None,
    'branch': '${AGENT_BRANCH}',
    'workdir': '${AGENT_WORKDIR}',
    'session_metadata': {
        'working_directory': '${REPO_ROOT}',
        'parent_branch': '${PARENT_BRANCH}',
        'parent_commit': '${PARENT_COMMIT}'
    },
    'status': 'active'
}
print(json.dumps(entry))
" >> "$REGISTRY" 2>/dev/null

# Fallback if python3 not available
if [ $? -ne 0 ]; then
    echo "{\"agent_run_id\": \"${AGENT_RUN_ID}\", \"registered_at\": \"${SESSION_START}\", \"agent_type\": \"devin-cli\", \"parent_task\": \"${PARENT_TASK}\", \"branch\": \"${AGENT_BRANCH}\", \"workdir\": \"${AGENT_WORKDIR}\", \"session_metadata\": {\"parent_branch\": \"${PARENT_BRANCH}\", \"parent_commit\": \"${PARENT_COMMIT}\"}, \"status\": \"active\"}" >> "$REGISTRY"
fi

echo ""
echo "[BOOTSTRAP] ============================================"
echo "[BOOTSTRAP]  Agent bootstrapped successfully"
echo "[BOOTSTRAP]  ID:     $AGENT_RUN_ID"
echo "[BOOTSTRAP]  Branch: $AGENT_BRANCH"
echo "[BOOTSTRAP]  Dir:    $AGENT_WORKDIR"
echo "[BOOTSTRAP] ============================================"
echo ""
echo "[BOOTSTRAP] Environment variables exported:"
echo "  \$AGENT_RUN_ID  = $AGENT_RUN_ID"
echo "  \$AGENT_BRANCH  = $AGENT_BRANCH"
echo "  \$AGENT_WORKDIR = $AGENT_WORKDIR"
echo "  \$AGENT_TASK    = $AGENT_TASK"
echo ""
echo "[BOOTSTRAP] You are now on branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[BOOTSTRAP] Working directory created at: $AGENT_WORKDIR/"
echo ""
