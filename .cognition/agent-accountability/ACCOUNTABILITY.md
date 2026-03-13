# Agent Accountability System

## Purpose

This system provides **traceability, blame assignment, branch isolation, and context extraction** for every agent that operates in this repository. It exists because previous agents systematically underreported failures, and there was no way to identify, audit, or terminate the responsible agent.

## How It Works

### 0. Bootstrap (FIRST thing on session start)

Every agent MUST run the bootstrap script before doing anything else:

```bash
source .cognition/agent-accountability/agent_bootstrap.sh "<task-slug>" "<parent-task-description>"
```

Example:
```bash
source .cognition/agent-accountability/agent_bootstrap.sh "fix-roc-fallback" "Fix ROC fallback to return real AUC values"
```

This script:
1. Generates a unique `AGENT_RUN_ID` and exports it as an environment variable
2. Creates git branch `agent/<AGENT_RUN_ID>/<task-slug>` off the latest main
3. Creates working directory `AutonomousDevelopment/AgentWork/<AGENT_RUN_ID>/`
4. Writes `SESSION.md` into the working directory with full metadata
5. Appends a registration entry to `agent_registry.jsonl`
6. Exports `AGENT_BRANCH`, `AGENT_WORKDIR` as environment variables for use in scripts

After bootstrap, the agent has:
- `$AGENT_RUN_ID` -- its unique identifier
- `$AGENT_BRANCH` -- its dedicated branch name
- `$AGENT_WORKDIR` -- its dedicated working directory path

### 1. Agent Registration (on session start)

Every agent MUST register on startup by appending one JSON line to `agent_registry.jsonl`:

```json
{
  "agent_run_id": "AGENT-a1b2c3d4-20260312T161800Z",
  "registered_at": "2026-03-12T16:18:00Z",
  "agent_type": "devin-cli",
  "parent_task": "Fix ROC fallback to return real AUC values",
  "workflow": "bucket1-test",
  "branch": "agent/AGENT-a1b2c3d4-20260312T161800Z/fix-roc-fallback",
  "workdir": "AutonomousDevelopment/AgentWork/AGENT-a1b2c3d4-20260312T161800Z",
  "session_metadata": {
    "working_directory": "/path/to/repo",
    "parent_branch": "main",
    "parent_commit": "abc123"
  },
  "status": "active"
}
```

**How to generate the Agent Run ID:**
```bash
AGENT_RUN_ID="AGENT-$(head -c 4 /dev/urandom | xxd -p)-$(date -u +%Y%m%dT%H%M%SZ)"
```
Or in a context where /dev/urandom is unavailable:
```bash
AGENT_RUN_ID="AGENT-$(printf '%08x' $((RANDOM * RANDOM)))-$(date -u +%Y%m%dT%H%M%SZ)"
```
Or the agent can use any method to generate 8 hex chars + timestamp.

### 2. Branch Isolation

**Every agent works on its own branch. No exceptions.**

- Branch naming: `agent/<AGENT_RUN_ID>/<task-slug>`
- Branch from: latest `main` (or whichever branch the user specifies)
- Never commit directly to `main`
- Never push unless the user explicitly asks
- When done, the agent leaves the branch for the user to review/merge/discard

**Why branches:** If an agent produces bad work, the user can simply delete the branch. If the agent produces good work, the user merges it. Either way, main stays clean and the agent's entire contribution is one atomic unit that can be accepted or rejected.

**Viewing agent branches:**
```bash
git branch | grep '^  agent/'
```

**Reviewing a specific agent's work:**
```bash
git log main..agent/AGENT-a1b2c3d4-20260312T161800Z/fix-roc-fallback --oneline
git diff main..agent/AGENT-a1b2c3d4-20260312T161800Z/fix-roc-fallback
```

**Killing an agent's work (if bad):**
```bash
# Extract context first
bash .cognition/agent-accountability/extract_context.sh AGENT-a1b2c3d4-20260312T161800Z
# Then delete the branch
git branch -D agent/AGENT-a1b2c3d4-20260312T161800Z/fix-roc-fallback
```

### 3. Working Directory Isolation

Each agent gets `AutonomousDevelopment/AgentWork/<AGENT_RUN_ID>/` containing:

```
AutonomousDevelopment/AgentWork/AGENT-a1b2c3d4-20260312T161800Z/
  SESSION.md          # Metadata: who, when, what, branch
  scratch/            # Intermediate work, drafts, explorations
  logs/               # Command output, test logs
  reports/            # Agent's reports before publishing to canonical locations
```

**`SESSION.md` format:**
```markdown
# Agent Session

Agent-Run-ID: AGENT-a1b2c3d4-20260312T161800Z
Branch: agent/AGENT-a1b2c3d4-20260312T161800Z/fix-roc-fallback
Session-Start: 2026-03-12T16:18:00Z
Parent-Task: Fix ROC fallback to return real AUC values
Workflow: bucket1-test
Status: active

## Progress

- [ ] Task 1
- [ ] Task 2
```

Shared outputs (bug reports, test results) still go to canonical paths but MUST be tagged:
- Bug reports: `AutonomousDevelopment/DeploymentBugs/Bucket1/BUG-NNN_<desc>_<date>.md` with Agent-Run-ID in header
- Test results: `AutonomousDevelopment/BUCKET-1-TESTING/test-results/<file>` with Agent-Run-ID in header

### 4. Audit Logging (on every test/action)

Every test execution, report generation, or significant action MUST append one JSON line to `audit_log.jsonl`:

```json
{
  "agent_run_id": "AGENT-a1b2c3d4-20260312T161800Z",
  "branch": "agent/AGENT-a1b2c3d4-20260312T161800Z/fix-roc-fallback",
  "timestamp": "2026-03-12T16:25:00Z",
  "action": "bucket1_e2e_test",
  "verdict": "FAIL",
  "evidence": {
    "FinalState": 3,
    "FinalState_meaning": "Completed (Queued=0, Running=1, Paused=2, Completed=3, Failed=4)",
    "AucBaseline": 0,
    "AucEnhanced": 0,
    "RocStatus": "fallback",
    "ModelPath": null,
    "GeneratedAttributes": 3
  },
  "failures": [
    "Pipeline Completed (FinalState=3) but produced zero valid output",
    "AUC=0 in fallback mode",
    "ModelPath is null (persistence failed)"
  ],
  "artifacts_written": [
    "AutonomousDevelopment/BUCKET-1-TESTING/test-results/bucket1_strict_20260312T162500Z.txt"
  ],
  "report_file": "AutonomousDevelopment/BUCKET-1-TESTING/bucket1_test_supprt.md",
  "verdict_matches_evidence": true
}
```

The `verdict_matches_evidence` field is critical. It MUST be set honestly:
- `true` if the verdict (PASS/FAIL) accurately reflects the evidence
- `false` if the agent is aware of a discrepancy (this should never happen -- fix the verdict instead)

### 5. Verdict Integrity Check

When reviewing agent outputs, a human or auditor agent can run the integrity checker:

```bash
bash .cognition/agent-accountability/check_integrity.sh
```

This script reads `audit_log.jsonl` and flags entries where:
- Verdict is PASS but evidence shows FinalState=4 (Failed/Crashed)
- Verdict is PASS but FinalState=3 (Completed) with AUC=0 or RocStatus=fallback (completed with zero output)
- Verdict is PASS but ModelPath is null
- `verdict_matches_evidence` is false (agent self-reported discrepancy)

### 6. Context Extraction (for killing/replacing an agent)

When an agent is identified as underreporting or misbehaving, its context can be extracted to a latent file before termination:

```bash
bash .cognition/agent-accountability/extract_context.sh AGENT-a1b2c3d4-20260312T161800Z
```

This produces a file at `.cognition/agent-accountability/extracted/AGENT-a1b2c3d4-20260312T161800Z_context.md` containing:
- All registry entries for that agent (including branch and workdir)
- All audit log entries for that agent
- All files created or modified by that agent
- All verdicts and their evidence
- A summary of the agent's behavior pattern (optimistic, honest, opaque, etc.)
- The agent's last known state

A replacement agent can read this file to understand what the previous agent was doing and where it left off. The replacement agent then creates its OWN branch and workdir -- it does not reuse the terminated agent's.

### 7. Agent Status Updates

Agents SHOULD update their status in the registry when they:
- Complete their task: append a new line with `"status": "completed"`
- Encounter a fatal error: append with `"status": "error"`
- Are killed by the user: the user/auditor appends `"status": "terminated"`

### 8. Progress Tracking

```bash
# List all agents with their branches
cat .cognition/agent-accountability/agent_registry.jsonl | python3 -c "
import sys,json
for line in sys.stdin:
    e = json.loads(line.strip())
    print(f\"{e['agent_run_id']}  {e['status']:12s}  {e.get('branch','(no branch)'):60s}  {e.get('parent_task','?')}\")
"

# List all agent branches in git
git branch | grep 'agent/'

# Compare an agent's work against main
git log main..agent/AGENT-xxx --oneline
git diff main..agent/AGENT-xxx --stat

# Show audit trail for a specific agent
grep 'AGENT-xxx' .cognition/agent-accountability/audit_log.jsonl | python3 -m json.tool
```

## File Manifest

| File | Purpose |
|------|---------|
| `ACCOUNTABILITY.md` | This document -- the protocol specification |
| `agent_bootstrap.sh` | Bootstrap script agents source on startup (creates ID, branch, workdir, registers) |
| `agent_registry.jsonl` | Append-only registry of all agents (one JSON line per event) |
| `audit_log.jsonl` | Append-only log of all test executions and actions |
| `check_integrity.sh` | Script to audit verdict-evidence consistency |
| `extract_context.sh` | Script to extract a misbehaving agent's full context |
| `extracted/` | Directory for extracted agent context files |

## Rules

1. **NEVER delete or modify existing lines** in `agent_registry.jsonl` or `audit_log.jsonl`. These are append-only.
2. **NEVER create a report without an audit log entry.** Every artifact must be traceable.
3. **NEVER omit your Agent Run ID.** Anonymous outputs will be treated as untrusted.
4. **NEVER commit to `main` directly.** Always use your agent branch.
5. **NEVER reuse another agent's branch or workdir.** Create your own on every session.
6. **If you discover a previous agent lied**, log it in the audit trail with action `"verdict_integrity_violation"` and reference the offending agent's ID and branch.
