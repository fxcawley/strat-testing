# Bug Verification Skill

## Purpose

This skill enforces the bug verification protocol. Any agent that believes it has found a bug MUST execute this skill before reporting the finding to the user.

## When to Invoke

Invoke this skill whenever you:
- Observe unexpected behavior (HTTP errors, wrong return values, crashes, timeouts)
- See error messages in logs or API responses
- Notice data mismatches or missing data
- Encounter test failures
- Find code that appears incorrect

## Procedure

### Step 1: Read the Bug Tracker

Read `AutonomousDevelopment/DeploymentBugs/BUG_TRACKER.md` in its entirety. This file contains:
- **Open Bugs** — currently known, validated bugs with severity, component, and reproduction steps
- **Resolved Bugs** — previously fixed issues with root cause and fix description
- **Debunked Claims** — observations that were investigated and found to NOT be bugs (PEBDAK, by-design, speculative)
- **Root Cause Cascade** — shows how multiple symptoms can trace to a single root cause

### Step 2: Cross-Reference Your Observation

Compare your observation against EVERY entry in BUG_TRACKER.md:

| Your observation matches... | Action |
|---|---|
| An **Open Bug** | Reference the existing BUG-ID. Do NOT report as new. |
| A **Resolved Bug** | Investigate regression. Only report if you confirm the fix was reverted or bypassed. |
| A **Debunked Claim** | Do NOT report as a bug. Cite the debunked entry. Explain what led you to re-encounter it. |
| **Nothing in the tracker** | Proceed to Step 3. |

### Step 3: Validate Before Reporting

If your finding is genuinely new (not in BUG_TRACKER.md), you must still validate:

1. **Reproduce it** — Run the failing operation at least twice. One-off errors are not bugs.
2. **Isolate the component** — Is the error from the service you think, or is it propagated from a dependency?
3. **Check infrastructure** — Is this a transient network/mount/container issue? Check service health first.
4. **Read the source code** — If possible, trace the code path to understand whether the behavior is intentional.
5. **Check the Root Cause Cascade** — Many symptoms trace to a single infrastructure failure. Don't report 5 bugs when there's 1 root cause.

### Step 4: Report with Maximum Verbosity

If validated as a new bug, write a report to the appropriate bucket directory:
- `AutonomousDevelopment/DeploymentBugs/Bucket1/` for Bucket 1 test failures
- `AutonomousDevelopment/DeploymentBugs/Bucket2/` for Bucket 2 test failures

Report format:
```markdown
# BUG-NNN: <Title>

**Severity:** Critical | High | Medium | Low
**Component:** <service name and version>
**Endpoint/Code:** <specific endpoint or source file:line>
**Date Discovered:** <YYYY-MM-DD HH:MM UTC>
**Discovered By:** <agent/workflow name>

## Symptom

<What you observed — exact error messages, HTTP codes, unexpected values>

## Raw Evidence

<Full curl commands and responses, log excerpts, timestamps>

## Reproduction Steps

1. <Step-by-step instructions to reproduce>

## Root Cause Analysis

<What you believe is causing this and why>

## Cross-Reference

- Checked BUG_TRACKER.md: <confirmation that this is not a duplicate/debunked>
- Related entries: <any related BUG-IDs>

## Suggested Fix

<If you can identify a fix, describe it here>
```

### Anti-Patterns (DO NOT)

- Do NOT report "BIDS NaN bug" — this has been debunked (see BUG_TRACKER.md Debunked Claims)
- Do NOT report "ChildDHL always calls BIDS" — this is by-design (see BUG_TRACKER.md)
- Do NOT report "NoMmsLotSourceAdapter fails" — this is a known stub, not production path
- Do NOT report "ROC class scheme mismatch" as a bug — it is PEBDAK (see BUG-003)
- Do NOT report AUC=0.0 as a bug when using fallback mode — this is expected behavior
- Do NOT confuse infrastructure issues (SMB down, container not running) with code bugs

### DalJobState Enum Reference (verified from DalEnums.cs source code)

```
Queued=0, Running=1, Paused=2, Completed=3, Failed=4, Cancelled=5
```

**FinalState=3 means COMPLETED.** FinalState=4 means FAILED. A previous agent session
incorrectly claimed 3=Failed and 2=Completed. That error was propagated into multiple
files. Always verify enum values against `DalEnums.cs`, not other agent reports.

### File Tagging Requirement

All bug reports produced by this skill MUST comply with the Agent File-Tagging Protocol
defined in the root `AGENTS.md`. Specifically:

1. **Filename**: `BUG-NNN_<description>_<YYYYMMDDTHHMMSSZ>_<AGENT_ID>.md`
2. **Metadata footer**: Must include Agent, Timestamp, Session Context, Confidence,
   Depends On, and Supersedes fields
3. **Agent ID**: Use your platform name + a session-unique identifier
   (e.g., `devin-session-4a2c`, `claude-review-9f1b`)

Untagged bug reports are untraceable and may be ignored by downstream agents.
