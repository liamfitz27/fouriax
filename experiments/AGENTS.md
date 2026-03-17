# Experiments Guide

Before editing anything in `experiments/`, read:

- `docs/development/CLUSTER_SYNC_WORKFLOW.md`
- `.codex/skills/experiment-run-workflow/SKILL.md`

Use the local clone as the editing source of truth. Treat the SSH clone as
execute-only. For experiment and job-script work, prefer serial parameter
sweeps, branch-based sync, and selective artifact tracking on run branches.
