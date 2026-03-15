# Cluster Sync Workflow

This workflow is for running experiments on a remote SSH/Slurm environment while
keeping source editing local.

## Goal

Use the local clone as the only place where tracked source files are edited.
Use the remote clone only to pull a branch, run jobs, and inspect artifacts.

This avoids divergence between a local checkout and an SSH checkout of the same
repository.

## Rules

1. Do not edit tracked source files in the SSH clone.
2. Do not work directly on `main`.
3. Use a dedicated branch for each remote experiment run.
4. On these experiment branches, artifacts are tracked so they can be pulled
   back and inspected locally.
5. Do not run the normal test/lint workflow on these experiment branches.

## Branch Model

Use a branch name that clearly identifies the run, for example:

```bash
git switch -c run/rgb-e2e-a100-bs128
```

Keep `main` clean. Merge or cherry-pick only the source changes you want into
the run branch.

## Local Workflow

1. Create a run branch locally from the source branch you want to execute.
2. Make source edits locally.
3. Adjust `.gitignore` on the run branch if needed so the relevant artifact
   paths are tracked.
4. Commit locally.
5. Push the run branch.

Example:

```bash
git switch -c run/rgb-e2e-a100-bs128
git add -A
git commit -m "Prepare RGB E2E run on A100"
git push -u origin run/rgb-e2e-a100-bs128
```

## Remote Workflow

On the SSH/Slurm clone:

```bash
git fetch origin
git switch run/rgb-e2e-a100-bs128
git pull --ff-only
```

Run jobs from that branch only. Do not make source edits in the remote clone.

## Returning Outputs

Because artifacts are tracked on the run branch, the remote workflow is:

```bash
git add -A
git commit -m "Add run outputs"
git push
```

Then locally:

```bash
git switch run/rgb-e2e-a100-bs128
git pull --ff-only
```

This keeps both source and outputs attached to the same branch history.

## Artifact Tracking

The normal repo workflow ignores artifact directories. On a run branch, change
that only as needed for the outputs you want to inspect locally.

Keep this scoped to the experiment branch. Do not carry broad artifact tracking
back to `main`.

## Tests

These experiment branches are for execution and output collection, not for the
normal development quality gate.

Do not run:

```bash
scripts/tests_local.sh
```

Do not treat these branches as PR-ready development branches. Promote source
changes back into a normal feature branch separately if they should become part
of regular development.
