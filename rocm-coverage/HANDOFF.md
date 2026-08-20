# Handoff: verl standalone code coverage

You have everything needed to measure verl coverage on an AMD GPU host and to add
new tests. Start with `README.md` for how the lane works; this file covers what
changed, how to get running, and how to land new tests upstream.

Current baseline on a single MI325X (verl `0.9.0.dev`, primus v26.5 + vLLM 0.23):
**68.67% ROCm-subsystem, 15.11% whole-package, 404 passed / 0 failed / 18 skipped.**

## What is in here

See `README.md` for the file-by-file layout. The short version: `verl_coverage.sh`
is the only thing you invoke, everything else in `rocm-coverage/` is either a
config it feeds to the container or a script that runs inside it.

## The lane does not modify the checkout it measures

It reads the tree, packages it, and measures it in a container. The only paths it
writes are `rocm-coverage/build/` and `rocm-coverage/artifacts/`, both gitignored
and both excluded from the source tarball, so you can rebase or switch branches
freely.

## The extra ROCm tests live in a different repo

Seven of the test files that produce the ROCm-scoped number are not in this tree.
They are in `aisw-ci-builder-tester` under `verl/common/extra-tests/`, and phase 2
stages them from a clone beside this one. They are deliberately not copied here:
the CI coverage lane reads the same directory, and a second copy would drift.

Clone it beside your verl checkout and the default path resolves:

```
<parent>/verl/                      this repo
<parent>/aisw-ci-builder-tester/    the extras
```

One file there needed a fix that may still be uncommitted on `amd-integration`:

- `verl/common/extra-tests/workers/rollout/rollout_vllm/test_rollout_utils_rocm.py`

It imported `get_device_uuid` from `verl.workers.rollout.vllm_rollout.utils`,
which verl 0.9 moved onto the platform plugin (`PlatformBase.get_device_uuid`).
That one missing name failed collection for the entire file, silently dropping 11
tests and pinning `vllm_rollout/utils.py` at 16%. The fix imports it tolerantly
and skips the two dependent tests when it is absent, so it stays backward
compatible with the CI lane on verl 0.7.1. `utils.py` went to 40% and the ROCm
total from 67% to 69%. Check whether it has landed before you trust a baseline
comparison; if it has not, that file contributes nothing.

## Getting running

```bash
rocm-coverage/verl_coverage.sh --all
```

No arguments needed if the extras are cloned beside this checkout; otherwise pass
`--extra-tests PATH`. Phase 1 builds the runtime and takes hours because vLLM
compiles from source. Run it once. After that:

```bash
rocm-coverage/verl_coverage.sh --tester --run   # ~25s rebuild + ~10 min measure
```

Phase 1 normally derives its Dockerfile from `Dockerfile_rocm.ci` at the checkout
root. That file is not in this repo, so without it the build falls back to the
committed `Dockerfile.base.prederived` snapshot; see the README section on
derivation before relying on that.

The host needs Docker, `/dev/kfd` and `/dev/dri`, and roughly 90 GB of free disk
for the two 43 GB images.

### If phase 1 stalls

vLLM's CMake `FetchContent` does a full clone of `triton-lang/triton`, and that
connection can die with no timeout to abort it. Symptom: hours pass with load
near zero. Find the clone and kill just that one process; CMake retries and
succeeds:

```bash
ps -eo pid,etime,args | grep 'git clone.*triton'
kill -TERM <pid>
```

Do not kill the `docker build` client. If you do, the daemon keeps compiling but
nothing tags the image.

## Adding tests

Decide where the test belongs.

**Upstream verl tests** (things that should live in verl itself) go in this tree
under `tests/` and get added to `VERL_CC_TESTS_DEFAULT` in
`rocm-coverage/entrypoint.sh`. Paths are relative to the checkout root.

**ROCm-only tests** (not appropriate upstream) go in
`aisw-ci-builder-tester/verl/common/extra-tests/`, mirroring the verl source
layout, and get added to `VERL_CC_EXTRA_TESTS_DEFAULT`. Paths there are relative
to `extra-tests/`. Import from the installed package (`from verl.utils... import`)
so the file is location-independent and needs no `conftest.py`.

### Fast iteration loop

Bind-mount the extras over the baked copy and run only your new file. No rebuild:

```bash
VERL_CC_TESTS="" \
VERL_CC_EXTRA_TESTS="utils/test_my_new_thing_rocm.py" \
rocm-coverage/verl_coverage.sh --run \
  --docker-run-arg "-v/path/to/aisw-ci-builder-tester/verl/common/extra-tests:/verl-extra-tests:ro"
```

An empty `VERL_CC_TESTS` means "run nothing from the upstream list"; leaving it
unset means "use the default list". Coverage percentages from a narrowed run are
meaningless; use it to get the test passing, then do a full `--run` for the number.

Editing a test file only needs the bind mount above. Editing `entrypoint.sh` or
either `.coveragerc` needs `--tester` first, since those are baked into the image.

For an interactive shell in the same environment:

```bash
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  --ipc=host --shm-size=8G --entrypoint /bin/bash -w /workspace/verl \
  verl-cov-tester:latest
```

### Making a new test move the headline number

The ROCm-subsystem percentage only counts files in the `include` list of
`.coveragerc.rocm`. A test covering a module that is not listed raises the
whole-package number and leaves the headline flat. If you are targeting new
surface, add the module there, and add a comment saying why it is ROCm-relevant.
Live-serving modules (`vllm_async_server.py`, `vllm_pd_replica.py`,
`vllm_rollout.py`) and `platform_npu.py` are excluded on purpose; see the comments
in that file before adding them.

### Highest-value gaps right now

From the baseline run's `coverage_report.txt`:

| Module | Cover | Note |
|---|---|---|
| `verl/utils/ulysses.py` | 0% | Needs `world_size=2`; 185 stmts, 5% of the denominator. Biggest single win, needs a 2-GPU host |
| `verl/workers/rollout/vllm_rollout/weight_update_utils.py` | 19% | 27 stmts, easy unit-test target |
| `verl/workers/rollout/vllm_rollout/utils.py` | 40% | 166 lines still missing |
| `verl/plugin/platform/platform_base.py` | 49% | New plugin layer, mostly untested |
| `verl/single_controller/ray/base.py` | 58% | 199 lines missing, much of it multi-GPU |

Tests that need more than one GPU should self-gate on the visible GPU count. The
entrypoint enables the `>=2` tier automatically and appends those files only then.

## Pushing tests to aisw-ci-builder-tester

Repo: `git@github-amd-eng:AMD-AIOSS/aisw-ci-builder-tester.git`, base branch
`amd-integration`, merged via pull request.

```bash
cd aisw-ci-builder-tester
git checkout amd-integration && git pull
git checkout -b verl-coverage-<topic>
# add files under verl/common/extra-tests/, mirroring the verl source layout
git add verl/common/extra-tests
git commit -m "verl: add ROCm coverage test for <module>"
git push -u origin verl-coverage-<topic>
gh pr create --base amd-integration --title "verl: ROCm coverage tests for <module>"
```

Do not commit `__pycache__` (there is a stale `.pyc` under
`workers/rollout/rollout_vllm/` in the repo already; leave it or clean it in a
separate commit).

Two consumers read these files, so keep both working:

1. **This standalone lane**, on verl `0.9.0.dev`.
2. **The CI coverage lane**, at
   `aisw-ci-builder-tester/verl/generic/code-coverage/`, which runs on
   `rocm/vllm:latest` with verl 0.7.1 and reaches these same files through its own
   `entrypoint.sh`.

That version spread is the main review concern: a symbol present in one verl and
not the other must be imported tolerantly and the dependent tests skipped, as
`test_rollout_utils_rocm.py` now does. A bare import failure costs the whole file
in both lanes and shows up only as a quiet collection error.

Adding a test file to this lane also means adding it to the CI lane's
`VERL_CC_EXTRA_TESTS_DEFAULT` in
`verl/generic/code-coverage/tester/entrypoint.sh` if you want it to run there too.
The two test lists are independent.
