# verl standalone code coverage

Measures verl's test coverage on AMD GPUs from a local checkout, with no CI
harness, Jenkins job, or manifest involved. The runtime is built from verl's own
`Dockerfile_rocm.ci` (`rocm/primus` + vLLM from source + megatron-core), and the
collection mechanics are ported from the CI lane at
`aisw-ci-builder-tester/verl/generic/code-coverage`.

## Quick start

From anywhere in the checkout:

```bash
rocm-coverage/verl_coverage.sh --all
```

That runs three phases in order. After the first time, skip the slow one:

```bash
rocm-coverage/verl_coverage.sh --tester --run   # re-measure after editing the source
rocm-coverage/verl_coverage.sh --run            # re-measure with different env knobs
```

| Phase | Flag | What it does | Cost |
|---|---|---|---|
| 1 | `--base` | Builds the runtime image from `Dockerfile_rocm.ci` | Hours (vLLM compiles from source) |
| 2 | `--tester` | Overlays the local source + coverage harness | Minutes |
| 3 | `--run` | Runs the suites under coverage, writes the reports | ~5-15 min |

Results land in `rocm-coverage/artifacts/`, and `result.txt` is echoed at the end.

## Why the base Dockerfile is derived rather than used directly

`Dockerfile_rocm.ci` `COPY`s six files from the build-context root that are not
present in the checkout: two Qwen3.5 training launchers, two unit-test runners, a
fixture downloader, and a patch for `tests/plugin/test_platform_abstraction.py`.
A missing `COPY` source is a hard build failure, and an empty placeholder for the
patch would fail too, because `patch` rejects garbage input under the
Dockerfile's `set -e`.

None of the six affect measured coverage, so
`derive_base_dockerfile.py` writes `build/Dockerfile.base` with exactly those
instructions removed and every other layer byte-identical, so layer caching still
applies. The dropped instructions are printed on every run.

If you later obtain the real files, drop them at the root of the verl checkout
and phase 1 stages them into the build context automatically; only the ones still
missing get dropped.

`Dockerfile_rocm.ci` itself is not carried in this repo. When it is absent, phase 1
falls back to `Dockerfile.base.prederived`, which is the derivation output from the
file as of the baseline run, committed so the lane stays runnable. It is a snapshot
and will drift: once you have the real `Dockerfile_rocm.ci`, put it at the checkout
root or pass `--dockerfile PATH` and the derivation runs for real.

The one thing the patch did that matters is handled directly: verl registers
`PlatformROCm` as `"amd"`, so `test_empty_triggers_auto_detection` (which asserts
detection yields `nvidia` or `huawei`) fails on an AMD host. That single node id
is in `VERL_CC_DESELECT_DEFAULT`; the rest of the file still runs and is the only
coverage of `platform_rocm.py`.

## The two numbers

Both come from the same combined coverage data, reported through different configs:

- **ROCm-subsystem** (`.coveragerc.rocm`) is the headline. Scoped to code that
  actually exercises ROCm: the device abstraction and platform plugins, torch and
  attention kernels, the FSDP/sharding runtime, the `single_controller` Ray plane,
  and the vLLM rollout utilities. Live-serving modules (`vllm_async_server.py`,
  `vllm_pd_replica.py`, `vllm_rollout.py`) are out of scope because they need a
  real multi-replica deployment; counting their unreachable bodies would
  misrepresent the result. `platform_npu.py` is out for the same reason.
- **Whole-package** (`.coveragerc`) covers all of `verl` and is the number to use
  for gap analysis, not for reporting ROCm support.

### Measured baseline

On a single MI325X with verl `0.9.0.dev` on the primus/vLLM 0.23 stack:

| | ROCm-subsystem | Whole-package | Tests |
|---|---|---|---|
| This lane, 1 GPU | **68.67%** (3671 stmts) | 15.11% | 404 passed, 0 failed, 18 skipped |
| CI lane, 8 GPUs, verl 0.7.1 | 49.93% | 13.8% | 312 passed |

The single-GPU number is higher than the 8-GPU CI baseline despite fewer tests,
because verl 0.9 added the platform plugin layer (`platform_rocm.py` and friends,
which this scope includes) and the newer stack exercises more of the rollout path.

The GPU count still matters, so check `visible_gpus`, `multi_gpu_tier2` and
`multi_gpu_tier4` in `result.txt` before comparing runs. The clearest cost of a
single GPU is `verl/utils/ulysses.py`, which sits at 0% because its
sequence-parallel test needs `world_size=2`. At 185 statements it is 5% of the
denominator, so a 2-GPU host should gain several points from that file alone.

## How subprocess coverage works

verl runs much of its work in Ray worker processes, which a plain `pytest --cov`
never sees. Three pieces fix that:

1. `.coveragerc` sets `parallel = true`, `concurrency = multiprocessing` and
   `sigterm = true`, so every process writes its own `.coverage.*` file.
2. `sitecustomize.py` sits on `PYTHONPATH` and calls `coverage.process_startup()`,
   so any interpreter launched with `COVERAGE_PROCESS_START` set starts recording
   at startup. It chains to any base-image `sitecustomize` it shadows rather than
   replacing it.
3. `coverage combine` merges the per-process files before reporting.

Data is written to `/tmp/verlcov` inside the container, not the mounted artifacts
directory, because Ray can spawn a lot of workers.

The suite also runs in two pytest phases. The heavy real-engine test
(`test_vllm_smoke_rocm.py`) goes last in its own process, because a host OOM is a
SIGKILL that coverage cannot trap: in a single process it would discard the whole
run's in-process data and collapse the headline to near zero.

## Environment knobs

Any `VERL_CC_*` variable exported on the host is forwarded into the container.

| Variable | Default | Purpose |
|---|---|---|
| `VERL_CC_TESTS` | curated 35-file list | Upstream test paths, relative to the source root |
| `VERL_CC_EXTRA_TESTS` | 7 ROCm files | Extra test paths, relative to `VERL_CC_EXTRA_TESTS_DIR` |
| `VERL_CC_EXTRA_TESTS_DIR` | `/verl-extra-tests` | Where the baked ROCm extras live |
| `VERL_CC_KEXPR` | `not distributed` | pytest `-k` expression |
| `VERL_CC_DESELECT` | 2 node ids | Space-separated node ids to deselect |
| `VERL_CC_ISOLATE` | `test_vllm_smoke_rocm.py` | Basenames to run last in their own process |
| `VERL_CC_TIMEOUT` | 300 / 600 / 900 | Per-test timeout, scaled by GPU tier |
| `VERL_CC_GPU_COUNT` | torch's count | Override the detected GPU count |
| `VERL_CC_ENABLE_MULTI_GPU_TESTS` | unset | Set to `0` to force the multi-GPU tiers off |
| `VERL_CC_HF_OFFLINE` | `auto` | `1` forces offline, `0` forces online |
| `VERL_CC_RCFILE` | `.coveragerc.rocm` | Config for the scoped report |
| `VERL_CC_REFRESH_SOURCE` | `1` | Re-extract the baked tarball at startup |
| `VERL_CC_FAIL_ON_TEST_FAILURE` | `0` | Exit non-zero when tests failed |

Narrow the run to one file:

```bash
VERL_CC_TESTS="tests/test_protocol_on_cpu.py" VERL_CC_EXTRA_TESTS="" ./verl_coverage.sh --run
```

Iterate on the extra ROCm tests without rebuilding:

```bash
./verl_coverage.sh --run --docker-run-arg \
  "-v/home/AMD/diptodeb/devel/aisw-ci-builder-tester/verl/common/extra-tests:/verl-extra-tests:ro"
```

## Artifacts

| File | Contents |
|---|---|
| `result.txt` | PASS/FAIL, both percentages, test tallies, GPU tier flags |
| `coverage_report.txt` | Per-file ROCm-subsystem report |
| `coverage_report_wholepackage.txt` | Per-file whole-package report |
| `pytest_code_coverage.json` | ROCm-subsystem report, JSON |
| `coverage_wholepackage.json` | Whole-package report, JSON |
| `coverage.xml` | ROCm-subsystem report, Cobertura |
| `coverage.combined.dat` | Merged raw coverage database |
| `log_pytest_core.txt`, `log_pytest_isolated.txt` | Per-phase pytest output |
| `pytest_core.xml`, `pytest_isolated.xml` | Per-phase JUnit XML |

## Layout

Everything lives in this one directory, at the root of the verl checkout:

```
rocm-coverage/
  verl_coverage.sh                driver: the only thing you invoke
  derive_base_dockerfile.py       strips the unbuildable COPYs from Dockerfile_rocm.ci
  Dockerfile.base.prederived      pre-derived fallback, used when Dockerfile_rocm.ci is absent
  Dockerfile.tester               phase-2 overlay
  entrypoint.sh                   runs the suites, combines, reports
  .coveragerc                     collection + whole-package report
  .coveragerc.rocm                ROCm-subsystem scoped report
  sitecustomize.py                subprocess coverage auto-start
  build/                          generated: derived Dockerfile + build contexts (ignored)
  artifacts/                      output (ignored)
```

The driver finds the enclosing checkout by walking up for `setup.py` plus
`verl/__init__.py`, so it needs no arguments from anywhere in the tree. It also
still works if you copy the directory out and park it beside a checkout instead.

The extra ROCm tests are not duplicated here; phase 2 stages them from
`aisw-ci-builder-tester/verl/common/extra-tests`, guessed as a clone beside the
checkout, so the two lanes cannot drift. Override with `--extra-tests PATH` or
`VERL_CC_EXTRA_TESTS_SRC`. Without them the upstream subset still runs, but the
ROCm-scoped number comes out lower.
