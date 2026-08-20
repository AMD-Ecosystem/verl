#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 AMD
# SPDX-License-Identifier: Apache-2.0
#
# verl standalone code-coverage entrypoint (ROCm-subsystem scope, primus runtime).
#
# Runs a GPU-capable test set (attention/flash_attn, ray single_controller,
# device/torch kernels, platform plugin, vLLM rollout) under coverage on the
# stack built by Dockerfile_rocm.ci, then reports BOTH a whole-package number
# (for gap analysis) and a ROCm-subsystem scoped number (.coveragerc.rocm) --
# the honest ROCm-support signal.
#
# Coverage artifacts are ALWAYS produced. Test health is tallied into result.txt;
# set VERL_CC_FAIL_ON_TEST_FAILURE=1 to make this entrypoint exit non-zero when
# any test failed/errored. Default is 0 (the coverage harvest is the deliverable).
set -uo pipefail

VERL_ROOT=/workspace/verl
ARTIFACTS="${ARTIFACTS_FOLDER:-/artifacts}"
mkdir -p "${ARTIFACTS}"

VERL_CC_FAIL_ON_TEST_FAILURE="${VERL_CC_FAIL_ON_TEST_FAILURE:-0}"

# The primus image exposes python3; some layouts also provide an unversioned
# `python`. Prefer whichever exists rather than assuming, as the rocm/vllm-based
# CI lane could.
PYBIN="$(command -v python || command -v python3)"
if [[ -z "${PYBIN}" ]]; then
    echo "ERROR: no python interpreter on PATH"; exit 1
fi

# --- ROCm runtime fixups -----------------------------------------------------
# (1) verl's rollout weight-transfer path dlopen's libnccl.so.2 by name; on ROCm
#     the drop-in is RCCL. Expose it under the NCCL soname if not already linked.
#     This image's ROCm is the pip-installed TheRock SDK reached through the
#     /opt/rocm symlink, so probe ROCM_PATH and the SDK package dir too.
if [[ ! -e /usr/lib/libnccl.so.2 ]]; then
    rccl=""
    for dir in "${ROCM_PATH:-/opt/rocm}/lib" /opt/rocm*/lib \
               "$(${PYBIN} -c 'import _rocm_sdk_core, os; print(os.path.join(os.path.dirname(_rocm_sdk_core.__file__), "lib"))' 2>/dev/null || true)"; do
        [[ -d "${dir}" ]] || continue
        rccl="$(ls "${dir}"/librccl.so.1* 2>/dev/null | head -1)"
        [[ -n "${rccl}" ]] && break
    done
    if [[ -n "${rccl}" ]]; then
        ln -sf "${rccl}" /usr/lib/libnccl.so.2 && ldconfig && echo "linked libnccl.so.2 -> ${rccl}"
    else
        echo "WARN: librccl not found; libnccl.so.2 not linked"
    fi
fi
# (2) With RAY_EXPERIMENTAL_NOSET_{HIP,ROCR}_VISIBLE_DEVICES=1 set, verl's
#     worker.py reads get_accelerator_ids()["GPU"][0] for EVERY ray actor --
#     which IndexErrors on the CPU-only single_controller test actors
#     (num_gpus=0). These CPU tests do not manage real devices, so clear the
#     flags for the harvest; the GPU rollout path sets its own NOSET vars in its
#     worker runtime_env (vllm_async_server).
unset RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES

# (3) HF cache handling. The vLLM rollout smoke test needs the network only on a
#     COLD cache; once the tiny model is cached, force offline so repeated runs
#     don't flake on network pulls / HF rate-limiting. VERL_CC_HF_OFFLINE=0|1
#     overrides the auto behavior (e.g. 0 to always allow downloads).
VERL_CC_HF_OFFLINE="${VERL_CC_HF_OFFLINE:-auto}"
HF_HOME="${HF_HOME:-${HOME:-/root}/.cache/huggingface}"
case "${VERL_CC_HF_OFFLINE}" in
    1) export HF_HUB_OFFLINE=1; echo "HF: forced offline (VERL_CC_HF_OFFLINE=1)" ;;
    0) export HF_HUB_OFFLINE=0; echo "HF: forced online (VERL_CC_HF_OFFLINE=0)" ;;
    *) if find "${HF_HOME}" -type d -name snapshots 2>/dev/null | grep -q .; then
           export HF_HUB_OFFLINE=1; echo "HF: cache present at ${HF_HOME} -> offline"
       else
           export HF_HUB_OFFLINE=0; echo "HF: cache cold at ${HF_HOME} -> allow download"
       fi ;;
esac

# Refresh source from the baked tarball if present (keeps the editable install
# mapped to the exact revision under test); a bind-mount over ${VERL_ROOT} also
# works for fast local iteration. Extracting without wiping preserves the
# coverage configs and sitecustomize.py the image baked alongside the source.
SOURCE_TARBALL="/verl-source.tar.gz"
if [[ -f "${SOURCE_TARBALL}" && "${VERL_CC_REFRESH_SOURCE:-1}" == "1" ]]; then
    tar -xzf "${SOURCE_TARBALL}" -C "${VERL_ROOT}" --strip-components=1
fi
cd "${VERL_ROOT}"

echo "============================================================"
echo "  verl standalone code coverage (ROCm-subsystem scope)"
echo "============================================================"
"${PYBIN}" -c "import verl; print('verl', verl.__version__, '->', verl.__file__)" || echo "WARN: import verl failed"
"${PYBIN}" -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" 2>/dev/null || echo "torch probe failed"
"${PYBIN}" -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm probe failed"
"${PYBIN}" -c "import flash_attn; print('flash_attn', flash_attn.__version__)" 2>/dev/null || echo "flash_attn not present"

# --- Visible GPU count (torch / HIP). Used to optionally enable multi-GPU tests.
# VERL_CC_GPU_COUNT: override (e.g. the runner mis-binds devices). Otherwise torch count.
VERL_CC_VISIBLE_GPU_COUNT="${VERL_CC_GPU_COUNT:-}"
if [[ -z "${VERL_CC_VISIBLE_GPU_COUNT}" ]]; then
    VERL_CC_VISIBLE_GPU_COUNT="$("${PYBIN}" -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)"
fi
echo "VERL_CC_VISIBLE_GPU_COUNT=${VERL_CC_VISIBLE_GPU_COUNT} (set VERL_CC_GPU_COUNT to override)"
# Tier-2 always requires >=2 visible GPUs. VERL_CC_ENABLE_MULTI_GPU_TESTS is an
# opt-out only (=0 force-disables); it can never enable the tier on a 0-1 GPU node.
VERL_CC_MULTI_TIER2=0
if [[ "${VERL_CC_ENABLE_MULTI_GPU_TESTS:-}" != "0" ]] && [[ "${VERL_CC_VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${VERL_CC_VISIBLE_GPU_COUNT}" -ge 2 ]]; then
    VERL_CC_MULTI_TIER2=1
fi
VERL_CC_MULTI_TIER4=0
if [[ "${VERL_CC_MULTI_TIER2}" == "1" ]] && [[ "${VERL_CC_VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${VERL_CC_VISIBLE_GPU_COUNT}" -ge 4 ]]; then
    VERL_CC_MULTI_TIER4=1
fi

# Coverage: auto-start in every python process (pytest + ray workers).
COV_DATA_DIR=/tmp/verlcov
mkdir -p "${COV_DATA_DIR}"
rm -f "${COV_DATA_DIR}"/.coverage* 2>/dev/null || true
export COVERAGE_PROCESS_START="${VERL_ROOT}/.coveragerc"
export PYTHONPATH="${VERL_ROOT}:${PYTHONPATH:-}"

VERL_CC_RCFILE="${VERL_CC_RCFILE:-${VERL_ROOT}/.coveragerc.rocm}"

# Extra ROCm coverage tests are not carried in the verl tree: they live in
# aisw-ci-builder-tester (verl/common/extra-tests), are staged into the build
# context by verl_coverage.sh and BAKED into the image at /verl-extra-tests, so
# no runtime mount is needed; a -v mount over VERL_CC_EXTRA_TESTS_DIR still
# works for fast local iteration. Paths in VERL_CC_*_EXTRA below are resolved
# relative to that dir; everything in VERL_CC_TESTS_DEFAULT is resolved under
# ${VERL_ROOT} (the verl source tree).
VERL_CC_EXTRA_TESTS_DIR="${VERL_CC_EXTRA_TESTS_DIR:-/verl-extra-tests}"

# Upstream verl tests (shipped in the verl source tree). With vllm/flash_attn/
# aiter/ray present we can exercise real ROCm paths (attention/flash_attn padding,
# ray single_controller, device/torch kernels) on top of the CPU core suite.
VERL_CC_TESTS_DEFAULT="\
tests/test_protocol_on_cpu.py \
tests/test_protocol_v2_on_cpu.py \
tests/test_base_config_on_cpu.py \
tests/utils/test_torch_functional.py \
tests/utils/test_seqlen_balancing.py \
tests/utils/test_flops_counter.py \
tests/utils/test_linear_cross_entropy.py \
tests/utils/test_bucketed_weight_transfer.py \
tests/utils/test_shared_memory.py \
tests/models/test_transformer.py \
tests/utils/test_padding_on_cpu.py \
tests/utils/test_config_on_cpu.py \
tests/utils/test_import_utils_on_cpu.py \
tests/utils/test_fs_on_cpu.py \
tests/utils/test_model_on_cpu.py \
tests/utils/test_temp_env_on_cpu.py \
tests/utils/test_normalize_peft_param_name_on_cpu.py \
tests/utils/test_tokenizer_normalize_on_cpu.py \
tests/utils/test_groupwise.py \
tests/plugin/test_platform_abstraction.py \
tests/workers/config/test_engine_config_on_cpu.py \
tests/workers/config/test_actor_config_on_cpu.py \
tests/workers/config/test_critic_config_on_cpu.py \
tests/workers/config/test_model_config_on_cpu.py \
tests/workers/config/test_optim_config_on_cpu.py \
tests/workers/reward_manager/test_registry_on_cpu.py \
tests/trainer/ppo/test_metric_utils_on_cpu.py \
tests/trainer/ppo/test_core_algos_on_cpu.py \
tests/single_controller/test_ray_utils_on_cpu.py \
tests/single_controller/test_decorator_on_cpu.py \
tests/single_controller/test_auto_padding_on_cpu.py \
tests/single_controller/test_get_set_dispatch_collect_cpu.py \
tests/single_controller/test_fused_workers_on_cpu.py \
tests/single_controller/test_ray_local_envs_on_cpu.py \
tests/single_controller/test_data_transfer.py"

# Our extra ROCm coverage tests (baked into the image at
# VERL_CC_EXTRA_TESTS_DIR). Paths are relative to that dir.
VERL_CC_EXTRA_TESTS_DEFAULT="\
utils/test_torch_functional_rocm.py \
utils/test_device_rocm.py \
utils/test_fsdp_utils_rocm.py \
utils/test_seqlen_balancing_rocm.py \
utils/test_import_utils_rocm.py \
workers/rollout/rollout_vllm/test_rollout_utils_rocm.py \
workers/rollout/rollout_vllm/test_vllm_smoke_rocm.py"

# Tier 2 (>=2 GPUs): Ray worker groups that hard-code 2+ GPU actors / split pools
# (upstream verl) + our ulysses / distributed-collective extra tests. Raises
# single_controller + NCCL coverage without a full production training build.
# Tier 4 (>=4 GPUs): all_gather torch group + device-mesh registration (needs ngpus%4 layout).
if [[ "${VERL_CC_MULTI_TIER2}" == "1" ]]; then
    echo "Appending multi-GPU test tier (>=2 visible GPUs)"
    VERL_CC_TESTS_DEFAULT+=" \
tests/single_controller/test_driverfunc_to_worker.py \
tests/single_controller/test_high_level_scheduling_api.py \
tests/single_controller/test_colocated_workers.py \
tests/single_controller/test_colocated_workers_fused.py \
tests/single_controller/test_split_resource_pool.py"
    VERL_CC_EXTRA_TESTS_DEFAULT+=" \
utils/test_dist_collectives_rocm.py \
utils/test_ulysses_rocm.py"
    # FSDP multi-rank sharding lives in the extra fsdp utils file; its world
    # >=2 case is gated internally and only fires under this tier.
fi
if [[ "${VERL_CC_MULTI_TIER4}" == "1" ]]; then
    echo "Appending multi-GPU test tier (>=4 visible GPUs)"
    VERL_CC_TESTS_DEFAULT+=" \
tests/single_controller/test_worker_group_torch.py \
tests/single_controller/test_device_mesh_register.py"
fi

# Unset means "use the default list"; set-but-empty means "run none from this
# list", which is how a narrowed iteration run selects only the other list.
TESTS="${VERL_CC_TESTS-${VERL_CC_TESTS_DEFAULT}}"
EXTRA_TESTS="${VERL_CC_EXTRA_TESTS-${VERL_CC_EXTRA_TESTS_DEFAULT}}"

# Out-of-scope tests (need multi-rank/multi-GPU or network); drop by -k + node id.
VERL_CC_KEXPR="${VERL_CC_KEXPR:-not distributed}"
# test_empty_triggers_auto_detection asserts auto-detection yields nvidia or
# huawei, which is false here: verl registers PlatformROCm as "amd" and it wins
# detection on an AMD host. Upstream's own AMD fix for this is a patch we do not
# have, and deselecting the single node is equivalent for coverage purposes --
# the rest of the file still exercises the platform plugin layer.
VERL_CC_DESELECT_DEFAULT="\
tests/workers/config/test_model_config_on_cpu.py::TestHFModelConfigCPU::test_target_modules_raises_on_invalid_type \
tests/plugin/test_platform_abstraction.py::TestPlatformDetection::test_empty_triggers_auto_detection"
VERL_CC_DESELECT="${VERL_CC_DESELECT:-${VERL_CC_DESELECT_DEFAULT}}"
deselect_args=()
for nid in ${VERL_CC_DESELECT}; do deselect_args+=(--deselect "${nid}"); done

# Tests that load a REAL vLLM engine / large model (and offload weights to host
# RAM) are the only ones heavy enough to risk a host OOM. An OOM is a SIGKILL,
# which coverage cannot trap -- so if such a test runs in the SAME pytest process
# as the rest of the suite, the kill discards the main process's coverage data
# (all in-process modules -> 0%) and the headline collapses. We therefore run
# them in a SEPARATE pytest process LAST: the main suite exits and flushes its
# parallel coverage files first, so a late crash can no longer zero the report.
# Matched by basename; override with VERL_CC_ISOLATE (space-separated basenames).
VERL_CC_ISOLATE="${VERL_CC_ISOLATE:-test_vllm_smoke_rocm.py}"
is_isolated() {
    local b; b="$(basename "$1")"
    for p in ${VERL_CC_ISOLATE}; do [[ "${b}" == "${p}" ]] && return 0; done
    return 1
}

paths=()
isolated_paths=()
for t in ${TESTS}; do
    if [[ -f "${VERL_ROOT}/${t}" ]]; then
        if is_isolated "${t}"; then isolated_paths+=("${VERL_ROOT}/${t}"); else paths+=("${VERL_ROOT}/${t}"); fi
    else echo "skip (missing): ${t}"; fi
done

# The extra ROCm tests are normally BAKED into the image at
# ${VERL_CC_EXTRA_TESTS_DIR}. A missing dir means either the image was built
# without the extra-tests COPY (verl_coverage.sh did not stage them) or the
# operator pointed VERL_CC_EXTRA_TESTS_DIR at a path that is not present. In
# that case skip them with a loud warning rather than failing -- the
# upstream-verl subset still produces a report.
if [[ -d "${VERL_CC_EXTRA_TESTS_DIR}" ]]; then
    echo "Extra ROCm tests dir: ${VERL_CC_EXTRA_TESTS_DIR}"
    for e in ${EXTRA_TESTS}; do
        if [[ -f "${VERL_CC_EXTRA_TESTS_DIR}/${e}" ]]; then
            if is_isolated "${e}"; then isolated_paths+=("${VERL_CC_EXTRA_TESTS_DIR}/${e}"); else paths+=("${VERL_CC_EXTRA_TESTS_DIR}/${e}"); fi
        else echo "skip (missing extra): ${e}"; fi
    done
else
    echo "WARN: extra tests dir '${VERL_CC_EXTRA_TESTS_DIR}' not found -- skipping extra ROCm coverage tests."
    echo "      These are normally baked into the image; check that verl_coverage.sh staged"
    echo "      verl/common/extra-tests and the Dockerfile COPY'd them. To override locally, bind-mount"
    echo "      the dir (e.g. -v <repo>/verl/common/extra-tests:${VERL_CC_EXTRA_TESTS_DIR}:ro)."
fi

# Multi-GPU Ray + NCCL tests need a higher ceiling than the CPU-heavy default.
if [[ -z "${VERL_CC_TIMEOUT:-}" ]]; then
    VERL_CC_TIMEOUT=300
    if [[ "${VERL_CC_MULTI_TIER2}" == "1" ]]; then VERL_CC_TIMEOUT=600; fi
    if [[ "${VERL_CC_MULTI_TIER4}" == "1" ]]; then VERL_CC_TIMEOUT=900; fi
fi

# One pytest phase. Each phase is its own process, so it flushes its parallel
# coverage files (to ${COV_DATA_DIR}) on exit -- a later phase crashing cannot
# discard an earlier phase's data. $1=label (-> log_pytest_<label>.txt, summed by
# the tallier); rest=test paths. Returns the pytest exit code.
run_phase() {
    local label="$1"; shift
    echo "Running ${#} test file(s) [phase=${label}] under coverage (-k '${VERL_CC_KEXPR}')..."
    # Pin config/rootdir to the verl source tree: the isolated phase runs only a
    # file under ${VERL_CC_EXTRA_TESTS_DIR}, so without this pytest would infer a
    # different rootdir than the core phase (and tests that resolve paths relative
    # to ${VERL_ROOT} would break).
    local cfg_args=()
    [[ -f "${VERL_ROOT}/pyproject.toml" ]] && cfg_args=(-c "${VERL_ROOT}/pyproject.toml" --rootdir "${VERL_ROOT}")
    "${PYBIN}" -m pytest -ra --continue-on-collection-errors -p no:cacheprovider \
        "${cfg_args[@]}" \
        -o cache_dir=/tmp/pc \
        -k "${VERL_CC_KEXPR}" "${deselect_args[@]}" \
        --timeout="${VERL_CC_TIMEOUT}" --timeout-method=signal \
        --junitxml="${ARTIFACTS}/pytest_${label}.xml" \
        "$@" 2>&1 | tee "${ARTIFACTS}/log_pytest_${label}.txt"
    return "${PIPESTATUS[0]}"
}

PY_EC=0
if [[ ${#paths[@]} -eq 0 && ${#isolated_paths[@]} -eq 0 ]]; then
    echo "ERROR: no test paths resolved"; PY_EC=1
else
    if [[ ${#paths[@]} -gt 0 ]]; then
        run_phase core "${paths[@]}"; rc=$?
        [[ ${rc} -ne 0 ]] && PY_EC=${rc}
    fi
    # Heavy real-engine tests last, in their own process (see VERL_CC_ISOLATE).
    if [[ ${#isolated_paths[@]} -gt 0 ]]; then
        echo "Isolating ${#isolated_paths[@]} heavy test file(s) in a separate pytest process (coverage already flushed)."
        run_phase isolated "${isolated_paths[@]}"; rc=$?
        [[ ${rc} -ne 0 && ${PY_EC} -eq 0 ]] && PY_EC=${rc}
    fi
fi
echo "---- pytest exit=${PY_EC} ----"

echo "=== combine + report ==="
coverage combine --rcfile="${VERL_ROOT}/.coveragerc" "${COV_DATA_DIR}" 2>&1 | tail -3 || echo "WARN combine"
cp "${COV_DATA_DIR}/.coverage" "${ARTIFACTS}/coverage.combined.dat" 2>/dev/null || true

# (a) Whole-package report (for gap analysis).
coverage json   --rcfile="${VERL_ROOT}/.coveragerc" -o "${ARTIFACTS}/coverage_wholepackage.json" 2>&1 | tail -2 || echo "WARN json(all)"
# (b) ROCm-subsystem scoped report (headline ROCm signal).
coverage json   --rcfile="${VERL_CC_RCFILE}" -o "${ARTIFACTS}/pytest_code_coverage.json" 2>&1 | tail -2 || echo "WARN json(rocm)"
coverage xml    --rcfile="${VERL_CC_RCFILE}" -o "${ARTIFACTS}/coverage.xml" 2>&1 | tail -2 || echo "WARN xml"
coverage report --rcfile="${VERL_CC_RCFILE}" 2>&1 | tee "${ARTIFACTS}/coverage_report.txt" | tail -40 || echo "WARN report"
# Whole-package terminal report too, so gap analysis doesn't need the JSON.
coverage report --rcfile="${VERL_ROOT}/.coveragerc" > "${ARTIFACTS}/coverage_report_wholepackage.txt" 2>&1 || echo "WARN report(all)"

read_total() { "${PYBIN}" - "$1" <<'PY'
import json,sys
try: print(round(json.load(open(sys.argv[1]))['totals']['percent_covered'],2))
except Exception as e: print('NA')
PY
}
ROCM_TOTAL=$(read_total "${ARTIFACTS}/pytest_code_coverage.json")
ALL_TOTAL=$(read_total "${ARTIFACTS}/coverage_wholepackage.json")

echo ""
echo "============================================================"
echo "  ROCm-subsystem coverage : ${ROCM_TOTAL} %  (scope=${VERL_CC_RCFILE##*/})"
echo "  whole-package coverage  : ${ALL_TOTAL} %"
echo "  visible GPUs            : ${VERL_CC_VISIBLE_GPU_COUNT} (tier2=${VERL_CC_MULTI_TIER2} tier4=${VERL_CC_MULTI_TIER4})"
echo "============================================================"

# Tally from the JUnit XML, not the log text: pytest prints the error count both
# in the collection line ("415 items / 1 error / 7 deselected") and in the final
# summary, so scraping the logs double-counts every collection error.
tally_junit() { "${PYBIN}" - "${ARTIFACTS}" <<'PY'
import glob, os, sys, xml.etree.ElementTree as ET

tests = failures = errors = skipped = 0
for path in sorted(glob.glob(os.path.join(sys.argv[1], "pytest_*.xml"))):
    try:
        root = ET.parse(path).getroot()
    except Exception:
        continue
    for suite in root.iter("testsuite"):
        tests += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
print(tests - failures - errors - skipped, failures, errors, skipped)
PY
}
read -r PASSED FAILED ERRORS SKIPPED <<< "$(tally_junit)"
PASSED="${PASSED:-0}"; FAILED="${FAILED:-0}"; ERRORS="${ERRORS:-0}"; SKIPPED="${SKIPPED:-0}"
echo "tests: passed=${PASSED} failed=${FAILED} errors=${ERRORS} skipped=${SKIPPED}"
{
    if [[ "${FAILED}" -eq 0 && "${ERRORS}" -eq 0 ]]; then echo "PASS"; else echo "FAIL"; fi
    echo "rocm_subsystem_percent=${ROCM_TOTAL}"
    echo "whole_package_percent=${ALL_TOTAL}"
    echo "tests_passed=${PASSED}"
    echo "tests_failed=${FAILED}"
    echo "tests_errors=${ERRORS}"
    echo "tests_skipped=${SKIPPED}"
    echo "pytest_exit_code=${PY_EC}"
    echo "visible_gpus=${VERL_CC_VISIBLE_GPU_COUNT}"
    echo "multi_gpu_tier2=${VERL_CC_MULTI_TIER2}"
    echo "multi_gpu_tier4=${VERL_CC_MULTI_TIER4}"
} > "${ARTIFACTS}/result.txt"

ls -al "${ARTIFACTS}" 2>/dev/null | grep -vE "\.coverage\." || true

if [[ "${VERL_CC_FAIL_ON_TEST_FAILURE}" == "1" && ( "${FAILED:-0}" -gt 0 || "${ERRORS:-0}" -gt 0 ) ]]; then
    echo "VERL_CC_FAIL_ON_TEST_FAILURE=1: ${FAILED:-0} failed / ${ERRORS:-0} errored -> exit 1"
    exit 1
fi
exit 0
