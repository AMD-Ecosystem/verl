#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 AMD
# SPDX-License-Identifier: Apache-2.0
#
# Standalone verl code-coverage runner.
#
# Builds a coverage-instrumented image from verl's own Dockerfile_rocm.ci
# (rocm/primus + vLLM from source + megatron-core) and measures coverage of a
# LOCAL verl checkout on the GPUs of this host. No Jenkins, no manifests, no
# harness: three phases you can run together or one at a time.
#
#   base    build the heavy runtime image from Dockerfile_rocm.ci (slow, cached)
#   tester  overlay the local source + coverage harness onto it (fast)
#   run     execute the suites under coverage and write the reports
#
# The coverage mechanics (parallel + multiprocessing collection, sitecustomize
# auto-start in Ray workers, combine, dual-scope reporting) are ported from the
# CI lane at aisw-ci-builder-tester/verl/generic/code-coverage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${SCRIPT_DIR}/build"

# This lane lives INSIDE the verl tree (rocm-coverage/), but it also has to keep
# working when the directory is copied somewhere else and sits BESIDE a checkout.
# So find the checkout by walking up from here, and only fall back to the sibling
# layout if that finds nothing. setup.py pins it: a bare verl/ dir would match a
# parent that merely contains a clone.
find_checkout_root() {
    local d="$1"
    while [[ "${d}" != "/" ]]; do
        [[ -f "${d}/setup.py" && -f "${d}/verl/__init__.py" ]] && { echo "${d}"; return 0; }
        d="$(dirname "${d}")"
    done
    return 1
}

# ── Defaults ──────────────────────────────────────────────────────────────────
PROJECT_SRC="$(find_checkout_root "${SCRIPT_DIR}" || echo "${SCRIPT_DIR}/verl")"
ARTIFACTS_DIR="${SCRIPT_DIR}/artifacts"
# The extra ROCm tests live in aisw-ci-builder-tester, not in this tree. Guess the
# usual side-by-side clone; override with --extra-tests or VERL_CC_EXTRA_TESTS_SRC.
EXTRA_TESTS_SRC="${VERL_CC_EXTRA_TESTS_SRC:-$(dirname "${PROJECT_SRC}")/aisw-ci-builder-tester/verl/common/extra-tests}"
DOCKERFILE=""
BASE_TAG="verl-cov-base:latest"
TESTER_TAG="verl-cov-tester:latest"
HF_CACHE="${HOME}/.cache/huggingface"
BUILD_ARGS=()
DOCKER_RUN_EXTRA=()
NO_CACHE=""
PRINT_PATHS=0
RUN_BASE=0
RUN_TESTER=0
RUN_RUN=0

usage() {
    cat <<'EOF'
Usage: verl_coverage.sh [PHASES] [OPTIONS]

Phases (choose one or more; --all is the default when none are given):
  --all                 base + tester + run
  --base                build the runtime image from Dockerfile_rocm.ci
  --tester              build the coverage overlay image
  --run                 run the suites under coverage

Options:
  --src PATH            verl checkout to measure   (default: enclosing checkout)
  --artifacts PATH      output directory           (default: <script dir>/artifacts)
  --dockerfile PATH     base Dockerfile            (default: <src>/Dockerfile_rocm.ci,
                                                    else the Dockerfile.base.prederived snapshot)
  --extra-tests PATH    ROCm extra tests to bake   (default: ../aisw-ci-builder-tester/
                                                    verl/common/extra-tests, beside the checkout)
  --base-tag TAG        phase-1 image tag          (default: verl-cov-base:latest)
  --tester-tag TAG      phase-2 image tag          (default: verl-cov-tester:latest)
  --hf-cache PATH       host HF cache to mount     (default: ~/.cache/huggingface)
  --build-arg K=V       extra docker build arg (repeatable, applies to phase 1)
  --docker-run-arg ARG  extra docker run arg   (repeatable, applies to phase 3)
  --no-cache            build both images without the layer cache
  --print-paths         show the resolved paths and exit (nothing is built)
  -h, --help            this message

Any VERL_CC_* variable in the environment is forwarded into the test container.
Useful ones: VERL_CC_TESTS, VERL_CC_EXTRA_TESTS, VERL_CC_TIMEOUT, VERL_CC_KEXPR,
VERL_CC_DESELECT, VERL_CC_ISOLATE, VERL_CC_GPU_COUNT, VERL_CC_HF_OFFLINE,
VERL_CC_FAIL_ON_TEST_FAILURE. See README.md, and HANDOFF.md for adding tests.

Examples:
  rocm-coverage/verl_coverage.sh --all
  rocm-coverage/verl_coverage.sh --tester --run       # after the base exists
  VERL_CC_TESTS="tests/test_protocol_on_cpu.py" rocm-coverage/verl_coverage.sh --run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)              RUN_BASE=1; RUN_TESTER=1; RUN_RUN=1; shift ;;
        --base)             RUN_BASE=1; shift ;;
        --tester)           RUN_TESTER=1; shift ;;
        --run)              RUN_RUN=1; shift ;;
        --src)              PROJECT_SRC="$2"; shift 2 ;;
        --artifacts)        ARTIFACTS_DIR="$2"; shift 2 ;;
        --dockerfile)       DOCKERFILE="$2"; shift 2 ;;
        --extra-tests)      EXTRA_TESTS_SRC="$2"; shift 2 ;;
        --base-tag)         BASE_TAG="$2"; shift 2 ;;
        --tester-tag)       TESTER_TAG="$2"; shift 2 ;;
        --hf-cache)         HF_CACHE="$2"; shift 2 ;;
        --build-arg)        BUILD_ARGS+=(--build-arg "$2"); shift 2 ;;
        --docker-run-arg)   DOCKER_RUN_EXTRA+=("$2"); shift 2 ;;
        --no-cache)         NO_CACHE="--no-cache"; shift ;;
        --print-paths)      PRINT_PATHS=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ${RUN_BASE} -eq 0 && ${RUN_TESTER} -eq 0 && ${RUN_RUN} -eq 0 ]]; then
    RUN_BASE=1; RUN_TESTER=1; RUN_RUN=1
fi

PROJECT_SRC="$(cd "${PROJECT_SRC}" 2>/dev/null && pwd || echo "${PROJECT_SRC}")"
ARTIFACTS_DIR="$(readlink -m "${ARTIFACTS_DIR}")"

# Dockerfile_rocm.ci is not carried in this repo. If the caller has it, derive
# from it as designed; otherwise fall back to the derived Dockerfile committed
# next to this script, which is the output of that derivation from the file as of
# the baseline run. The fallback keeps phase 1 runnable but is a SNAPSHOT: point
# --dockerfile at the real thing whenever you have it.
USE_PREDERIVED=0
if [[ -z "${DOCKERFILE}" ]]; then
    if [[ -f "${PROJECT_SRC}/Dockerfile_rocm.ci" ]]; then
        DOCKERFILE="${PROJECT_SRC}/Dockerfile_rocm.ci"
    elif [[ -f "${SUPPORT_DIR}/Dockerfile.base.prederived" ]]; then
        USE_PREDERIVED=1
    else
        DOCKERFILE="${PROJECT_SRC}/Dockerfile_rocm.ci"
    fi
fi

log() { echo -e "\n=== $* ==="; }

# Printed before the checkout check on purpose: if detection went wrong, this is
# what you need to see.
if [[ ${PRINT_PATHS} -eq 1 ]]; then
    exists() { [[ -e "$1" ]] && echo "ok" || echo "MISSING"; }
    echo "script dir     ${SCRIPT_DIR}"
    echo "checkout       ${PROJECT_SRC}  [$(exists "${PROJECT_SRC}/verl/__init__.py")]"
    echo "artifacts      ${ARTIFACTS_DIR}"
    echo "build dir      ${BUILD_DIR}"
    echo "extra tests    ${EXTRA_TESTS_SRC}  [$(exists "${EXTRA_TESTS_SRC}")]"
    if [[ ${USE_PREDERIVED} -eq 1 ]]; then
        echo "dockerfile     ${SUPPORT_DIR}/Dockerfile.base.prederived  [pre-derived fallback]"
    else
        echo "dockerfile     ${DOCKERFILE}  [$(exists "${DOCKERFILE}")]"
    fi
    echo "hf cache       ${HF_CACHE}"
    exit 0
fi

if [[ ! -d "${PROJECT_SRC}/verl" ]]; then
    echo "ERROR: ${PROJECT_SRC} does not look like a verl checkout (no verl/ package dir)" >&2
    exit 1
fi

# ── Phase 1: base runtime image ───────────────────────────────────────────────
build_base() {
    log "Phase 1/3: base runtime image (${BASE_TAG})"
    local context="${BUILD_DIR}/base-context"

    if [[ ${USE_PREDERIVED} -eq 1 ]]; then
        echo "WARN: no Dockerfile_rocm.ci at ${PROJECT_SRC}; using the committed"
        echo "      Dockerfile.base.prederived snapshot instead. Pass --dockerfile"
        echo "      <path> to derive from the real file."
        rm -rf "${context}"; mkdir -p "${context}"
        mkdir -p "${BUILD_DIR}"
        cp "${SUPPORT_DIR}/Dockerfile.base.prederived" "${BUILD_DIR}/Dockerfile.base"
        log "Building ${BASE_TAG} (vLLM compiles from source; expect a long first run)"
        docker build ${NO_CACHE} \
            -f "${BUILD_DIR}/Dockerfile.base" \
            -t "${BASE_TAG}" \
            "${BUILD_ARGS[@]}" \
            "${context}"
        return
    fi

    [[ -f "${DOCKERFILE}" ]] || { echo "ERROR: dockerfile not found: ${DOCKERFILE}" >&2; exit 1; }

    # A dedicated, tiny build context. Dockerfile_rocm.ci clones verl itself, so
    # the context only has to carry the files it COPYs. Any of those present at
    # the checkout root are staged here and used; the rest are dropped from the
    # derived Dockerfile (they are training launchers / test runners / a test
    # patch, none of which affect measured coverage).
    rm -rf "${context}"; mkdir -p "${context}"

    local staged=0
    while read -r src; do
        [[ -z "${src}" ]] && continue
        if [[ -f "${PROJECT_SRC}/${src}" ]]; then
            mkdir -p "$(dirname "${context}/${src}")"
            cp "${PROJECT_SRC}/${src}" "${context}/${src}"
            echo "staged COPY source from checkout root: ${src}"
            staged=$((staged + 1))
        fi
    done < <(awk '/^[[:space:]]*COPY[[:space:]]/ {print $2}' "${DOCKERFILE}")
    echo "staged ${staged} COPY source(s) into the base build context"

    python3 "${SUPPORT_DIR}/derive_base_dockerfile.py" \
        --dockerfile "${DOCKERFILE}" \
        --context "${context}" \
        --output "${BUILD_DIR}/Dockerfile.base"

    log "Building ${BASE_TAG} (vLLM compiles from source; expect a long first run)"
    docker build ${NO_CACHE} \
        -f "${BUILD_DIR}/Dockerfile.base" \
        -t "${BASE_TAG}" \
        "${BUILD_ARGS[@]}" \
        "${context}"
}

# ── Phase 2: coverage overlay image ───────────────────────────────────────────
build_tester() {
    log "Phase 2/3: coverage overlay image (${TESTER_TAG})"
    if ! docker image inspect "${BASE_TAG}" >/dev/null 2>&1; then
        echo "ERROR: base image '${BASE_TAG}' not found. Run with --base first." >&2
        exit 1
    fi

    local context="${BUILD_DIR}/tester-context"
    rm -rf "${context}"; mkdir -p "${context}"

    # Source tarball. Name-based excludes so generated/hidden dirs are dropped at
    # ANY depth -- a path-anchored 'verl/__pycache__' would only match the top
    # level and still bundle nested caches, bloating the context.
    # This lane lives inside the tree it packages, so its own build contexts and
    # artifacts would otherwise be tarred INTO the image -- including the previous
    # copy of this very tarball. Exclude them, anchored, when they are inside.
    local src_name; src_name="$(basename "${PROJECT_SRC}")"
    local self_excludes=()
    local d
    for d in "${BUILD_DIR}" "${ARTIFACTS_DIR}"; do
        case "${d}/" in
            "${PROJECT_SRC}"/*) self_excludes+=(--exclude="${src_name}/${d#"${PROJECT_SRC}"/}") ;;
        esac
    done

    log "Packaging source from ${PROJECT_SRC}"
    tar -czf "${context}/verl-source.tar.gz" \
        -C "$(dirname "${PROJECT_SRC}")" \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='.pytest_cache' \
        --exclude='outputs' \
        "${self_excludes[@]}" \
        "${src_name}"
    echo "tarball: $(du -h "${context}/verl-source.tar.gz" | cut -f1)"

    for f in Dockerfile.tester entrypoint.sh .coveragerc .coveragerc.rocm sitecustomize.py; do
        [[ -f "${SUPPORT_DIR}/${f}" ]] || { echo "ERROR: missing ${SUPPORT_DIR}/${f}" >&2; exit 1; }
        cp "${SUPPORT_DIR}/${f}" "${context}/${f}"
    done

    # The extra ROCm tests are not carried in the verl tree; bake them in so the
    # run needs no bind-mount. Staged (not duplicated) so the two lanes cannot drift.
    if [[ -d "${EXTRA_TESTS_SRC}" ]]; then
        cp -RL "${EXTRA_TESTS_SRC}" "${context}/extra-tests"
        echo "staged extra ROCm tests from ${EXTRA_TESTS_SRC}"
    else
        echo "WARN: extra tests not found at ${EXTRA_TESTS_SRC}; baking an empty dir."
        echo "      The upstream-verl subset still runs, but the ROCm-scoped number will be lower."
        mkdir -p "${context}/extra-tests"
    fi

    docker build ${NO_CACHE} \
        -f "${context}/Dockerfile.tester" \
        -t "${TESTER_TAG}" \
        --build-arg "BASE_IMAGE=${BASE_TAG}" \
        "${context}"
}

# ── Phase 3: measure ──────────────────────────────────────────────────────────
run_coverage() {
    log "Phase 3/3: running suites under coverage"
    if ! docker image inspect "${TESTER_TAG}" >/dev/null 2>&1; then
        echo "ERROR: tester image '${TESTER_TAG}' not found. Run with --tester first." >&2
        exit 1
    fi

    mkdir -p "${ARTIFACTS_DIR}" "${HF_CACHE}"

    # Forward every VERL_CC_* knob the caller exported.
    local env_args=()
    while read -r var; do
        [[ -z "${var}" ]] && continue
        env_args+=(-e "${var}=${!var}")
        echo "forwarding ${var}=${!var}"
    done < <(compgen -v | grep '^VERL_CC_' || true)

    docker run --rm \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --group-add render \
        --security-opt seccomp=unconfined --cap-add=SYS_PTRACE \
        --ipc=host --shm-size=8G \
        -v "${ARTIFACTS_DIR}:/artifacts" \
        -v "${HF_CACHE}:/root/.cache/huggingface" \
        -e ARTIFACTS_FOLDER=/artifacts \
        "${env_args[@]}" \
        "${DOCKER_RUN_EXTRA[@]}" \
        "${TESTER_TAG}"

    log "Artifacts in ${ARTIFACTS_DIR}"
    [[ -f "${ARTIFACTS_DIR}/result.txt" ]] && cat "${ARTIFACTS_DIR}/result.txt"
}

mkdir -p "${BUILD_DIR}"
if [[ ${RUN_BASE}   -eq 1 ]]; then build_base;    fi
if [[ ${RUN_TESTER} -eq 1 ]]; then build_tester;  fi
if [[ ${RUN_RUN}    -eq 1 ]]; then run_coverage;  fi
exit 0
