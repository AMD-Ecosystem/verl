ARG VLLM_ROCM_TAG=nightly
ARG VLLM_ROCM_REPO=rocm/vllm-dev
FROM ${VLLM_ROCM_REPO}:${VLLM_ROCM_TAG}

SHELL ["/bin/bash", "-ceuxo", "pipefail"]

ARG GPU_ARCH="gfx942;gfx942"
ARG MAX_JOBS=64

ARG RAY_SPEC="ray[default,serve]>=2.47.0"
ARG RAY_SOURCE="nightly"
ARG PYTHON_VERSION

ARG VERL_REPO=https://github.com/ROCm/verl.git
ARG VERL_BRANCH="mintz/nightly-ci-integration"

# ── Runtime environment variables ──────────────────────────────────────────────
ENV ROCM_PATH=/opt/rocm
ENV PYTORCH_ROCM_ARCH="${GPU_ARCH}"
ENV AMDGPU_TARGETS="${GPU_ARCH}"
ENV VLLM_TARGET_DEVICE=rocm

# Prevent Ray from overwriting HIP_VISIBLE_DEVICES (ray>=2.45.0 requirement)
#ENV RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1

# vLLM ROCm performance flags
ENV VLLM_ROCM_USE_AITER=1
ENV VLLM_FP8_PADDING=1
ENV VLLM_FP8_ACT_PADDING=1
ENV VLLM_FP8_WEIGHT_PADDING=1
ENV VLLM_FP8_REDUCE_CONV=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# RCCL tuning (can be overridden at runtime for multi-node)
ENV NCCL_MIN_NCHANNELS=112
ENV HSA_NO_SCRATCH_RECLAIM=1
ENV HIP_FORCE_DEV_KERNARG=1

ENV MAX_JOBS=${MAX_JOBS}

# ── Ray ───────────────────────────────────────────────────────────────────────
RUN PY_TAG="cp$(echo ${PYTHON_VERSION} | tr -d '.')" && \
    if [ "$RAY_SOURCE" = "nightly" ]; then \
      pip uninstall -y ray 2>/dev/null || true && \
      pip install -U \
        "ray[default,serve] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-${PY_TAG}-${PY_TAG}-manylinux2014_x86_64.whl"; \
    else \
      pip uninstall -y ray 2>/dev/null || true \
      && pip install --no-cache-dir "${RAY_SPEC}"; \
    fi

# ── torch_memory_saver (NUMA-aware weight/optimizer offload for verl actors) ──
#RUN pip install --no-cache-dir \
#    "git+https://github.com/YangWang92/torch_memory_saver_numa.git@numa"

# ── verl runtime dependencies ─────────────────────────────────────────────────
# tensordict>=0.10 is required by verl/protocol.py's DataProto.to_tensordict();
# verl's own requirements.txt pins tensordict>=0.8.0,<=0.10.0,!=0.9.0, so 0.10.0
# is the newest version that satisfies both constraints. tensordict>=0.7 needs
# `pyvers` (not auto-installed with --no-deps), so it's pinned explicitly here;
# tensordict 0.10.0 requires pyvers>=0.1.0,<0.2.0.
#
# megatron-core + mbridge enable the "megatron" engine backend
# (verl/workers/engine/megatron/transformer_impl.py); without megatron-core the
# backend never registers ("Unknown backend: megatron"), and without mbridge
# engine.initialize() fails with "No module named 'mbridge'" the first time it
# builds the transformer config. Version/ref match the other ROCm Dockerfiles
# (docker/rocm/Dockerfile.rocm, docker/rocm/verl0.7.1-amd0/Dockerfile).
RUN pip install --no-cache-dir \
        "tensordict==0.10.0" --no-deps \
    && pip install --no-cache-dir \
        "pyvers>=0.1.0,<0.2.0" \
    && pip install --no-cache-dir \
        "git+https://github.com/ISEEKYAN/mbridge.git" \
    && pip install --no-cache-dir \
        accelerate \
        codetiming \
        datasets \
        dill \
        hydra-core \
        liger-kernel \
        megatron-core==0.16.0 \
        numpy \
        pandas \
        peft \
        "pyarrow>=15.0.0" \
        pylatexenc \
        torchdata \
        wandb \
        orjson \
        pybind11 \
        setuptools==75.8.0

# ── verl ──────────────────────────────────────────────────────────────────────
WORKDIR /workspace
RUN git clone --branch ${VERL_BRANCH} ${VERL_REPO} verl \
    && cd verl \
    && pip install --no-cache-dir --no-deps -e .

WORKDIR /workspace/verl

# Override base image entrypoint (vllm serve) for interactive/training use
ENTRYPOINT ["/bin/bash"]
