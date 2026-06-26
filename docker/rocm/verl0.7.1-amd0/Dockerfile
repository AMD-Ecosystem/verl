ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS ubuntu

#
# Install basic packages from OS distro
#
FROM ubuntu AS base

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt update && \
    apt install -y git software-properties-common curl rsync dialog gfortran wget sqlite3

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    if ! python3 --version | grep -q ${PYTHON_VERSION} ; then \
    add-apt-repository -y ppa:deadsnakes/ppa && apt update && \
    apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
                       python${PYTHON_VERSION}-lib2to3 python-is-python3 ; fi

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}



RUN wget -nv -O /tmp/cmake-3.26.4-linux-x86_64.tar.gz https://cmake.org/files/v3.26/cmake-3.26.4-linux-x86_64.tar.gz && \
    tar zfx /tmp/cmake-3.26.4-linux-x86_64.tar.gz -C /opt/ && \
    mv /opt/cmake-3.26.4-linux-x86_64 /opt/cmake-3.26.4 && \
    rm -f /tmp/cmake-3.26.4-linux-x86_64.tar.gz

ENV PATH=/opt/cmake-3.26.4/bin:$PATH

#
# Install ROCm rpm packages
#
FROM base AS rocm_deb

ARG ROCM_VERSION=7.0.2
ARG AMDGPU_VERSION=7.0.2
ARG GFX_ARCH=gfx942|gfx950

RUN curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
    && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ jammy main\n" | tee /etc/apt/sources.list.d/rocm.list \
    && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu jammy main\n" | tee /etc/apt/sources.list.d/amdgpu.list \
    && printf "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" | tee /etc/apt/preferences.d/rocm-pin-600 \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y rocm && \
    find /opt/rocm/lib -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/hipblaslt/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/lib/rocblas/library -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    find /opt/rocm/share/miopen/db -type f -name '*gfx*' | grep -Ev "${GFX_ARCH}" | xargs rm -f && \
    sqlite3 /opt/rocm/lib/rocfft/rocfft_kernel_cache.db "delete from cache_v1 where arch != '${GFX_ARCH}' ; vacuum"

ENV ROCM_HOME=/opt/rocm
ENV CPLUS_INCLUDE_PATH=/opt/rocm/include:
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH

#
# Install pytorch packages
#
FROM rocm_deb AS rocm_torch


RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip "setuptools<80" wheel numpy einops ninja && \
    pip install /opt/rocm/share/amd_smi

RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torch-2.9.1.dev20251204+rocm7.0.2.lw.git351ff442-cp312-cp312-linux_x86_64.whl -O torch-2.9.1.dev20251204+rocm7.0.2.lw.git351ff442-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/apex-1.9.0a0+rocm7.0.2.git07c3ee53-cp312-cp312-linux_x86_64.whl -O apex-1.9.0a0+rocm7.0.2.git07c3ee53-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchaudio-2.9.0+rocm7.0.2.gite3c6ee2b-cp312-cp312-linux_x86_64.whl -O torchaudio-2.9.0+rocm7.0.2.gite3c6ee2b-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/torchvision-0.24.0+rocm7.0.2.gitb919bd0c-cp312-cp312-linux_x86_64.whl -O torchvision-0.24.0+rocm7.0.2.gitb919bd0c-cp312-cp312-linux_x86_64.whl && \
    wget -nv https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/triton-3.5.1+rocm7.0.2.gita272dfa8-cp312-cp312-linux_x86_64.whl -O triton-3.5.1+rocm7.0.2.gita272dfa8-cp312-cp312-linux_x86_64.whl && \
    pip3 install *.whl && \
    rm -f *.whl

ARG GPU_ARCH=gfx942;gfx950
ENV PYTORCH_ROCM_ARCH=${GPU_ARCH}

WORKDIR /root

FROM rocm_torch AS new_toolset

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install numpy einops packaging psutil ninja

#
# Build Flash Attention wheel
#
FROM new_toolset AS fa_build

ARG FA_REPO="https://github.com/ROCm/flash-attention"
ARG FA_TAG="83f9e450cd10e20701fb109db9c7703d376f282b"

RUN git clone ${FA_REPO} \
 && cd flash-attention \
 && git checkout ${FA_TAG} \
 && git submodule init \
 && git submodule update

RUN cd flash-attention \
 && printf '[BUILD-DIAG fa_build] nproc=%s cpu.max=%s %s\n' "$(nproc)" "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo NA)" "$(grep -E 'MemTotal|MemAvailable' /proc/meminfo | tr '\n' ' ')" \
 && GPU_ARCHS=${GPU_ARCH} BUILD_TARGET=rocm MAX_JOBS=$(nproc) python3 setup.py bdist_wheel \
 && mkdir /install && cp dist/*.whl /install

#
# Install Flash Attention
#
FROM new_toolset AS install_fa

RUN --mount=type=bind,from=fa_build,source=/install,target=/tmp/install \
    --mount=type=cache,target=/root/.cache/pip \
    pip install /tmp/install/*.whl

#
# Install TE
#
FROM install_fa AS te

ENV NVTE_USE_HIPBLASLT=1
ENV NVTE_USE_ROCM=1
ENV NVTE_FRAMEWORK=pytorch
ENV NVTE_ROCM_ARCH=${GPU_ARCH}
ENV NVTE_USE_CAST_TRANSPOSE_TRITON=0
# Skip the from-source CK/AITER fused-attention build (the multi-hour ck_fused_attn
# compile). Set as ENV so it also applies at runtime, where TE requires the same
# flag for any backend disabled at compile time. AOTriton remains the fused-attn backend.
ENV NVTE_FUSED_ATTN_CK=0
ENV NVTE_CK_USES_BWD_V3=1
ENV NVTE_CK_V3_BF16_CVT=2

ARG TE_TAG="386bd316"
RUN pip install pybind11 && git clone https://github.com/ROCm/TransformerEngine.git && \
    cd TransformerEngine && git checkout ${TE_TAG} && git submodule update --init --recursive && \
    printf '[BUILD-DIAG te] nproc=%s cpu.max=%s %s\n' "$(nproc)" "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo NA)" "$(grep -E 'MemTotal|MemAvailable' /proc/meminfo | tr '\n' ' ')" && \
    GPU_ARCHS=${GPU_ARCH} MAX_JOBS=$(nproc) python3 setup.py install && \
    cd .. && rm -rf TransformerEngine
# TE patch
RUN FILE=$(find /usr/local/lib/python3.12/dist-packages/ -path "*transformer_engine*/transformer_engine/pytorch/attention/dot_product_attention/utils.py") && \
    sed -i '121s/2\.8\.3/2.8.4/' "$FILE"
#
# Install vllm
#
FROM te AS install_vllm

ARG VLLM_TAG="1ff9d3353"
# Cap vLLM compile parallelism: the vLLM/CSRC build is the most memory-hungry
# step (each job ~2GB), and on high-core CI hosts $(nproc) (e.g. 256) can exhaust
# RAM and OOM. Clamp to min(nproc, VLLM_MAX_JOBS) so it still scales down on
# small runners. The last known-good CI build used 64.
ARG VLLM_MAX_JOBS=64
RUN pip install setuptools_scm && \
    mkdir /workspace && cd /workspace && \
    ln -sf /opt/rocm/lib/libamdhip64.so /usr/lib/libamdhip64.so && \
    git clone https://github.com/vllm-project/vllm && \
    cd vllm && git checkout ${VLLM_TAG} && pip install -r requirements/rocm.txt && \
    VLLM_JOBS=$(n=$(nproc); cap=${VLLM_MAX_JOBS}; echo $(( n < cap ? n : cap ))) && \
    printf '[BUILD-DIAG vllm] nproc=%s MAX_JOBS=%s cpu.max=%s %s\n' "$(nproc)" "${VLLM_JOBS}" "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo NA)" "$(grep -E 'MemTotal|MemAvailable' /proc/meminfo | tr '\n' ' ')" && \
    MAX_JOBS=${VLLM_JOBS} python3 setup.py develop --no-deps
# vLLM compat patch: env_override.py monkeypatches torch._dynamo for torch<2.12,
# importing GraphCaptureOutput (PyTorch PR #177558) unconditionally. The pinned
# ROCm torch wheel (2.9.1.dev20251204) predates that symbol, so the unguarded
# import crashes every `import vllm`. Gate the block on the symbol actually
# existing; the patch is a no-op once vLLM/torch ship a compatible pair.
RUN F=/workspace/vllm/vllm/env_override.py && \
    if grep -q 'if not is_torch_equal_or_newer("2.12.0"):' "$F"; then \
        sed -i 's|if not is_torch_equal_or_newer("2.12.0"):|if not is_torch_equal_or_newer("2.12.0") and hasattr(__import__("torch._dynamo.convert_frame", fromlist=["GraphCaptureOutput"]), "GraphCaptureOutput"):|' "$F" && \
        echo "[patch] vllm env_override GraphCaptureOutput guard applied"; \
    else echo "[patch] vllm env_override guard line not found; skipping"; fi
RUN pip uninstall tilelang -y && pip install xgrammar==0.1.32

RUN apt-get update && apt install vim -y

FROM install_vllm AS install_verl
RUN cd /workspace && git clone --recursive -b amd-integration https://github.com/ROCm/verl.git && \
    cd verl && pip install -e .
RUN pip install cupy-rocm-7-0


ENV MIOPEN_DEBUG_CONV_DIRECT=0
ARG AITER_TAG="45c428e54"
RUN cd /workspace && git clone --recursive https://github.com/ROCm/aiter.git && cd aiter && git checkout ${AITER_TAG} && printf '[BUILD-DIAG aiter] nproc=%s cpu.max=%s %s\n' "$(nproc)" "$(cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo NA)" "$(grep -E 'MemTotal|MemAvailable' /proc/meminfo | tr '\n' ' ')" && MAX_JOBS=$(nproc) python3 setup.py develop

# for qwen3.5
RUN pip install -U git+https://github.com/ISEEKYAN/mbridge.git && \
    pip install transformers --upgrade && \
    pip install megatron-core==0.16.0 && \
    pip install mathruler && \
    pip install qwen_vl_utils && \
    pip install flash-linear-attention && \
    pip install fastapi==0.136.3
