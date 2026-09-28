# ROCm Dockerfile for verl 0.9.0.amd0

[`Dockerfile.rocm`](Dockerfile.rocm) builds a training image for **AMD Instinct GPUs**
from `rocm/primus:v26.7`. Primus ships the training stack. This Dockerfile adds
the pieces it does not: **Ray 2.58.0**, **vLLM 0.27.0** built from source,
SGLang, verl `release/0.9.0.amd0`, and the Qwen3.5 extras.

The NVIDIA Dockerfiles in [`../README.md`](../README.md) do not run on AMD
hardware. Use `Dockerfile.rocm` for this release.

Other `Dockerfile.rocm*` and `Apptainerfile.rocm` files in this directory are
older recipes. New work should use `Dockerfile.rocm`.

## What you get

| Component | Pin in `Dockerfile.rocm` |
| --- | --- |
| Base image | `rocm/primus:v26.7` |
| verl | `AMD-Ecosystem/verl` branch `release/0.9.0.amd0` |
| Python | 3.12 |
| Ray | `ray[default,serve]==2.58.0` |
| vLLM | tag `v0.27.0`, built from source with `VLLM_TARGET_DEVICE=rocm` |
| GPU arch | `gfx942` (MI300 series) and `gfx950` (MI350 series) |
| Megatron-Core | 0.18.0 |
| CuPy | `cupy-rocm-7-0` |

The image also installs SGLang from a pinned commit, `amdsmi` so Ray can see
Instinct GPUs, and sets `SGLANG_ATTENTION_BACKEND=triton`.

## Prerequisites

1. A ROCm host driver that matches the Primus v26.7 image.
2. Docker with access to `/dev/kfd` and `/dev/dri`.
3. BuildKit. If the build fails with `--mount option requires BuildKit`, install
   the `buildx` plugin and prefix the command with `DOCKER_BUILDKIT=1`.

## Install with vLLM 0.27.0 and Ray 2.58.0

Those versions are the Dockerfile defaults. Pass them explicitly so a later
default change cannot change the image.

From a clone of this branch:

```bash
git clone --recursive -b release/0.9.0.amd0 https://github.com/AMD-Ecosystem/verl.git
cd verl

DOCKER_BUILDKIT=1 docker build \
  -f docker/rocm/Dockerfile.rocm \
  --build-arg RAY_SOURCE=stable \
  --build-arg RAY_SPEC='ray[default,serve]==2.58.0' \
  --build-arg VLLM_REPO=https://github.com/vllm-project/vllm.git \
  --build-arg VLLM_TAG=v0.27.0 \
  --build-arg VERL_SOURCE=release \
  --build-arg VERL_BRANCH=release/0.9.0.amd0 \
  --build-arg GPU_ARCH='gfx942;gfx950' \
  --build-arg MAX_JOBS=64 \
  -t verl-rocm:0.9.0.amd0-vllm0.27.0-ray2.58.0 \
  .
```

What those arguments do:

| Build arg | Value | Effect |
| --- | --- | --- |
| `RAY_SOURCE` | `stable` | Installs the pinned wheel. `nightly` ignores `RAY_SPEC` and installs a Ray 3.0 dev wheel. |
| `RAY_SPEC` | `ray[default,serve]==2.58.0` | Uninstalls the Ray package already in the base image, then installs Ray **2.58.0** with the `default` and `serve` extras. |
| `VLLM_TAG` | `v0.27.0` | Clones vLLM at that tag and builds it for ROCm. Do not use `main`; newer trees need a `torch::stable::Tensor` API this Primus PyTorch does not provide. |
| `VERL_SOURCE` | `release` | Clones `VERL_REPO` at `VERL_BRANCH`. `nightly` clones upstream `verl-project/verl` `main` instead. |
| `GPU_ARCH` | `gfx942;gfx950` | Kernels for MI300 and MI350. Use one arch to shorten the vLLM and SGLang builds. |
| `MAX_JOBS` | `64` | Parallel compile jobs. Lower this if the vLLM or `sgl-kernel` build runs out of memory. |

Build for one GPU family:

```bash
# MI355X / MI350X
--build-arg GPU_ARCH=gfx950

# MI300X / MI325X
--build-arg GPU_ARCH=gfx942
```

## Run the image

```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 64G \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  verl-rocm:0.9.0.amd0-vllm0.27.0-ray2.58.0
```

Do not mount a host directory over `/workspace`. That hides the image's
`/workspace/verl`, `/workspace/vllm`, and `/workspace/sglang` trees. Mount a
subdirectory, such as `/workspace/logs`, if you need to keep files on the host.

## Check the install

Inside the container:

```bash
python3 - <<'PY'
import ray, vllm, torch
print("ray ", ray.__version__)
print("vllm", vllm.__version__)
print("torch", torch.__version__, "hip", torch.version.hip)
print("gpus", torch.cuda.device_count())
PY
python3 -c "import amdsmi; print('amdsmi ok')"
```

Expect `ray  2.58.0` and `vllm 0.27.0`. `torch.cuda.device_count()` must match
the GPUs `rocm-smi` lists. A count of `0` means Ray will later fail with
"Total available GPUs 0 is less than total desired GPUs".

## Choose the rollout engine

Both engines are in the image. verl selects one at launch:

```bash
# vLLM (image default path)
actor_rollout_ref.rollout.name=vllm

# SGLang
actor_rollout_ref.rollout.name=sglang
```

vLLM AITER flags (`VLLM_ROCM_USE_AITER=1` and the FP8 padding variables) are
set in the image. SGLang defaults are `SGLANG_USE_AITER=0` and
`SGLANG_ATTENTION_BACKEND=triton`. Keep the Triton attention backend on ROCm.
