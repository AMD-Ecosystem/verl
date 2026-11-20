# ROCm (AMD GPU) Dockerfile of verl

This directory provides the Docker recipe for running verl on **AMD GPUs with
the ROCm software stack**. The NVIDIA images described in
[`../README.md`](../README.md) do not work on AMD hardware, so use
[`Dockerfile.rocm`](Dockerfile.rocm) instead.

For an end-to-end walkthrough (build, run, and example PPO/GRPO commands), see
the tutorial: [`docs/amd_tutorial/amd_quick_start.rst`](../../docs/amd_tutorial/amd_quick_start.rst).

> The other `Dockerfile.rocm*` / `Apptainerfile.rocm` files in this directory are
> kept only as historical references for older verl releases (ROCm 6.x, pinned
> verl 0.3.x / 0.4.x). New work should target `Dockerfile.rocm`.

## Supported Hardware

The image targets the following GPU architectures (set via `PYTORCH_ROCM_ARCH`
in `Dockerfile.rocm`):

- `gfx942` — MI300 series (MI300X / MI300A / MI325X)
- `gfx950` — MI350 series (MI350X / MI355X)

Other architectures (e.g. `gfx90a` for MI200/MI250) are not validated here.

## Key Versions

| Component | Version |
| --------- | ------- |
| Base image | `rocm/primus:v26.9.1` (ROCm, torch, triton, Flash Attention, TransformerEngine, aiter, amdsmi) |
| Python | 3.12 (`/opt/venv`) |
| vLLM | v0.32.0 source @ `5df63a720` + [vllm#61872](https://github.com/vllm-project/vllm/pull/61872) (`3b43d2b01c`) |
| SGLang | source @ `0801cc05ed` (Python package installed with `--no-deps`) |
| verl | `VERL_SOURCE=release` (default) checks out `VERL_TAG` `3503bec34`; `nightly` checks out upstream `main` |
| causal-conv1d | `e940ead2` |
| megatron-core | 0.19.2 |
| megatron-bridge | 0.6.2 (`--no-deps`) |
| megatron-energon | 7.4.1 (`--no-deps`) |
| transformers | `>=5.12` (upgraded in the verl layer; resolved version not pinned) |
| wandb | 0.29.0 |
| flash-linear-attention | `main` from GitHub (not pinned) |
| TileLang | 0.1.10 (from vLLM's requirements) |
| cupy | `cupy-rocm-7-0` (not pinned) |

ROCm, torch, triton, Flash Attention, TransformerEngine, and aiter come from the
base image and are not pinned in `Dockerfile.rocm`. During the vLLM build,
`/tmp/rocm-constraints.txt` pins torch to the version the base image provides,
so pip cannot replace it with a CUDA wheel.

## What the Image Contains

Starting from the base image `rocm/primus:v26.9.1`, `Dockerfile.rocm` adds:

**Provided by the base image (not installed here):**
- ROCm runtime, `torch`, `torchvision`, `torchaudio`, `triton`
- Flash Attention, TransformerEngine (ROCm fork), aiter
- `amdsmi`, made importable by a site-packages `.pth` file

**Built from source (pinned commits):**
- causal-conv1d (`e940ead2`)
- vLLM (`5df63a720` + vllm#61872)
- verl, installed with `pip install .`. `VERL_SOURCE=release` (the default) checks out `VERL_TAG` (`3503bec34`). `VERL_SOURCE=nightly` checks out upstream `main` at build time. Any other value fails the build.
- SGLang (`0801cc05ed`): the Python package and `sgl-kernel`

**Installed from PyPI or GitHub:** `megatron-core` 0.19.2, `megatron-bridge`
0.6.2, `megatron-energon` 7.4.1, `multi-storage-client`, `braceexpand`,
`filetype`, `mfusepy`, `rapidyaml`, `webdataset`, `s3fs` 2025.3.0,
`transformers` >=5.12, `accelerate`, `wandb` 0.29.0, `torchao`, `torchdata`,
`cupy-rocm-7-0`, `flash-linear-attention`, `mistral-common`, `mathruler`,
`model-hosting-container-standards`, `transferqueue`, and the SGLang runtime
dependencies.

**Patches:** `docker/rocm/patches/` is copied into the image and applied after
the last pip install. The eight patches cover TileLang HIP FP8/FP4 headers and
the verl QAT and Qwen image-pad decode code. The last four also apply to
`/workspace/verl`, since tests import from there.

**Runtime fixes:**
- megatron-core's attention-backend assertion is converted to a warning, because
  the base image sets `NVTE_FLASH_ATTN=0` and `NVTE_FUSED_ATTN=1`.
- `rocm_sdk` preloads libraries with `RTLD_LOCAL` so that `amdsmi` and
  `torch.cuda.device_count()` work together.
- `z3-solver` 4.15.4.0 is reinstalled so that tilelang can load `libz3.so.4.15`.

## Building Locally

Build from the repository root. `Dockerfile.rocm` copies
`docker/rocm/patches/`, so the build context must include that directory.

```sh
DOCKER_BUILDKIT=1 docker build \
    -f docker/rocm/Dockerfile.rocm \
    -t verl-rocm:local .
```

The vLLM build uses `MAX_JOBS=64`.

### Useful build arguments

| Build arg | Default | Purpose |
| --------- | ------- | ------- |
| `BASE_IMAGE` | `rocm/primus:v26.9.1` | Base image providing ROCm, torch, Flash Attention, TransformerEngine and aiter. |
| `VLLM_TAG` | `5df63a7203cdb9328d824a27b682e77c7b9928be` | vLLM commit built from source. |
| `VLLM_PR_61872` | `3b43d2b01c2fb667101746e1acd33b6a3c9d0863` | vLLM PR applied to `vllm/` after checkout. |
| `SGLANG_TAG` | `0801cc05ed202544e2e5c67d6b0a1d454de54fca` | SGLang commit built from source. |
| `VERL_SOURCE` | `release` | `nightly` checks out upstream `main`. `release` checks out `VERL_TAG`. |
| `VERL_TAG` | `3503bec3420dbb8ee67f8c9da6c0721570ae75ac` | verl commit used when `VERL_SOURCE=release`. |

## Release History

- 2026/11/20: `VERL_SOURCE` selects the verl checkout. `release` (the default)
  checks out `VERL_TAG`. `nightly` checks out upstream `main` at build time.

- 2026/11/20: `BASE_IMAGE` moved to `rocm/primus:v26.9.1`. The Qwen3.5 Megatron
  stack now matches `pipelines/verl/Dockerfile` (nightly path): megatron-bridge
  0.6.2, megatron-energon 7.4.1, megatron-core 0.19.2, and wandb 0.29.0, replacing
  mbridge, mcore_bridge 1.7.0, megatron-core 0.18.0, and the pinned
  megatron-bridge 0.5.1. Other changes: the vLLM install is constrained to the
  base image's torch; the SGLang runtime dependency list is installed under the
  same constraint and built as C++20; amdsmi is loaded through a `.pth` file;
  and the pipeline's TileLang and verl patches are applied. Commits
  `d9df96bb` and `8fe9d2b9` on `release/0.9.1.amd0`.

- 2026/11/13: vLLM v0.32.0 @`5df63a720` (+ vllm#61872), aligned with verl's CUDA
  stack; TileLang 0.1.10 kept (no longer uninstalled); transformers==5.16.1,
  mcore_bridge==1.7.0, TransferQueue==0.1.11, verl @`3503bec34`. Same ROCm 7.14
  base (`rocm/primus:v26.4`), torch==2.12.0, triton==3.7.0; targets gfx942 / gfx950.
  Prebuilt as `verlai/verl:rocm7.14_torch2.12_release_1112`.

- 2026/07/20: ROCm 7.14 stack — torch==2.12.0, triton==3.7.0, vLLM @`18d87a87d`,   SGLang @`0801cc05ed`, 
  Flash Attention (CK) @`v2.8.3`, TransformerEngine @`e6ede467`,
  aiter @`b5e03ed19`, megatron-core==0.18.0; targets gfx942 / gfx950.

- 2026/06/03: ROCm 7.0.2 stack — torch==2.9.1, triton==3.5.1, vLLM @`1ff9d3353`,
  Flash Attention (CK) @`83f9e450`, TransformerEngine @`386bd316`,
  aiter @`45c428e54`, megatron-core==0.16.0; targets gfx942 / gfx950.
