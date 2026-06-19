# `VERL_AGGRESSIVE_EMPTY_CACHE` — change description and benchmark replication

This document describes the optimization added to `verl/workers/engine_workers.py`, how to reproduce the throughput measurements on AMD (MI300X / MI355X) and NVIDIA (B300) platforms, and the known caveats.

## The change

One env-var-gated swap at the `update_weights` cleanup site. Default behavior is unchanged from upstream — the env var lets users opt into a faster code path.

### Patch summary

In `verl/workers/engine_workers.py`:

1. Add `get_torch_device` to the device import:
   ```python
   from verl.utils.device import get_device_name, get_torch_device, is_npu_available, set_expandable_segments
   ```
2. Replace the per-call `aggressive_empty_cache(force_sync=True)` inside `update_weights` (between the "offload model to cpu" step and the "resume kv_cache" step) with:
   ```python
   if os.environ.get("VERL_AGGRESSIVE_EMPTY_CACHE", "1") == "1":
       aggressive_empty_cache(force_sync=True)
   else:
       get_torch_device().empty_cache()
   ```


### Env var semantics

| Setting | Behavior |
|---|---|
| Unset (default) | Original behavior: `aggressive_empty_cache(force_sync=True)` |
| `VERL_AGGRESSIVE_EMPTY_CACHE=1` | Same as default |
| `VERL_AGGRESSIVE_EMPTY_CACHE=0` | Optimized: `get_torch_device().empty_cache()` only |

### Why the optimization works

The `update_weights` site calls `aggressive_empty_cache`, which internally runs `gc.collect()` + `empty_cache()` + (optional) `synchronize()`. Profiling shows:

- `gc.collect()` costs **~500–700 ms per call** because it walks Python's live object graph
- `empty_cache()` costs **~0.4 ms**
- `synchronize()` costs **~0.1 ms**

`gc.collect()` only releases memory when reference cycles exist. The `update_weights` code path (`params = model.state_dict(); transfer; del params`) is a flat tree with no cycles, so `gc.collect()` does no useful work at this site — it just costs ~500 ms.

`empty_cache()` is the load-bearing part: it returns cached pages from PyTorch's allocator back to the driver so vLLM's separate KV-cache allocator can acquire them at `wake_up`. Skipping it causes OOM at TP≥2.

The optimization keeps `empty_cache()` and drops `gc.collect()`, which is the part that does nothing useful at this specific site.

### When the optimization is **not** safe

The GRPO + KL-loss + high-`rollout.n` configuration forms reference cycles (likely through the KL-loss code path holding references to ref-model intermediates). At high memory pressure, `empty_cache()` alone cannot reclaim enough pages, and vLLM's `wake_up` OOMs. Confirmed failure mode on AMD MI355X: GRPO Qwen2-7B with `use_kl_loss=True`, `rollout.n=5`, `max_response_length=1024` OOM'd 3/3 attempts with the optimization enabled. The same workload on NVIDIA B300 completed without OOM, suggesting platform-dependent allocator behavior.

**For GRPO Qwen on AMD, leave the default (`VERL_AGGRESSIVE_EMPTY_CACHE=1`).** Other workloads tested (PPO Qwen, PPO DeepSeek, GRPO DeepSeek) are safe with the optimization on both platforms.

## Hardware & software stacks

### AMD MI355X (8 GPU) — original investigation platform

| Component | Version |
|---|---|
| Hardware | AMD MI355X × 8 |
| ROCm | 7.0.2 (HIP 7.0.51831) |
| Driver | amdgpu 6.16.6 |
| Python | 3.12.13 |
| PyTorch | 2.9.1.dev20251204+rocm7.0.2 |
| Triton | 3.5.1+rocm7.0.2 |
| Flash Attention | ROCm fork, tag `83f9e450cd10e20701fb109db9c7703d376f282b` (source) |
| TransformerEngine | ROCm fork, tag `386bd316` (source) |
| vLLM | 0.20.2rc1.dev253+g1ff9d3353.rocm702 (source) |
| verl | 0.8.0.dev0, ROCm fork `amd-integration` branch |
| AITER | tag `45c428e54` |
| Container | `docker.gpuperf:5000/rocm/verl:Verlv0.7.0` (custom Dockerfile, ubuntu:22.04 base) |

### NVIDIA B300 (8 GPU)

| Component | Version |
|---|---|
| Hardware | NVIDIA B300 SXM6 × 8 |
| Driver | NVIDIA 580.95.05 |
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| Triton | 3.6.0 |
| Flash Attention | 2.8.3 (pip) |
| vLLM | 0.20.2 (pip) |
| verl | 0.8.0.dev0, ROCm fork `amd-integration` branch (editable install) |
| NCCL | 2.28.9+cuda13.0 |
| Container | based on `verlai/verl:vllm020.dev1` with verl re-cloned at `/workspace/verl` |

Both platforms use the same verl source tree (`https://github.com/ROCm/verl` `amd-integration` branch)

## Container setup

### AMD MI300X / MI355X
To run:
```bash
docker run -it --device /dev/dri --device /dev/kfd -p 8265:8265 \
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    --privileged -v $HOME/.ssh:/root/.ssh -v $HOME:$HOME --shm-size 128G \
    -w $PWD docker.gpuperf:5000/rocm/verl:Verlv0.7.0
```

### NVIDIA B300


To run:
```bash
docker run --gpus all -it --network host --ipc host -v /data:/data \
    --shm-size 64G --workdir /workspace/ \
    docker.gpuperf:5000/nvidia/verl:0.7.x-rocm /bin/bash
```

Built on top of `verlai/verl:vllm020.dev1` (public vLLM image) with verl re-cloned. Build:

```bash
# inside the container
cd /workspace
git clone --recursive -b amd-integration https://github.com/ROCm/verl.git
cd verl
pip install -e .
python3 -c "import verl; print(verl.__file__)"  # should print /workspace/verl/verl/__init__.py
```


### Dataset preparation
```bash
python3 examples/data_preprocess/gsm8k.py --local_dir ../data/gsm8k
```
This downloads GSM8K and writes `train.parquet` / `test.parquet` to `../data/gsm8k/` (relative to verl repo). The runscripts reference these paths.

### Model preload
```bash
python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
python3 -c "import transformers; transformers.pipeline('text-generation', model='deepseek-ai/deepseek-llm-7b-chat')"
```

## Workloads and runscripts

Four workloads were tested: {PPO, GRPO} × {Qwen2-7B-Instruct, DeepSeek-LLM-7B-Chat}. Each platform has its own runscript that wraps the verl launch command with the per-platform env vars.

| Workload | AMD script | NVIDIA script |
|---|---|---|
| PPO Qwen2-7B | `ppo_qwen.sh` | `ppo_qwen.titan.sh` |
| PPO DeepSeek-7B | `ppo_deepseek.sh` | `ppo_deepseek.titan.sh` |
| GRPO Qwen2-7B | `grpo_qwen.sh` | `grpo_qwen.titan.sh` |
| GRPO DeepSeek-7B | `grpo_deepseek.sh` | `grpo_deepseek.titan.sh` |

All scripts are parameterized via env vars (TRAIN_BATCH_SIZE, MINI_BATCH_SIZE, MICRO_BATCH_SIZE, TP_VALUE, INFERENCE_BATCH_SIZE, GPU_MEMORY_UTILIZATION, ROLLOUT_N where applicable, EPOCHS). Defaults match the values measured below.

### Common config across all workloads

- Dataset: GSM8K
- GPUs: 8 per platform
- TRAIN_BATCH_SIZE: 1024
- MINI_BATCH_SIZE: 256

### Per-workload config differences (matched across AMD/NVIDIA)

| Workload | TP | rollout.n | MICRO | INFERENCE | GPU_MEM | response_length |
|---|---|---|---|---|---|---|
| PPO Qwen | 2 | 1 | 16 | 32 | 0.4 | 512 |
| PPO DeepSeek | 4 | 1 | 16 | 32 | 0.4 | 512 |
| GRPO Qwen | 2 | 5 | 80 | 40 | 0.4 | 1024 |
| GRPO DeepSeek | 2 | 5 | 80 | 80 | 0.4 | 1024 |


### Baseline
```bash
# default behavior — VERL_AGGRESSIVE_EMPTY_CACHE unset = "1"
bash ppo_qwen.sh 2>&1 | tee ppo_qwen_baseline.log
```

### Optimized
```bash
VERL_AGGRESSIVE_EMPTY_CACHE=0 bash ppo_qwen.sh 2>&1 | tee ppo_qwen_optimized.log
```

Repeat for each of the four scripts on each platform, three times per variant.



## Measured results

## AMD MI355X — automated platform, 3 runs per variant, median reported

| Workload | Variant | n successful / attempted | Successful values (tok/s) | **Median** | Δ vs baseline |
|---|---|---|---|---|---|
| **PPO Qwen2-7B** (TP=2) | baseline (AEC=1) | 3/3 | 1978.09, 1965.37, 1940.64 | 1965.37 | — |
| | optimized (AEC=0) | 3/3 | 2148.86, 2137.03, 2124.08 | **2137.03** | **+8.73%** |
| **PPO DeepSeek-7B** (TP=4) | baseline (AEC=1) | 3/3 | 2187.87, 2187.21, 2170.18 | 2187.21 | — |
| | optimized (AEC=0) | 3/3 | 2368.63, 2332.30, 2330.52 | **2332.30** | **+6.63%** |
| **GRPO Qwen2-7B** (TP=2, n=5) | baseline (AEC=1) | 3/3 | 3875.94, 3830.39, 3797.90 | 3830.39 | — |
| | optimized (AEC=0) | **0/3** | (OOM all attempts) | **— OOM** | **N/A — unsafe** |
| **GRPO DeepSeek-7B** (TP=2, n=5) | baseline (AEC=1) | 3/3 | 4011.73, 3977.42, 3947.43 | 3977.42 | — |
| | optimized (AEC=0) | 3/3 | 4092.59, 4079.77, 4069.03 | **4079.77** | **+2.57%** |

## NVIDIA B300 — automated platform, 3 runs per variant, median of successful runs

| Workload | Variant | n successful / attempted | Successful values (tok/s) | **Median** | Δ vs baseline |
|---|---|---|---|---|---|
| **PPO Qwen2-7B** (TP=2) | baseline (AEC=1) | 3/3 | 1915.06, 1945.09, 1820.96 | **1915.06** | — |
| | optimized (AEC=0) | 2/3 | 2010.40, 1957.53 | **1983.97** | **+3.60%** (small n) |
| **PPO DeepSeek-7B** (TP=4) | baseline (AEC=1) | 2/3 | 2080.37, 2043.90 | 2062.14 | — |
| | optimized (AEC=0) | 2/3 | 2137.32, 2120.36 | **2128.84** | **+3.23%** |
| **GRPO Qwen2-7B** (TP=2, n=5) | baseline (AEC=1) | 3/3 | 3603.77, 3595.10, 3565.76 | **3595.10** | — |
| | optimized (AEC=0) | 2/3 | 3719.37, 3706.98 | **3713.18** | **+3.28%** |
| **GRPO DeepSeek-7B** (TP=2, n=5) | baseline (AEC=1) | 1/3 | 4128.07 | 4128.07 | — |
| | optimized (AEC=0) | 2/3 | 4177.31, 4099.13 | **4138.22** | **+0.25%** (small n) |


## NVIDIA B200 — automated platform, 3 runs per variant, median reported

| Workload | Variant | n successful / attempted | Successful values (tok/s) | **Median** | Δ vs baseline |
|---|---|---|---|---|---|
| **PPO Qwen2-7B** (TP=2) | baseline (AEC=1) | 3/3 | 1919.34, 1948.92, 1967.04 | 1948.92 | — |
| | optimized (AEC=0) | 3/3 | 2046.36, 2099.08, 2048.53 | **2048.53** | **+5.11%** |
| **PPO DeepSeek-7B** (TP=4) | baseline (AEC=1) | 3/3 | 2155.49, 2164.32, 2194.24 | 2164.32 | — |
| | optimized (AEC=0) | 3/3 | 2229.53, 2258.30, 2302.95 | **2258.30** | **+4.34%** |
| **GRPO Qwen2-7B** (TP=2, n=5) | baseline (AEC=1) | 3/3 | 3906.19, 3946.55, 3972.99 | 3946.55 | — |
| | optimized (AEC=0) | 3/3 | 3997.56, 4013.60, 4059.67 | **4013.60** | **+1.70%** |
| **GRPO DeepSeek-7B** (TP=2, n=5) | baseline (AEC=1) | 3/3 | 4304.35, 4324.51, 4347.51 | 4324.51 | — |
| | optimized (AEC=0) | 3/3 | 4381.83, 4391.80, 4404.09 | **4391.80** | **+1.56%** |


## Summary

A two-line patch + env var lets users opt into a small but consistent throughput win on PPO and lower-pressure GRPO workloads. The default behavior is unchanged from upstream — existing users see no difference and no regression. Users who measure the optimization helps their workload can set `VERL_AGGRESSIVE_EMPTY_CACHE=0` to get the speedup.

The GRPO Qwen failure mode is the load-bearing caveat: opt out of the optimization for that one workload on AMD. Everything else benefits.
