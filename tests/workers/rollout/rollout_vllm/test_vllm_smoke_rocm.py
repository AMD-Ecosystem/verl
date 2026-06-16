# Copyright 2026 AMD
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROCm vLLM rollout smoke test (code-coverage tier).

Single-GPU, tiny-model (Qwen2.5-0.5B-Instruct), single short generation through
verl's standalone vLLM rollout server. The point is COVERAGE of the real ROCm
rollout path (vllm_async_server / vllm_rollout adapter / rollout utils), not a
correctness/perf assertion -- so it only checks that generation produced tokens.

Skips cleanly (rather than failing) when the prerequisites for a meaningful run
are absent (no GPU, vllm missing, or the model cannot be fetched), so it is safe
in the curated coverage set.
"""

import asyncio
import os

import pytest

MODEL_ID = os.environ.get("VERL_SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _require(cond, reason):
    if not cond:
        pytest.skip(reason)


def test_vllm_rollout_smoke_rocm():
    # ---- Preconditions (skip, don't fail, if the tier can't run this) --------
    try:
        import torch
    except Exception:
        pytest.skip("torch not importable")
    _require(torch.cuda.is_available(), "no GPU visible")
    _require(pytest.importorskip("vllm") is not None, "vllm not installed")

    # ---- Fetch the tiny model (local snapshot) -------------------------------
    try:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(MODEL_ID)
    except Exception as exc:  # network/gated/etc.
        pytest.skip(f"could not fetch {MODEL_ID}: {exc}")

    import ray

    ray.init(
        runtime_env={"env_vars": {"VLLM_USE_V1": "1", "TOKENIZERS_PARALLELISM": "false"}},
        ignore_reinit_error=True,
    )
    try:
        from hydra import compose, initialize_config_dir

        from verl.utils.tokenizer import normalize_token_ids

        config_dir = os.path.abspath("verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/verl/trainer/config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer")

        config.trainer.n_gpus_per_node = 1
        config.trainer.nnodes = 1
        config.actor_rollout_ref.model.path = model_path
        config.actor_rollout_ref.rollout.name = "vllm"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
        config.actor_rollout_ref.rollout.prompt_length = 128
        config.actor_rollout_ref.rollout.response_length = 32
        # Keep it light: skip CUDA-graph capture, modest mem.
        for k, v in (("enforce_eager", True), ("gpu_memory_utilization", 0.5), ("free_cache_engine", True)):
            if k in config.actor_rollout_ref.rollout:
                config.actor_rollout_ref.rollout[k] = v

        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_server_class = get_rollout_replica_class("vllm")
        server = rollout_server_class(
            replica_rank=0,
            config=config.actor_rollout_ref.rollout,
            model_config=config.actor_rollout_ref.model,
            gpus_per_node=1,
        )
        asyncio.run(server.init_standalone())
        server_handle = server._server_handle

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_ids = normalize_token_ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Say hello in one short sentence."}],
                add_generation_prompt=True,
                tokenize=True,
            )
        )

        output = ray.get(
            server_handle.generate.remote(
                request_id="smoke_0",
                prompt_ids=prompt_ids,
                sampling_params={"temperature": 0.0, "top_p": 1.0, "logprobs": False},
                image_data=None,
            ),
            timeout=240.0,
        )
        assert output is not None and len(output.token_ids) > 0, "rollout produced no tokens"
        print("SMOKE OK:", tokenizer.decode(output.token_ids)[:120])
    finally:
        ray.shutdown()


if __name__ == "__main__":
    test_vllm_rollout_smoke_rocm()
