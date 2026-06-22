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
"""ROCm code-coverage tests for verl.utils.fsdp_utils.

Two groups:

1. Pure-python helpers (no GPU / no torch.distributed): wrap-policy selection,
   fsdp_version, optimizer offload/load, the meta-device init context manager,
   and the shard-placement helper. These run on CPU and always execute.

2. A SINGLE-GPU FSDP2 path (world_size=1, spawned via mp.spawn so distributed
   state stays isolated from the rest of the curated suite). It wraps a tiny
   Qwen2 model with fully_shard and exercises offload/load to CPU/GPU, the full
   state-dict collection, fsdp2 grad-norm clipping, and reshard-after-forward.
   Genuinely multi-rank FSDP1 sharding paths are out of scope here -- they
   belong to the multi-GPU coverage tier.

The point is COVERAGE of verl's own ROCm-relevant code, not numerical
correctness, so assertions are intentionally light.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Group 1 -- pure-python helpers (CPU, no distributed)
# ---------------------------------------------------------------------------


class _MockDecoderLayer(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)


class _MockCausalLM(nn.Module):
    _no_split_modules = ["_MockDecoderLayer"]

    def __init__(self, hidden=32, layers=2):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.embed = nn.Embedding(16, hidden)
        self.layers = nn.ModuleList([_MockDecoderLayer(hidden) for _ in range(layers)])
        self.lm_head = nn.Linear(hidden, 16, bias=False)


def test_get_fsdp_wrap_policy_disable():
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    assert get_fsdp_wrap_policy(_MockCausalLM(), config={"disable": True}) is None


def test_get_fsdp_wrap_policy_min_num_params():
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    policy = get_fsdp_wrap_policy(_MockCausalLM(), config={"min_num_params": 100})
    assert policy is not None


def test_get_fsdp_wrap_policy_transformer_cls():
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    # default_transformer_cls_names_to_wrap comes from _no_split_modules
    policy = get_fsdp_wrap_policy(_MockCausalLM(), config={})
    assert policy is not None


def test_get_fsdp_wrap_policy_lora():
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    policy = get_fsdp_wrap_policy(_MockCausalLM(), config={}, is_lora=True)
    assert policy is not None


def test_get_fsdp_wrap_policy_none_config():
    from verl.utils.fsdp_utils import get_fsdp_wrap_policy

    # config=None -> {} ; no min params and a model with _no_split_modules
    assert get_fsdp_wrap_policy(_MockCausalLM(), config=None) is not None


def test_fsdp_version_plain_module_is_zero():
    from verl.utils.fsdp_utils import fsdp_version

    assert fsdp_version(nn.Linear(4, 4)) == 0


def test_get_shard_placement_fn():
    from verl.utils.fsdp_utils import get_shard_placement_fn

    fn = get_shard_placement_fn(fsdp_size=4)
    # a param whose dim 0 is divisible by 4
    placement = fn(torch.empty(8, 3))
    assert placement is not None
    # a param whose only divisible dim is not 0
    placement2 = fn(torch.empty(3, 8))
    assert placement2 is not None
    # nothing divisible -> falls back to Shard(0)
    placement3 = fn(torch.empty(3, 3))
    assert placement3 is not None


def test_offload_and_load_fsdp_optimizer_cpu():
    from verl.utils.fsdp_utils import load_fsdp_optimizer, offload_fsdp_optimizer

    model = nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # empty state -> early return path
    offload_fsdp_optimizer(opt)
    load_fsdp_optimizer(opt, device_id=torch.device("cpu"))
    # build optimizer state via a step
    loss = model(torch.randn(2, 8)).sum()
    loss.backward()
    opt.step()
    offload_fsdp_optimizer(opt)
    load_fsdp_optimizer(opt, device_id=torch.device("cpu"))


def test_meta_device_init_creates_meta_params():
    from verl.utils.fsdp_utils import meta_device_init

    with meta_device_init():
        m = nn.Linear(8, 8)
    assert m.weight.is_meta


def test_replace_lora_wrapper_passthrough():
    """Non-target keys pass through unchanged; target keys get base_layer suffix."""
    from verl.utils.fsdp_utils import replace_lora_wrapper

    try:
        from peft import LoraConfig

        peft_config = LoraConfig(r=8, target_modules=["q_proj"], task_type="CAUSAL_LM")
    except Exception as exc:  # pragma: no cover - peft must be present in the tier
        pytest.skip(f"peft unavailable: {exc}")

    # a non-weight/bias key returns unchanged
    assert replace_lora_wrapper("model.norm", peft_config) == "model.norm"
    # a stacked-param weight gets rewritten to base_layer
    out = replace_lora_wrapper("model.layers.0.self_attn.q_proj.weight", peft_config)
    assert out.endswith(".weight")


# ---------------------------------------------------------------------------
# Group 2 -- single-GPU FSDP2 path (world_size=1, isolated subprocess)
# ---------------------------------------------------------------------------


def _fsdp2_single_gpu_worker(rank, rendezvous_file):
    import torch.distributed as dist
    from torch.distributed import init_device_mesh
    from transformers import Qwen2Config
    from transformers import AutoModelForCausalLM

    from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
    from verl.utils.fsdp_utils import (
        MixedPrecisionPolicy,
        apply_fsdp2,
        fsdp2_clip_grad_norm_,
        fsdp_version,
        get_fsdp_full_state_dict,
        get_init_weight_context_manager,
        load_fsdp2_model_to_gpu,
        offload_fsdp2_model_to_cpu,
        set_reshard_after_forward,
    )

    world_size = 1
    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(),
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device_mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("dp",))

        # init-weight context manager (both branches)
        _ = get_init_weight_context_manager(use_meta_tensor=True, mesh=device_mesh)
        _ = get_init_weight_context_manager(use_meta_tensor=False)

        cfg = Qwen2Config(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=128,
            intermediate_size=256,
            vocab_size=512,
        )
        with torch.device(get_device_name()):
            model = AutoModelForCausalLM.from_config(
                config=cfg, torch_dtype=torch.bfloat16, attn_implementation="eager"
            )
            model = model.to(device=get_device_name())

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
        )
        apply_fsdp2(model, {"mesh": device_mesh, "mp_policy": mp_policy}, {})
        assert fsdp_version(model) == 2

        # reshard-after-forward toggle (the runtime override helper)
        try:
            set_reshard_after_forward(model, True)
        except Exception as exc:  # torch < 2.8 may lack the internal API
            print(f"set_reshard_after_forward non-fatal: {type(exc).__name__}: {exc}")

        # offload to CPU then load back to GPU
        offload_fsdp2_model_to_cpu(model)
        load_fsdp2_model_to_gpu(model)

        # full (gathered) state dict -- fsdp2 path
        sd = get_fsdp_full_state_dict(model, offload_to_cpu=True, rank0_only=True)
        assert isinstance(sd, dict) and len(sd) > 0

        # a tiny fwd/bwd to populate grads, then fsdp2 grad-norm clip
        input_ids = torch.randint(0, 512, (1, 8), device=get_device_name())
        out = model(input_ids=input_ids, labels=input_ids)
        out.loss.backward()
        total_norm = fsdp2_clip_grad_norm_(model.parameters(), max_norm=1.0)
        assert total_norm is not None
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_fsdp2_single_gpu_paths():
    if not torch.cuda.is_available():
        pytest.skip("no GPU visible")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")

    import tempfile

    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        rendezvous_file = os.path.join(tmp, "rdzv_fsdp2_single")
        mp.spawn(fn=_fsdp2_single_gpu_worker, args=(rendezvous_file,), nprocs=1, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
