# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest
import torch
import torch.nn as nn

from verl.workers.rollout.vllm_rollout.weight_update_utils import apply_buffer_updates, split_buffer_updates


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.register_buffer("running_mean", torch.zeros(2))


def test_split_buffer_updates_separates_params_and_buffers():
    model = _Toy()
    weights = [
        ("weight", torch.full((2, 2), 2.0)),
        ("running_mean", torch.full((2,), 3.0)),
        ("unknown", torch.ones(1)),
    ]
    params, buffers, named = split_buffer_updates(model, weights)
    assert [n for n, _ in params] == ["weight", "unknown"]
    assert [n for n, _ in buffers] == ["running_mean"]
    assert "running_mean" in named


def test_apply_buffer_updates_empty_is_zero():
    model = _Toy()
    assert apply_buffer_updates(model, []) == 0


def test_apply_buffer_updates_copies_matching_buffers():
    model = _Toy()
    n = apply_buffer_updates(model, [("running_mean", torch.arange(2, dtype=torch.float32))])
    assert n == 1
    assert torch.equal(model.running_mean, torch.arange(2, dtype=torch.float32))


def test_apply_buffer_updates_skips_unknown_and_uses_named_map():
    model = _Toy()
    named = dict(model.named_buffers())
    n = apply_buffer_updates(model, [("missing", torch.ones(2))], named_buffers=named)
    assert n == 0


def test_apply_buffer_updates_shape_mismatch_raises():
    model = _Toy()
    with pytest.raises(ValueError, match="Buffer shape mismatch"):
        apply_buffer_updates(model, [("running_mean", torch.zeros(4))])
