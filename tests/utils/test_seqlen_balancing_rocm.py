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
"""ROCm code-coverage tests for verl.utils.seqlen_balancing pure helpers.

Complements tests/utils/test_seqlen_balancing.py (which focuses on
rearrange_micro_batches / Karmarkar-Karp on the default single-sample path)
by covering the greedy partitioner, the imbalance-logging metric helper,
the equal_size Karmarkar path, and the force_group_size / no-dynamic-balance
branches of rearrange_micro_batches. All pure-python / CPU.
"""

import torch

from verl import DataProto
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    greedy_partition,
    log_seqlen_unbalance,
    rearrange_micro_batches,
)


def _covers_all_indices(partitions, n):
    seen = set()
    for p in partitions:
        seen.update(p)
    return seen == set(range(n))


def test_greedy_partition_unequal():
    seqlen = [10, 20, 30, 40, 50, 60]
    parts = greedy_partition(seqlen, k_partitions=3, equal_size=False)
    assert len(parts) == 3
    assert _covers_all_indices(parts, len(seqlen))


def test_greedy_partition_equal_size():
    seqlen = [10, 20, 30, 40, 50, 60]
    parts = greedy_partition(seqlen, k_partitions=3, equal_size=True)
    assert len(parts) == 3
    # equal_size -> each partition has the same number of items
    assert all(len(p) * 3 == len(seqlen) for p in parts)
    assert _covers_all_indices(parts, len(seqlen))


def test_karmarkar_karp_equal_size():
    seqlen = [5, 1, 8, 3, 9, 2, 7, 4]
    parts = get_seqlen_balanced_partitions(seqlen, k_partitions=2, equal_size=True)
    assert len(parts) == 2
    assert all(len(p) * 2 == len(seqlen) for p in parts)
    assert _covers_all_indices(parts, len(seqlen))


def test_log_seqlen_unbalance_metrics():
    seqlen = [10, 20, 30, 40]
    partitions = [[0, 3], [1, 2]]  # balanced: 50 / 50
    metrics = log_seqlen_unbalance(seqlen, partitions, prefix="seq")
    for suffix in ("min", "max", "minmax_diff", "balanced_min", "balanced_max", "mean"):
        assert f"seq/{suffix}" in metrics
    assert metrics["seq/balanced_min"] == 50
    assert metrics["seq/balanced_max"] == 50


def _make_batch(batch_size=8, seqlen=16):
    input_ids = torch.randint(low=1, high=50, size=(batch_size, seqlen))
    attention_mask = torch.ones(batch_size, seqlen, dtype=torch.long)
    # vary effective lengths so partitioning is non-trivial
    for i in range(batch_size):
        attention_mask[i, seqlen - (i % 4) :] = 0
    data = {"input_ids": input_ids, "attention_mask": attention_mask}
    return DataProto.from_single_dict(data).batch


def test_rearrange_micro_batches_force_group_size():
    batch = _make_batch(batch_size=8, seqlen=16)
    micro_batches, idx = rearrange_micro_batches(batch, max_token_len=64, force_group_size=2)
    # every group of 2 consecutive samples must stay together in one micro-batch
    for partition in idx:
        partition = sorted(partition)
        for j in range(0, len(partition), 2):
            assert partition[j] // 2 == partition[j + 1] // 2
    assert sum(len(p) for p in idx) == 8


def test_rearrange_micro_batches_no_dynamic_balance():
    batch = _make_batch(batch_size=8, seqlen=16)
    micro_batches, idx = rearrange_micro_batches(batch, max_token_len=64, use_dynamic_bsz_balance=False)
    assert sum(len(p) for p in idx) == 8


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
