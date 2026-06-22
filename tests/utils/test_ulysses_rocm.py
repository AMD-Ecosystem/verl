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
"""Multi-GPU (world_size=2, sp_size=2) ROCm coverage for verl.utils.ulysses.

Exercises the Ulysses sequence-parallel collectives end-to-end on a real
process group: the all-to-all (SeqAllToAll) and all-gather (Gather) autograd
functions (forward + backward), pad/slice helpers, gather_outputs_and_unpad,
and the FSDPUlyssesShardingManager pre/post-process. Runs in the multi-GPU
coverage tier (>=2 visible GPUs) via mp.spawn.
"""

import os
import tempfile

import pytest
import torch


def _worker(rank, world_size, rendezvous_file):
    import torch.distributed as dist
    from torch.distributed import init_device_mesh

    from verl import DataProto
    from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
    from verl.utils import ulysses as U

    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(), init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size
    )
    dev = f"{get_device_name()}:{rank}"
    try:
        U.set_ulysses_sequence_parallel_group(dist.group.WORLD)
        assert U.get_ulysses_sequence_parallel_group() is not None
        assert U.get_ulysses_sequence_parallel_world_size() == world_size
        assert U.get_ulysses_sequence_parallel_rank() == rank

        # validate_ulysses_config: valid + invalid
        U.validate_ulysses_config(num_heads=4, ulysses_sequence_size=2)
        with pytest.raises(AssertionError):
            U.validate_ulysses_config(num_heads=3, ulysses_sequence_size=2)

        # ulysses_pad: sp_size<=1 no-op, and padding branch
        ii = torch.arange(5, device=dev).view(1, 5)
        pos = torch.arange(5, device=dev).view(1, 1, 5)
        _, _, ps0 = U.ulysses_pad(ii, pos, sp_size=1)
        assert ps0 == 0
        padded_ii, padded_pos, ps = U.ulysses_pad(ii, pos, sp_size=world_size)
        assert padded_ii.size(-1) % world_size == 0

        # ulysses_pad_and_slice_inputs -> each rank owns a slice
        sliced_ii, sliced_pos, ps2 = U.ulysses_pad_and_slice_inputs(ii, pos, sp_size=world_size)
        assert sliced_ii.size(-1) == padded_ii.size(-1) // world_size

        # slice_input_tensor (with padding)
        sl = U.slice_input_tensor(torch.arange(6, device=dev).view(1, 6).float(), dim=1, padding=True)
        assert sl.size(1) == 6 // world_size

        # all_to_all_tensor + all_gather_tensor (raw collectives)
        x = torch.randn(2, 4 * world_size, device=dev)
        a2a = U.all_to_all_tensor(x, scatter_dim=1, gather_dim=0, async_op=False)
        assert a2a.is_contiguous()
        ag = U.all_gather_tensor(torch.randn(2, 3, device=dev))
        assert ag.shape[0] == 2 * world_size

        # gather_seq_scatter_heads / gather_heads_scatter_seq round trip with backward
        # x: [bsz, seq/n, h, d] -> gather seq, scatter heads
        h = torch.randn(2, 4, 2 * world_size, 8, device=dev, requires_grad=True)
        gathered = U.gather_seq_scatter_heads(h, seq_dim=1, head_dim=2, unpadded_dim_size=4 * world_size)
        back = U.gather_heads_scatter_seq(gathered, head_dim=2, seq_dim=1)
        back.sum().backward()
        assert h.grad is not None

        # gather_outputs_and_unpad: padding_size==0 fast path and >0 unpad path, plus backward
        y = torch.randn(world_size, 5, device=dev, requires_grad=True)
        g0 = U.gather_outputs_and_unpad(y, gather_dim=0, unpad_dim=None)
        assert g0.shape[0] == world_size * world_size
        g1 = U.gather_outputs_and_unpad(y, gather_dim=0, unpad_dim=1, padding_size=0)
        assert g1 is not None
        g2 = U.gather_outputs_and_unpad(y, gather_dim=0, unpad_dim=1, padding_size=1)
        g2.sum().backward()
        assert y.grad is not None

        # the deprecated-typo alias must raise
        with pytest.raises(RuntimeError):
            U.gather_outpus_and_unpad(y)

        # group=None short-circuits (no sp group): returns input unchanged
        U.set_ulysses_sequence_parallel_group(None)
        same = U.gather_seq_scatter_heads(torch.randn(2, 2, device=dev), seq_dim=0, head_dim=1)
        assert same.shape == (2, 2)
        U.set_ulysses_sequence_parallel_group(dist.group.WORLD)

        # FSDPUlyssesShardingManager (device_mesh with an "sp" dim)
        mesh = init_device_mesh(get_device_name(), mesh_shape=(1, world_size), mesh_dim_names=("dp", "sp"))
        mgr = U.FSDPUlyssesShardingManager(mesh)
        data = DataProto.from_dict({"input_ids": torch.arange(8, device=dev).view(4, 2)})
        with mgr:
            data = mgr.preprocess_data(data)  # all_gather across sp
            data = mgr.postprocess_data(data)  # chunk back to this rank
        assert data is not None

        # BaseShardingManager no-op paths
        base = U.BaseShardingManager()
        with base:
            assert base.preprocess_data(data) is data
            assert base.postprocess_data(data) is data
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_ulysses_multi_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    pytest.importorskip("flash_attn")
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        rendezvous_file = os.path.join(tmp, "rdzv_ulysses")
        mp.spawn(fn=_worker, args=(2, rendezvous_file), nprocs=2, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
