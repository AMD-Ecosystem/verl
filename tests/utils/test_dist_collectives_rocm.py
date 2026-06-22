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
"""Multi-rank (world_size=2) ROCm coverage for the distributed collectives in
verl.utils.torch_functional: broadcast_dict_tensor, allgather_dict_tensors
(dict + TensorDict), allgather_dict_into_dict, distributed_mean_max_min_std and
distributed_masked_mean. These need a real process group + NCCL/RCCL, so they
run in the multi-GPU coverage tier (>=2 visible GPUs) via mp.spawn.
"""

import os
import tempfile

import pytest
import torch


def _worker(rank, world_size, rendezvous_file):
    import torch.distributed as dist
    from tensordict import TensorDict

    from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
    from verl.utils.torch_functional import (
        allgather_dict_into_dict,
        allgather_dict_tensors,
        broadcast_dict_tensor,
        distributed_masked_mean,
        distributed_mean_max_min_std,
    )

    get_torch_device().set_device(rank)
    dist.init_process_group(
        backend=get_nccl_backend(), init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size
    )
    dev = f"{get_device_name()}:{rank}"
    try:
        # broadcast_dict_tensor: rank0 values win on all ranks
        td = TensorDict({"a": torch.full((2, 3), float(rank), device=dev)}, batch_size=[2])
        broadcast_dict_tensor(td, src=0, group=dist.group.WORLD)
        assert torch.allclose(td["a"], torch.zeros(2, 3, device=dev))

        # allgather_dict_tensors over a TensorDict -> batch dim multiplied by world
        td2 = TensorDict({"x": torch.arange(3, device=dev).float().view(3, 1) + rank}, batch_size=[3])
        out = allgather_dict_tensors(td2, size=world_size, group=dist.group.WORLD, dim=0)
        assert out.batch_size[0] == 3 * world_size

        # allgather_dict_tensors over a plain dict
        plain = {"y": torch.ones(2, 2, device=dev) * rank}
        out_plain = allgather_dict_tensors(plain, size=world_size, group=dist.group.WORLD, dim=0)
        assert out_plain["y"].shape[0] == 2 * world_size

        # allgather_dict_into_dict -> dict of lists across ranks
        gathered = allgather_dict_into_dict({"loss": float(rank), "step": rank}, group=dist.group.WORLD)
        assert len(gathered["loss"]) == world_size and len(gathered["step"]) == world_size

        # distributed statistics
        local = torch.arange(4, device=dev).float() + rank * 4
        mean, mx, mn, std = distributed_mean_max_min_std(local)
        assert mean is not None and mx is not None and mn is not None and std is not None
        # disabled-metric branches
        mean2, mx2, mn2, std2 = distributed_mean_max_min_std(
            local, compute_max=False, compute_min=False, compute_std=False
        )
        assert mx2 is None and mn2 is None and std2 is None

        mask = torch.tensor([1.0, 0.0, 1.0, 0.0], device=dev)
        gm = distributed_masked_mean(local, mask)
        assert gm is not None
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_dist_collectives():
    # NOTE: deliberately avoid the substring "distributed" in the file/function
    # name -- the coverage gate runs with `-k 'not distributed'` to drop the
    # upstream torchrun-only tests, and this mp.spawn test is self-contained.
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("needs >=2 GPUs")
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as tmp:
        rendezvous_file = os.path.join(tmp, "rdzv_dist_tf")
        mp.spawn(fn=_worker, args=(2, rendezvous_file), nprocs=2, join=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
