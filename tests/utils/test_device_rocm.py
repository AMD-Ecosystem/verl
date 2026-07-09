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
"""Coverage for verl.utils.device on the ROCm (CUDA-compatible) path.

verl maps AMD/HIP onto torch's "cuda" device namespace, so on a ROCm host these
accessors exercise the is_cuda_available branches of the device abstraction --
the primary ROCm support surface. NPU-only helpers are out of scope (excluded in
.coveragerc.rocm). Falls back gracefully on a CPU-only host.
"""

import types

import pytest
import torch

from verl.utils import device as dev


def test_availability_flags():
    assert isinstance(dev.is_cuda_available, bool)
    assert isinstance(dev.is_npu_available, bool)
    assert dev.is_torch_npu_available(check_device=False) in (True, False)


def test_device_name_and_module():
    name = dev.get_device_name()
    assert name in ("cuda", "npu", "cpu")
    mod = dev.get_torch_device()
    assert mod is not None
    if dev.is_cuda_available:
        assert name == "cuda"
        assert mod is torch.cuda


def test_resource_and_visible_devices_keyword():
    assert dev.get_resource_name() in ("GPU", "NPU")
    kw = dev.get_visible_devices_keyword()
    assert kw in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
    if dev.is_cuda_available:
        assert kw == "CUDA_VISIBLE_DEVICES"


def test_nccl_backend():
    backend = dev.get_nccl_backend()
    assert backend in ("nccl", "hccl")
    if dev.is_cuda_available:
        assert backend == "nccl"


def test_device_id_and_capability():
    if not dev.is_cuda_available:
        pytest.skip("no accelerator visible")
    assert dev.get_device_id() >= 0
    major, minor = dev.get_device_capability(0)
    assert major is not None and minor is not None


def test_capability_when_no_cuda():
    # The (None, None) path is only taken without CUDA; just assert the contract.
    major, minor = dev.get_device_capability(0)
    if not dev.is_cuda_available:
        assert (major, minor) == (None, None)


def test_set_expandable_segments():
    # No-op on CPU; on ROCm/CUDA it forwards to the allocator. Must not raise.
    dev.set_expandable_segments(True)
    dev.set_expandable_segments(False)


def test_is_support_ipc():
    # On a GPU (ROCm/CUDA) host this is unconditionally True.
    result = dev.is_support_ipc()
    assert isinstance(result, bool)
    if dev.is_cuda_available:
        assert result is True


def test_auto_set_device_cuda_noop():
    # On a non-NPU host auto_set_device must leave a "cuda" trainer device alone.
    cfg = types.SimpleNamespace(trainer=types.SimpleNamespace(device="cuda"))
    dev.auto_set_device(cfg)
    if not dev.is_npu_available:
        assert cfg.trainer.device == "cuda"
    # None / missing-attr configs are tolerated.
    dev.auto_set_device(None)
    dev.auto_set_device(types.SimpleNamespace())
