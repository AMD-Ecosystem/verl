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
"""ROCm code-coverage tests for verl.workers.rollout.vllm_rollout.utils.

Targets the pure-python helpers that do NOT need a live vLLM worker / ZMQ
transport: CLI-arg building, prompt-logprob extraction, LoRA rank rounding,
the ROCm device-uuid fallback, the compute-logits monkeypatch, and the
signal-suppression context manager. The heavy IPC weight-update path
(update_weights_from_ipc) is exercised separately by the rollout smoke test.

Importing this module needs vllm present (the rollout utils import from
vllm.outputs), which the rocm/vllm coverage tier provides; the whole module
skips cleanly otherwise.
"""

import signal
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from verl.workers.rollout.vllm_rollout.utils import (  # noqa: E402
    build_cli_args_from_config,
    extract_prompt_logprobs,
    get_device_uuid,
    get_vllm_max_lora_rank,
    monkey_patch_compute_logits,
    SuppressSignalInThread,
)


def test_build_cli_args_from_config_all_branches():
    cfg = {
        "skip_none": None,  # dropped
        "flag_true": True,  # -> --flag-true
        "flag_false": False,  # dropped
        "empty_list": [],  # dropped
        "sizes": [1, 2, 4],  # expanded
        "json_dict": {"a": 1},  # json serialized
        "scalar": "hello",  # --scalar hello
        "number": 7,
    }
    args = build_cli_args_from_config(cfg)
    assert "--skip_none" not in args
    assert "--flag_true" in args
    assert "--flag_false" not in args
    assert "--empty_list" not in args
    # list expansion
    i = args.index("--sizes")
    assert args[i + 1 : i + 4] == ["1", "2", "4"]
    # dict -> json
    j = args.index("--json_dict")
    assert args[j + 1] == '{"a": 1}'
    # scalar / number
    assert args[args.index("--scalar") + 1] == "hello"
    assert args[args.index("--number") + 1] == "7"


def test_get_vllm_max_lora_rank_rounds_up():
    # rounds up to the nearest allowed rank
    assert get_vllm_max_lora_rank(4) >= 4
    assert get_vllm_max_lora_rank(8) >= 8


def test_get_vllm_max_lora_rank_invalid():
    with pytest.raises(AssertionError):
        get_vllm_max_lora_rank(0)
    with pytest.raises(ValueError):
        get_vllm_max_lora_rank(100000)


def _lp(logprob, rank):
    return SimpleNamespace(logprob=logprob, rank=rank)


def test_extract_prompt_logprobs_none_is_noop():
    result = {}
    extract_prompt_logprobs(SimpleNamespace(prompt_logprobs=None), None, result)
    assert result == {}


def test_extract_prompt_logprobs_zero():
    # num_prompt_logprobs == 0 -> single top token per position
    out = SimpleNamespace(prompt_logprobs=[None, {"5": _lp(-0.1, 1)}, {"7": _lp(-0.2, 1)}])
    result = {}
    extract_prompt_logprobs(out, 0, result)
    assert "prompt_ids" in result and "prompt_logprobs" in result
    assert result["prompt_ids"][0] == [5]


def test_extract_prompt_logprobs_topk():
    out = SimpleNamespace(
        prompt_logprobs=[
            None,
            {"5": _lp(-0.1, 1), "9": _lp(-0.5, 2)},
            {"7": _lp(-0.2, 1), "3": _lp(-0.9, 2)},
        ]
    )
    result = {}
    extract_prompt_logprobs(out, 2, result)
    assert result["prompt_ids"][0] == [5, 9]


def test_get_device_uuid_returns_string():
    # On ROCm the vLLM platform raises NotImplementedError and the fallback
    # derives a stable id from the visible-device env. Assert it produces a
    # non-empty identifier rather than a specific value (varies by build).
    import os

    prev = os.environ.get("HIP_VISIBLE_DEVICES")
    os.environ["HIP_VISIBLE_DEVICES"] = "0,1"
    try:
        uuid = get_device_uuid(0)
    finally:
        if prev is None:
            os.environ.pop("HIP_VISIBLE_DEVICES", None)
        else:
            os.environ["HIP_VISIBLE_DEVICES"] = prev
    assert isinstance(uuid, str) and len(uuid) > 0


def test_monkey_patch_compute_logits_masks_oov():
    import torch

    class _FakeModel:
        def compute_logits(self, *args, **kwargs):
            return torch.zeros(1, 8)

    model = _FakeModel()
    monkey_patch_compute_logits(model, vocab_size=4)
    logits = model.compute_logits()
    # tokens >= vocab_size masked to -inf
    assert torch.isinf(logits[..., 4:]).all()
    assert not torch.isinf(logits[..., :4]).any()


def test_extract_prompt_logprobs_topk_skips_out_of_rank():
    # An entry whose rank exceeds num_prompt_logprobs (the sampled token not in
    # the top-k) must be skipped via `continue`.
    out = SimpleNamespace(
        prompt_logprobs=[
            None,
            {"5": _lp(-0.1, 1), "9": _lp(-0.5, 2), "2": _lp(-3.0, 3)},
        ]
    )
    result = {}
    extract_prompt_logprobs(out, 2, result)
    # only the two in-rank tokens are kept; the rank-3 entry is dropped
    assert result["prompt_ids"][0] == [5, 9]


def test_set_death_signal_non_linux_is_noop(monkeypatch):
    import platform

    from verl.workers.rollout.vllm_rollout.utils import set_death_signal

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    # Should return immediately without touching libc.
    set_death_signal()


def _call_set_death_signal_child():
    # Runs in a forked child so the PR_SET_PDEATHSIG / potential self-SIGKILL
    # (when ppid == 1) can never take down the pytest process itself.
    from verl.workers.rollout.vllm_rollout.utils import set_death_signal

    set_death_signal()


def test_set_death_signal_linux_runs():
    import platform

    if platform.system() != "Linux":
        pytest.skip("Linux-only path")
    import multiprocessing as mp

    # IMPORTANT: never call set_death_signal() in-process. Under the coverage
    # container the entrypoint is PID 1, so the test process's ppid is 1 and
    # set_death_signal() would os.kill(self, SIGKILL). Isolate it in a child
    # (whose ppid is this process, not 1) so the prctl path is covered safely.
    ctx = mp.get_context("fork")
    p = ctx.Process(target=_call_set_death_signal_child)
    p.start()
    p.join(timeout=30)
    assert p.exitcode == 0


def test_get_device_uuid_torch_props_fallback():
    # With no visible-device env var set, the ROCm fallback drops past the env
    # loop into the torch device-properties / final-id path.
    import os

    saved = {k: os.environ.pop(k, None) for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    try:
        uuid = get_device_uuid(0)
        assert isinstance(uuid, str) and uuid.startswith("ROCm-")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_suppress_signal_in_thread():
    with SuppressSignalInThread():
        # registering a signal handler from a non-main thread is a no-op
        done = {}

        def worker():
            try:
                signal.signal(signal.SIGUSR1, signal.SIG_IGN)
                done["ok"] = True
            except Exception as exc:  # pragma: no cover
                done["err"] = exc

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert done.get("ok") is True
    # signal.signal is restored after the context: registering from the main
    # thread works normally again.
    prev = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGUSR1, prev)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
