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
"""Device-side coverage for verl.utils.torch_functional (code-coverage tier).

These exercise the pure tensor / scheduler helpers in torch_functional on the
active accelerator device (HIP on ROCm), including the flash-attn cross-entropy
path used by logprobs_from_logits. They are single-process and need only one
GPU, so they are safe in the standalone ROCm coverage gate. They fall back to
CPU when no accelerator is visible.
"""

import math

import pytest
import torch
import torch.nn as nn

from verl.utils import torch_functional as tf


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


DEV = _device()


def test_gather_from_labels():
    data = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], device=DEV)
    label = torch.tensor([2, 0], device=DEV)
    out = tf.gather_from_labels(data, label)
    torch.testing.assert_close(out, torch.tensor([0.3, 0.4], device=DEV))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_logprobs_from_logits_paths(dtype):
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 11, device=DEV, dtype=dtype)
    labels = torch.randint(0, 11, (4, 7), device=DEV)

    # public entry (flash-attn path on GPU, v2 path on CPU)
    lp = tf.logprobs_from_logits(logits, labels)
    assert lp.shape == (4, 7)

    # naive + v2 reference paths
    naive = tf.logprobs_from_logits_naive(logits.float(), labels)
    v2 = tf.logprobs_from_logits_v2(logits.float(), labels)
    torch.testing.assert_close(naive, v2, atol=1e-4, rtol=1e-4)


def test_clip_by_value():
    x = torch.tensor([-2.0, 0.0, 5.0], device=DEV)
    lo = torch.tensor([-1.0, -1.0, -1.0], device=DEV)
    hi = torch.tensor([1.0, 1.0, 1.0], device=DEV)
    out = tf.clip_by_value(x, lo, hi)
    torch.testing.assert_close(out, torch.tensor([-1.0, 0.0, 1.0], device=DEV))


def test_entropy_from_logits_and_chunking():
    torch.manual_seed(1)
    logits = torch.randn(5, 13, device=DEV)
    ent = tf.entropy_from_logits(logits)
    ent_chunked = tf.entropy_from_logits_with_chunking(logits, chunk_size=2)
    assert ent.shape == (5,)
    torch.testing.assert_close(ent, ent_chunked, atol=1e-4, rtol=1e-4)
    # entropy is non-negative
    assert torch.all(ent >= -1e-5)


def test_masked_sum_mean_var_whiten():
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=DEV)
    mask = torch.tensor([1.0, 1.0, 0.0, 1.0], device=DEV)
    assert tf.masked_sum(values, mask).item() == pytest.approx(7.0)
    assert tf.masked_mean(values, mask).item() == pytest.approx(7.0 / 3.0, abs=1e-4)

    var = tf.masked_var(values, mask)
    assert var.item() > 0
    whit = tf.masked_whiten(values, mask)
    assert whit.shape == values.shape
    # shift_mean=False re-adds the mean
    whit_nomean = tf.masked_whiten(values, mask, shift_mean=False)
    assert whit_nomean.shape == values.shape


def test_masked_var_error_paths():
    values = torch.tensor([1.0, 2.0], device=DEV)
    with pytest.raises(ValueError):
        tf.masked_var(values, torch.tensor([0.0, 0.0], device=DEV))
    with pytest.raises(ValueError):
        tf.masked_var(values, torch.tensor([1.0, 0.0], device=DEV))


@pytest.mark.parametrize("eos", [1, [1, 2]])
def test_get_response_mask(eos):
    response = torch.tensor(
        [[20, 10, 34, 1, 0, 0, 0], [78, 0, 76, 2, 1, 0, 0]],
        device=DEV,
    )
    mask = tf.get_response_mask(response, eos_token=eos)
    assert mask.shape == response.shape
    # every row keeps at least its first token
    assert torch.all(mask[:, 0] == 1)


def test_pad_helpers():
    padded = tf.pad_2d_list_to_length([[1, 2], [3]], pad_token_id=0, max_length=4)
    assert padded.tolist() == [[1, 2, 0, 0], [3, 0, 0, 0]]

    t = torch.ones(2, 3, device=DEV)
    right = tf.pad_sequence_to_length(t, max_seq_len=5, pad_token_id=0)
    assert right.shape == (2, 5) and right[:, 3:].sum() == 0
    left = tf.pad_sequence_to_length(t, max_seq_len=5, pad_token_id=0, left_pad=True)
    assert left.shape == (2, 5) and left[:, :2].sum() == 0
    # no-op when already long enough
    assert tf.pad_sequence_to_length(t, max_seq_len=2, pad_token_id=0).shape == (2, 3)


@pytest.mark.parametrize("truncation", ["left", "right", "middle"])
def test_postprocess_data_truncation(truncation):
    ids = torch.arange(2 * 8, device=DEV).reshape(2, 8)
    mask = torch.ones(2, 8, device=DEV, dtype=torch.long)
    out_ids, out_mask = tf.postprocess_data(ids, mask, max_length=4, pad_token_id=0, truncation=truncation)
    assert out_ids.shape == (2, 4) and out_mask.shape == (2, 4)


def test_postprocess_data_pad_and_error():
    ids = torch.arange(2 * 3, device=DEV).reshape(2, 3)
    mask = torch.ones(2, 3, device=DEV, dtype=torch.long)
    out_ids, out_mask = tf.postprocess_data(ids, mask, max_length=6, pad_token_id=0, left_pad=True)
    assert out_ids.shape == (2, 6)
    with pytest.raises(NotImplementedError):
        tf.postprocess_data(ids, mask, max_length=2, pad_token_id=0, truncation="error")


def test_remove_pad_token():
    ids = torch.tensor([[5, 6, 7], [8, 9, 0]], device=DEV)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], device=DEV)
    out = tf.remove_pad_token(ids, mask)
    assert out == [[5, 6, 7], [9, 0]]


def test_log_probs_from_logits_response():
    torch.manual_seed(2)
    input_ids = torch.randint(0, 11, (2, 9), device=DEV)
    logits = torch.randn(2, 9, 11, device=DEV)
    out = tf.log_probs_from_logits_response(input_ids, logits, response_length=4)
    assert out.shape == (2, 4)


def test_post_process_logits():
    logits = torch.randn(2, 5, device=DEV)
    out = tf.post_process_logits(None, logits.clone(), temperature=2.0, top_k=0, top_p=1.0)
    assert out.shape == logits.shape


def test_calculate_sum_pi_squared():
    logits = torch.randn(3, 8, device=DEV)
    out = tf.calculate_sum_pi_squared_from_logits(logits)
    expected = torch.softmax(logits, dim=-1).pow(2).sum(dim=-1)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_attention_mask_helpers():
    bsz, tgt = 2, 5
    causal = tf._make_causal_mask((bsz, tgt), torch.float32, DEV)
    assert causal.shape == (bsz, 1, tgt, tgt)
    attn = torch.ones(bsz, tgt, device=DEV)
    expanded = tf._expand_mask(attn, torch.float32, tgt_len=tgt)
    assert expanded.shape == (bsz, 1, tgt, tgt)
    embeds = torch.zeros(bsz, tgt, 4, device=DEV)
    combined = tf.prepare_decoder_attention_mask(attn, (bsz, tgt), embeds)
    assert combined.shape == (bsz, 1, tgt, tgt)


def test_get_unpad_data():
    attn = torch.tensor([[1, 1, 0], [1, 1, 1]], device=DEV)
    indices, cu_seqlens, max_seqlen = tf.get_unpad_data(attn)
    assert max_seqlen == 3
    assert cu_seqlens.tolist() == [0, 2, 5]
    assert indices.numel() == 5


@pytest.mark.parametrize(
    "factory",
    [
        lambda opt: tf.get_cosine_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=10),
        lambda opt: tf.get_cosine_schedule_with_warmup(
            opt, num_warmup_steps=2, num_training_steps=10, min_lr_ratio=0.1, zero_indexed_step=False
        ),
        lambda opt: tf.get_constant_schedule_with_warmup(opt, num_warmup_steps=3),
        lambda opt: tf.get_wsd_schedule_with_warmup(opt, num_warmup_steps=2, num_training_steps=10, stable_ratio=0.5),
    ],
)
def test_lr_schedules(factory):
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = factory(opt)
    lrs = []
    for _ in range(12):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert all(math.isfinite(lr) and lr >= 0 for lr in lrs)


def test_check_device_is_available():
    if torch.cuda.is_available():
        with tf.check_device_is_available():
            pass
    else:
        pytest.skip("no accelerator visible")


def test_compute_grad_norm():
    model = nn.Linear(3, 1)
    x = torch.randn(4, 3)
    model(x).sum().backward()
    gn = tf.compute_grad_norm(model)
    assert gn >= 0
