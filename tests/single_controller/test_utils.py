# Copyright 2026 Advanced Micro Devices, Inc.
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
"""Shared helpers for the single_controller test suite."""

import logging
import time

import ray

logger = logging.getLogger(__name__)


def ray_init_with_retry(*, max_retries: int = 5, **kwargs):
    """`ray.init(**kwargs)` with retries for transient cluster-bootstrap flakiness.

    On CI, these tests each run in their own pytest process, calling `ray.init()`
    fresh to bootstrap a brand-new local Ray cluster (GCS + raylet + object store).
    Under load from many back-to-back test processes on the same node, that
    bootstrap handshake occasionally times out, e.g.:

        Exception: The current node timed out during startup. This could happen
        because some of the raylet failed to startup or the GCS has become
        overloaded.

    even though the very next `ray.init()` call moments later succeeds with no
    code change in between -- i.e. it's infra flakiness, not a test bug. This
    mirrors the EADDRINUSE retry/backoff already used in
    `verl.utils.distributed.initialize_global_process_group_ray`.
    """
    for attempt in range(max_retries):
        try:
            return ray.init(**kwargs)
        except Exception as e:
            is_startup_timeout = "timed out during startup" in str(e)
            if is_startup_timeout and attempt < max_retries - 1:
                wait = 2**attempt  # 1, 2, 4, 8, 16 s
                logger.warning(
                    "ray_init_with_retry: Ray node startup timed out on attempt %d/%d, "
                    "retrying in %ds: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                try:
                    if ray.is_initialized():
                        ray.shutdown()
                except Exception:
                    pass
                time.sleep(wait)
                continue
            raise
