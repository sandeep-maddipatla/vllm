# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU model runner selection.

An encoder-only instance (`--mm-encoder-only`) builds the language model on the
meta device, so it must get the encoder-only runner instead of the shared V2
one -- the same choice `vllm.v1.worker.gpu_worker` makes for CUDA. Running the
shared runner instead fails during `profile_run` on meta tensors.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunnerV1
from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner
from vllm.v1.worker.xpu_model_runner import get_xpu_model_runner_cls

pytestmark = pytest.mark.cpu_test


def _config(use_v2_model_runner: bool, is_mm_encoder_only: bool) -> SimpleNamespace:
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        is_mm_encoder_only=is_mm_encoder_only,
    )


def test_encoder_only_uses_the_encoder_only_runner():
    runner_cls = get_xpu_model_runner_cls(_config(True, True))
    assert issubclass(runner_cls, MMEncoderModelRunner)


def test_other_models_use_the_shared_v2_runner():
    runner_cls = get_xpu_model_runner_cls(_config(True, False))
    assert issubclass(runner_cls, GPUModelRunnerV2)
    assert not issubclass(runner_cls, MMEncoderModelRunner)


@pytest.mark.parametrize("is_mm_encoder_only", [False, True])
def test_v2_disabled_uses_the_v1_runner(is_mm_encoder_only: bool):
    runner_cls = get_xpu_model_runner_cls(_config(False, is_mm_encoder_only))
    assert issubclass(runner_cls, GPUModelRunnerV1)
