# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest


def test_critic_value_diagnostics():
    torch = pytest.importorskip("torch")
    from rlinf.algorithms.losses import compute_ppo_critic_loss

    values = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([1.0, 3.0, 5.0])
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=values,
        value_clip=0.2,
        huber_delta=10.0,
    )

    assert metrics["critic/value_target_mean"].item() == pytest.approx(3.0)
    assert metrics["critic/value_target_std"].item() == pytest.approx(
        (8.0 / 3.0) ** 0.5
    )
    assert metrics["critic/value_pred_mean"].item() == pytest.approx(2.0)
    assert metrics["critic/value_pred_std"].item() == pytest.approx((2.0 / 3.0) ** 0.5)
    assert metrics["critic/value_mse"].item() == pytest.approx(5.0 / 3.0)
    assert metrics["critic/value_mae"].item() == pytest.approx(1.0)
    assert metrics["critic/value_target_pred_corr"].item() == pytest.approx(1.0)


def test_critic_value_diagnostics_respect_loss_mask():
    torch = pytest.importorskip("torch")
    from rlinf.algorithms.losses import compute_ppo_critic_loss

    values = torch.tensor([1.0, 100.0, 5.0])
    returns = torch.tensor([2.0, -100.0, 8.0])
    loss_mask = torch.tensor([True, False, True])
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=values,
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=loss_mask,
    )

    assert metrics["critic/value_target_mean"].item() == pytest.approx(5.0)
    assert metrics["critic/value_target_std"].item() == pytest.approx(3.0)
    assert metrics["critic/value_pred_mean"].item() == pytest.approx(3.0)
    assert metrics["critic/value_pred_std"].item() == pytest.approx(2.0)
    assert metrics["critic/value_mse"].item() == pytest.approx(5.0)
    assert metrics["critic/value_mae"].item() == pytest.approx(2.0)
    assert metrics["critic/value_target_pred_corr"].item() == pytest.approx(1.0)


def test_critic_value_correlation_is_zero_for_constant_target():
    torch = pytest.importorskip("torch")
    from rlinf.algorithms.losses import compute_ppo_critic_loss

    values = torch.tensor([1.0, 2.0, 3.0])
    returns = torch.tensor([4.0, 4.0, 4.0])
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=values,
        value_clip=0.2,
        huber_delta=10.0,
    )

    assert metrics["critic/value_target_pred_corr"].item() == 0.0
