# Copyright 2026 The RLinf Authors.
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

"""LoRA (Low-Rank Adaptation) modules for PyTorch, aligned with JAX models/lora.py."""

from __future__ import annotations

import dataclasses
import math
import re

import torch
import torch.nn as nn

from .utils import gelu_glu


@dataclasses.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    rank: int
    alpha: float = 1.0
    rslora: bool = False
    axes: tuple[int, int] = (-2, -1)
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return (
            self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank
        )


class Einsum(nn.Module):
    """Einsum with LoRA support. Drop-in replacement for a linear einsum operation.

    Follows the JAX convention of storing weight shape explicitly and
    computing via einsum rather than standard matmul.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        eqn: str,
        init_fn: str = "lecun_normal",
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__()
        self.shape = shape
        self.eqn = eqn
        self.lora_config = lora_config

        # Initialize weight
        w = torch.empty(*shape)
        if init_fn == "lecun_normal":
            # lecun_normal: std = sqrt(1 / fan_in)
            # Weight shape is typically (out_features, in_features) or (num_heads, width, head_dim)
            # For the weight tensor, fan_in is the last dimension size
            nn.init.normal_(w, std=1.0 / math.sqrt(shape[-1]))
        elif init_fn == "zeros":
            nn.init.zeros_(w)
        elif init_fn == "normal":
            nn.init.normal_(w, std=0.01)
        else:
            nn.init.normal_(w, std=0.01)
        self.w = nn.Parameter(w)

        # LoRA parameters
        if lora_config is not None:
            axes = lora_config.axes
            rank = lora_config.rank
            shape_a = list(shape)
            shape_a[axes[1]] = rank
            shape_b = list(shape)
            shape_b[axes[0]] = rank

            w_a = torch.empty(*shape_a)
            w_b = torch.empty(*shape_b)
            nn.init.normal_(w_a, std=0.01)
            nn.init.normal_(w_b, std=0.01)
            self.w_a = nn.Parameter(w_a)
            self.w_b = nn.Parameter(w_b)
            self._eqn_a, self._eqn_b = self._make_lora_eqns(
                eqn, axes, lora_config.label
            )
        else:
            self.w_a = None
            self.w_b = None

    def _make_lora_eqns(
        self, eqn: str, axes: tuple[int, int], label: str
    ) -> tuple[str, str]:
        """Create einsum equations for LoRA computation."""
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        m = re.match(r"(.*),(.*)->(.*)", eqn)
        if not m:
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        a_label, b_label = rhs[axes[0]], rhs[axes[1]]

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b

    def forward(self, x: torch.Tensor, eqn: str | None = None) -> torch.Tensor:
        """Forward pass with optional LoRA."""
        if eqn is None:
            eqn = self.eqn

        # Parse equation: "input_spec,weight_spec->output_spec"
        result = torch.einsum(eqn, x, self.w.to(x.dtype))

        if self.lora_config is not None:
            lora = torch.einsum(self._eqn_a, x, self.w_a.to(x.dtype))
            lora = torch.einsum(self._eqn_b, lora, self.w_b.to(x.dtype))
            result = result + lora * self.lora_config.scaling_value

        return result


class HeadwiseLoRALinear(nn.Linear):
    """Linear base projection with the JAX Pi0 head-wise LoRA layout.

    The base ``weight`` stays an ``nn.Linear`` leaf, which keeps a converted
    PyTorch Pi0 checkpoint key-compatible with the non-LoRA model. The delta
    is deliberately not flattened over attention heads:

    * Q/K/V: ``A[head, input, rank] @ B[head, rank, head_dim]``
    * O: ``A[head, head_dim, rank] @ B[head, rank, output]``

    This matches OpenPI JAX ``lora.Einsum``. Both A and B use JAX's
    ``normal(stddev=0.01)`` initializer; a zero B would change checkpoint-0
    behavior and the training trajectory.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        target: str,
        num_heads: int,
        head_dim: int,
        lora_config: LoRAConfig,
        bias: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        if target not in {"q", "k", "v", "o"}:
            raise ValueError(f"Unsupported attention LoRA target: {target}")
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive.")

        self.target = target
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.lora_config = lora_config

        if target == "o":
            expected_input = num_heads * head_dim
            if in_features != expected_input:
                raise ValueError(
                    f"O projection expects {expected_input} input features, got {in_features}."
                )
            shape_a = (num_heads, head_dim, lora_config.rank)
            shape_b = (num_heads, lora_config.rank, out_features)
        else:
            expected_output = num_heads * head_dim
            if out_features != expected_output:
                raise ValueError(
                    f"{target.upper()} projection expects {expected_output} output features, got {out_features}."
                )
            shape_a = (num_heads, in_features, lora_config.rank)
            shape_b = (num_heads, lora_config.rank, head_dim)

        self.lora_a = nn.Parameter(torch.empty(shape_a))
        self.lora_b = nn.Parameter(torch.empty(shape_b))
        nn.init.normal_(self.lora_a, mean=0.0, std=0.01)
        nn.init.normal_(self.lora_b, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = super().forward(x)
        a = self.lora_a.to(dtype=x.dtype)
        b = self.lora_b.to(dtype=x.dtype)
        if self.target == "o":
            x_heads = x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)
            delta = torch.einsum("...he,her,hrd->...d", x_heads, a, b)
        else:
            delta = torch.einsum("...d,hdr,hre->...he", x, a, b)
            delta = delta.reshape(*x.shape[:-1], self.out_features)
        return base + delta * self.lora_config.scaling_value


class FeedForward(nn.Module):
    """FeedForward module with optional LoRA support.

    Uses gated FFN: gate(x) * linear(x) -> output.
    """

    def __init__(
        self,
        features: int,
        hidden_dim: int,
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.lora_config = lora_config

        # Gating weights: shape (2, features, hidden_dim)
        w_gating = torch.empty(2, features, hidden_dim)
        nn.init.normal_(w_gating, std=1.0 / math.sqrt(features))
        self.w_gating = nn.Parameter(w_gating)

        # Linear weights: shape (hidden_dim, features)
        w_linear = torch.empty(hidden_dim, features)
        nn.init.normal_(w_linear, std=1.0 / math.sqrt(hidden_dim))
        self.w_linear = nn.Parameter(w_linear)

        # LoRA parameters for gating
        if lora_config is not None:
            rank = lora_config.rank
            # LoRA for w_gating[0] and w_gating[1] separately
            self.w_gating_lora_a = nn.Parameter(torch.empty(2, features, rank))
            self.w_gating_lora_b = nn.Parameter(torch.empty(2, rank, hidden_dim))
            self.w_linear_lora_a = nn.Parameter(torch.empty(hidden_dim, rank))
            self.w_linear_lora_b = nn.Parameter(torch.empty(rank, features))
            nn.init.normal_(self.w_gating_lora_a, std=0.01)
            nn.init.normal_(self.w_gating_lora_b, std=0.01)
            nn.init.normal_(self.w_linear_lora_a, std=0.01)
            nn.init.normal_(self.w_linear_lora_b, std=0.01)
        else:
            self.w_gating_lora_a = None
            self.w_gating_lora_b = None
            self.w_linear_lora_a = None
            self.w_linear_lora_b = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gating and optional LoRA."""
        # Gate
        ff_gate = self._dot(x, self.w_gating[0], 0)

        # FF1
        ff1 = self._dot(x, self.w_gating[1], 1)
        activations = gelu_glu(ff_gate, ff1)

        # Output
        outputs = self._dot(activations, self.w_linear, -1)
        return outputs

    def _dot(self, x: torch.Tensor, w: torch.Tensor, index: int) -> torch.Tensor:
        """Dot product with optional LoRA."""
        base = torch.matmul(x, w.to(x.dtype))
        if self.lora_config is not None:
            scaling = self.lora_config.scaling_value
            if index == 0:
                lora = torch.matmul(
                    torch.matmul(x, self.w_gating_lora_a[0].to(x.dtype)),
                    self.w_gating_lora_b[0].to(x.dtype),
                )
            elif index == 1:
                lora = torch.matmul(
                    torch.matmul(x, self.w_gating_lora_a[1].to(x.dtype)),
                    self.w_gating_lora_b[1].to(x.dtype),
                )
            else:  # linear output
                lora = torch.matmul(
                    torch.matmul(x, self.w_linear_lora_a.to(x.dtype)),
                    self.w_linear_lora_b.to(x.dtype),
                )
            base = base + lora * scaling
        return base
