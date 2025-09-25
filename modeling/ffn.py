from typing import Optional

import torch
from torch import nn, Tensor
from einops import einsum, rearrange

from .utils import get_act_fn, RMSNorm


class FFN(nn.Module):

    def __init__(
        self,
        d_model: int,
        is_gated: bool = True,
        d_mid: Optional[int] = None,
        tie_kv: bool = False,
        act_name: str = "silu",
    ):
        super().__init__()
        self.is_gated = is_gated
        self.d_mid = d_mid
        self.tie_kv = tie_kv
        self.act_name = act_name
        self.d_model = d_model

        if d_mid is None:
            # If not specified, make sure the param count is 8 * d ^ 2
            if is_gated:
                d_mid = int(8 * d_model / 3)
            else:
                d_mid = 4 * d_model
        # Here, we use nn.Linear instead of nn.Parameter use the
        # default initialization method.
        if tie_kv:
            linear = nn.Linear(d_model, d_mid, bias=False)
            self.w = linear.weight
        else:
            self.w1 = nn.Linear(d_model, d_mid, bias=False)
            self.w2 = nn.Linear(d_mid, d_model, bias=False)

        if is_gated:
            self.gate_proj = nn.Linear(d_model, d_mid, bias=False)

        self.act_fn = get_act_fn(act_name)

    def forward(self, x: Tensor) -> Tensor:
        if self.is_gated:
            gate = self.gate_proj(x)

        if self.tie_kv:
            x = self.act_fn(einsum(x, self.w, "b t d, m d -> b t m"))
        else:
            x = self.act_fn(self.w1(x))

        if self.is_gated:
            x = x * gate

        if self.tie_kv:
            x = einsum(x, self.w, "b t m, m d -> b t d")
        else:
            x = self.w2(x)
        return x


class MHF(nn.Module):
    """
    A multi-headed version of FFN, keys and values are learnable neural weights, while the input
    is mapped to a multi-headed query. Each head operates on a subset of dimensions, and
    the heads are computed in parallel.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_k: int,
        dim_v: int,
        tie_kv: bool = False,
        d_mid: Optional[int] = None,
        use_q_proj: bool = True,
        use_o_proj: bool = True,
        use_q_norm: bool = True,
        use_output_norm: bool = True,
        head_mixing: bool = False,
        act_name: str = "silu",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mid = d_mid
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.tie_kv = tie_kv
        self.act_name = act_name
        self.head_mixing = head_mixing

        # TODO: Support head mixing.

        if d_mid is None:
            if use_q_proj and use_o_proj:
                d_mid = d_model * 3
            elif use_q_proj or use_o_proj:
                d_mid = int(d_model * 3.5)
            else:
                d_mid = d_model * 4

        total_dim_k = n_head * dim_k
        total_dim_v = n_head * dim_v
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.use_q_proj = use_q_proj
        self.use_o_proj = use_o_proj
        self.use_q_norm = use_q_norm
        self.use_output_norm = use_output_norm

        if self.use_output_norm:
            assert self.use_o_proj, "Must have o_proj to use output norm"
            self.head_norm = RMSNorm(dim_v, eps=1e-6)

        if self.use_q_norm:
            assert self.use_q_proj, "Must have q_proj to use q norm"
            self.q_norm = RMSNorm(dim_k, eps=1e-6)

        if use_q_proj:
            self.Wq = nn.Linear(d_model, total_dim_k, bias=False)
        else:
            assert total_dim_k == d_model

        if use_o_proj:
            self.Wo = nn.Linear(total_dim_v, d_model, bias=False)
        else:
            assert total_dim_v == d_model

        if tie_kv:
            assert (
                dim_k == dim_v
            ), f"K and V dimensions must match when tying their weights, got {dim_k=}, {dim_v=}"
            self.M = nn.Parameter(torch.zeros(n_head, d_mid, dim_k))
            in_dim = dim_k
            torch.nn.init.normal_(self.M.data, std=in_dim ** (-0.5))
        else:
            self.K = nn.Parameter(torch.zeros(n_head, d_mid, dim_k))
            self.V = nn.Parameter(torch.zeros(n_head, d_mid, dim_v))
            torch.nn.init.normal_(self.K.data, std=dim_k ** (-0.5))
            torch.nn.init.normal_(self.V.data, std=dim_v ** (-0.5))

        self.act_fn = get_act_fn(act_name=act_name)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seqlen, d_model = x.shape
        if self.use_q_proj:
            Q = self.Wq(x)  # (B, T, H * DK)
        else:
            Q = x
        if self.use_q_norm:
            Q = self.q_norm(Q)
        Q = rearrange(Q, "b t (h dk) -> b t h dk", h=self.n_head)
        if self.tie_kv:
            scores = einsum(Q, self.M, "b t h dk, h m dk -> b t h m")
            scores = self.act_fn(scores)
            out = einsum(scores, self.M, "b t h m, h m dv -> b t h dv")
        else:
            scores = einsum(Q, self.K, "b t h dk, h m dk -> b t h m")
            scores = self.act_fn(scores)
            out = einsum(scores, self.V, "b t h m, h m dv -> b t h dv")
        out = rearrange(out, "b t h dv -> b t (h dv)")
        if self.use_output_norm:
            out = self.head_norm(out)
        if self.use_o_proj:
            y = self.Wo(out)  # (b, t, d)
        else:
            y = out
        return y
